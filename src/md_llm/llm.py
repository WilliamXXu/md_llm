"""Lightweight LLM clients for post-processing transcripts (summarize, etc.).

Five providers are supported:
  - Ollama: a local server reachable over HTTP (default).
  - OpenRouter: a hosted API keyed by OPENROUTER_API_KEY.
  - OpenAI: a generic OpenAI-compatible API keyed by OPENAI_API_KEY. Point
    its endpoint at any OpenAI-compatible host (OpenAI itself, Groq, Together,
    ...) and type a model name. Speaks the same /chat/completions wire format
    as OpenRouter, minus OpenRouter's attribution headers.
  - OpenCode: the open source coding AGENT, invoked as a subprocess
    (`opencode run --format json --auto`). Not a plain chat API — with --auto it
    can run tools (bash/read/edit/...) in a working directory. Auth + model
    routing are OpenCode's own; the chat panel streams its JSONL event output.
  - Cline: the Cline coding AGENT CLI, also invoked as a subprocess
    (`cline --json "prompt"`). Like OpenCode it runs tools (auto-approved) in a
    working directory; auth + model routing are Cline's own (`cline auth`), and
    the chat panel streams its NDJSON event output.

Uses only the standard library (urllib + json + subprocess) to match the
no-extra-deps style of transcribe_local.py's remote-Whisper client.
"""

import os
import json
import re
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request

from . import sandbox

DEFAULT_ENDPOINT = "http://127.0.0.1:11434"
DEFAULT_MODEL = ""
# Default instruction for the Transcripts & LLM and Autopilot panels. The chat
# tab hides its instruction field (the prompt comes from the chat box), so this
# default applies to those two panels only.
DEFAULT_INSTRUCTION = (
    "概括以下文本（如果输入是英文就用英文回答，"
    "如果输入是其他语言就用简体中文回答）："
)
REQUEST_TIMEOUT = 600  # seconds; generation can take a while on long transcripts

# A non-default User-Agent. Some OpenAI-compatible hosts (e.g. Groq) sit behind
# Cloudflare, which blocks the default "Python-urllib/<ver>" signature with
# HTTP 403 (Cloudflare error 1010). Any descriptive UA passes the WAF.
USER_AGENT = "local-transcriber/1.0 (python-urllib)"

# OpenRouter defaults. The API key is read from the environment so the UI can
# prefill it without forcing the user to paste it in every session.
OPENROUTER_DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"

# Generic OpenAI-compatible defaults. This provider speaks the same
# /chat/completions wire format as OpenRouter but is unbranded: no
# HTTP-Referer/X-Title attribution headers, and the API key defaults to the
# OPENAI_API_KEY env var. Point the endpoint at any OpenAI-compatible host
# (OpenAI itself, Groq's https://api.groq.com/openai/v1, Together, etc.) and
# type the model name. Models are free-form — these hosts don't expose
# Ollama's /api/tags discovery, so there is no auto-populated dropdown.
OPENAI_DEFAULT_ENDPOINT = "https://api.openai.com/v1"
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"

# OpenCode (the open source coding agent) defaults. Unlike the providers above,
# OpenCode is not an LLM API — it's an agent CLI invoked as a subprocess
# (`opencode run --format json --auto`). Auth + model routing are OpenCode's
# own (configure via `opencode auth login` / env). Model ids are
# `provider/model`, discoverable via `opencode models`.
OPENCODE_BIN = "opencode"
OPENCODE_DEFAULT_MODEL = ""

# Reasoning-effort labels opencode uses as model variant names, ordered least →
# most effort. `opencode models --verbose` reports each model's variants as a
# map of name → provider options; the names come from this closed set, the
# metadata carries no ordering, and in practice each name equals its variant's
# effort value — so this table is what "highest effort" means. Matches the
# catalog's own ascending presentation order (…, high, xhigh, max).
OPENCODE_EFFORT_ORDER = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]

# Fallback variant dropdown options for when per-model discovery is unavailable
# (older opencode without `models --verbose`): every effort label that is a
# valid `--variant` value, least → most effort. The UI also lets the user type
# a custom one.
OPENCODE_VARIANTS = [v for v in OPENCODE_EFFORT_ORDER if v != "none"]

# Cline (the Cline coding agent CLI) defaults. Like OpenCode it is an agent
# invoked as a subprocess (`cline --json "prompt"`, tools auto-approved in the
# working directory), not an LLM API. Auth + model routing are Cline's own
# (configure via `cline auth`); passing no --model uses the model Cline was
# configured with. Model ids are free-form (e.g. "z-ai/glm-5.3-flash"); the
# CLI itself has no non-interactive model listing, but Cline's provider API
# exposes a public OpenAI-style catalog at {base}/models (no auth) whose ids
# carry a ":free" suffix on the zero-cost models — the same convention as
# OpenRouter's catalog, so the UI fetches the free ids for its dropdown.
CLINE_BIN = "cline"
CLINE_API_ENDPOINT = "https://api.cline.bot/api/v1"

# The reasoning-effort levels `cline --thinking` accepts, least → most effort.
# A closed set advertised by the CLI itself, so no discovery subprocess is
# needed; omitting the flag leaves the provider's own default.
CLINE_THINKING_LEVELS = ["none", "low", "medium", "high", "xhigh"]

# ANSI colour escapes cline's own CLI errors carry (stripped before the raw
# stderr tail is surfaced to the user).
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def list_cline_models(endpoint=CLINE_API_ENDPOINT, timeout=30):
    """Return Cline's free model ids from its public provider-API catalog.

    ``GET {endpoint}/models`` is public (no auth) and lists every model
    routable through Cline's own provider; only the ``:free``-suffixed ids are
    returned — the catalog's zero-cost models, sorted — matching the
    convention of :func:`list_openrouter_models`. Returns an empty list on any
    connection / HTTP / parse error so callers (e.g. a UI selectbox) can
    degrade gracefully to manual entry.
    """
    url = _join_url(endpoint, "/models")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
        return []

    models = []
    for entry in payload.get("data") or []:
        model_id = entry.get("id") if isinstance(entry, dict) else None
        if isinstance(model_id, str) and model_id.endswith(":free"):
            models.append(model_id)
    return sorted(models)


def _join_url(endpoint, path):
    return endpoint.rstrip("/") + path


def _unlink_quietly(path):
    """Best-effort file removal (temp Seatbelt profiles); never raises."""
    try:
        os.unlink(path)
    except OSError:
        pass


def list_ollama_models(endpoint=DEFAULT_ENDPOINT, timeout=10):
    """Return model names advertised by the Ollama server.

    Returns an empty list on any connection / HTTP / parse error so callers
    (e.g. a UI selectbox) can degrade gracefully to manual entry.
    """
    url = _join_url(endpoint, "/api/tags")
    request = urllib.request.Request(
        url, method="GET", headers={"User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
        return []

    models = []
    for entry in payload.get("models", []) or []:
        name = entry.get("name") or entry.get("model")
        if name:
            models.append(name)
    return models


def list_openrouter_models(endpoint=OPENROUTER_DEFAULT_ENDPOINT, timeout=30):
    """Return OpenRouter's free model ids from the public ``/models`` catalog.

    Only the ``:free`` variants are listed — the catalog's zero-cost models
    (their ``pricing`` fields are 0; paid models keep their bare id). The
    catalog is public, so no API key is needed. Returns an empty list on any
    connection / HTTP / parse error so callers (e.g. a UI selectbox) can
    degrade gracefully to manual entry — mirrors :func:`list_ollama_models`.
    """
    url = _join_url(endpoint, "/models")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
        return []

    models = []
    for entry in payload.get("data") or []:
        model_id = entry.get("id") if isinstance(entry, dict) else None
        if isinstance(model_id, str) and model_id.endswith(":free"):
            models.append(model_id)
    return sorted(models)


def ollama_generate(
    text,
    instruction=DEFAULT_INSTRUCTION,
    endpoint=DEFAULT_ENDPOINT,
    model=DEFAULT_MODEL,
    timeout=REQUEST_TIMEOUT,
):
    """Send `text` to Ollama prefixed by `instruction` (default: '概括').

    Raises RuntimeError with a clear message on connection or server failure so
    the UI can surface it via st.error.
    """
    if not model:
        raise ValueError("No Ollama model specified.")

    prompt = f"{instruction}\n\n{text}".strip()
    body = json.dumps(
        {"model": model, "prompt": prompt, "stream": False}
    ).encode("utf-8")
    request = urllib.request.Request(
        _join_url(endpoint, "/api/generate"),
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama server returned HTTP {e.code}: {detail}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not connect to Ollama at {endpoint}: {e}"
        ) from e

    # Ollama returns a single JSON object when stream=false (but be defensive:
    # older servers sometimes stream newline-delimited objects).
    response_text = raw.strip()
    if not response_text:
        return ""
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError:
        # Take the last JSON line; that carries the final "response".
        for line in reversed(response_text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        else:
            return response_text

    if isinstance(payload, dict):
        if "response" in payload:
            return payload["response"]
        if "message" in payload and isinstance(payload["message"], dict):
            return payload["message"].get("content", "")
    return ""


def _post_json(url, body, headers, timeout):
    """POST a JSON body and return the decoded text, with shared error handling.

    Raises RuntimeError with a clear message on connection / HTTP failure so the
    UI can surface it via st.error. Mirrors the error handling the one-shot
    clients inline, so the multi-turn chat clients stay consistent.

    ``USER_AGENT`` is injected here (overriding any caller-supplied UA) so every
    non-streaming request carries a signature Cloudflare won't block (see
    ``USER_AGENT``).
    """
    headers = dict(headers)
    headers["User-Agent"] = USER_AGENT
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{url} returned HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not connect to {url}: {e}") from e


def _iter_stream_lines(response):
    """Yield decoded text lines from an HTTP response as soon as they arrive.

    Uses ``response.read1()`` (one socket read at a time, returning whatever is
    currently available) rather than the buffered ``readline()`` / line-iterator
    path. Under chunked transfer encoding (which OpenRouter's SSE uses), the
    line iterator routes through ``io.IOBase.readline`` backed by an 8 KiB
    ``BufferedReader`` that greedily pulls as much as the socket has — so a fast
    reply can land entirely in one batch and the caller never sees a token at a
    time. ``read1`` returns just the bytes available now, and a partial-line
    buffer here reassembles any line split across reads, so each SSE/NDJSON line
    is yielded the moment the network delivers it.
    """
    buf = ""
    while True:
        chunk = response.read1(4096)
        if not chunk:
            break
        buf += chunk.decode("utf-8", errors="replace")
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            yield line
    if buf:
        yield buf




def openrouter_chat(
    messages,
    *,
    api_key=None,
    model,
    endpoint=OPENROUTER_DEFAULT_ENDPOINT,
    instruction=None,
    timeout=REQUEST_TIMEOUT,
):
    """Multi-turn chat against OpenRouter's chat-completions API.

    `messages` is a list of ``{"role": ..., "content": ...}`` dicts (the live
    conversation). If `instruction` is given, it is prepended as a ``system``
    message so it shapes the assistant's behaviour. `api_key` defaults to the
    OPENROUTER_API_KEY env var. Returns the assistant's reply text.

    Raises RuntimeError with a clear message on auth / connection / server
    failure so the UI can surface it via st.error.
    """
    if api_key is None:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError(
            "No OpenRouter API key provided and OPENROUTER_API_KEY is unset."
        )
    if not model:
        raise ValueError("No OpenRouter model specified.")

    full = []
    if instruction:
        full.append({"role": "system", "content": instruction})
    full.extend(messages)

    body = {"model": model, "messages": full, "stream": False}
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/local-transcriber",
        "X-Title": "Local Transcriber",
    }
    raw = _post_json(
        _join_url(endpoint, "/chat/completions"), body, headers, timeout
    ).strip()
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw

    if isinstance(payload, dict):
        choices = payload.get("choices") or []
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message")
            if isinstance(msg, dict):
                return msg.get("content", "") or ""
        if "error" in payload:
            err = payload["error"]
            err_msg = err.get("message") if isinstance(err, dict) else str(err)
            raise RuntimeError(f"OpenRouter error: {err_msg}")
    return ""


def ollama_chat(
    messages,
    *,
    endpoint=DEFAULT_ENDPOINT,
    model,
    instruction=None,
    timeout=REQUEST_TIMEOUT,
):
    """Multi-turn chat against Ollama's /api/chat endpoint.

    `messages` is a list of ``{"role": ..., "content": ...}`` dicts. If
    `instruction` is given it is prepended as a ``system`` message. Returns the
    final assistant message content (Ollama's non-streaming /api/chat returns a
    single JSON object with ``message.content``).

    Raises RuntimeError with a clear message on connection / server failure so
    the UI can surface it via st.error.
    """
    if not model:
        raise ValueError("No Ollama model specified.")

    full = []
    if instruction:
        full.append({"role": "system", "content": instruction})
    full.extend(messages)

    body = {"model": model, "messages": full, "stream": False}
    headers = {"Content-Type": "application/json"}
    raw = _post_json(
        _join_url(endpoint, "/api/chat"), body, headers, timeout
    ).strip()
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        # Older servers sometimes stream newline-delimited objects; take the
        # last line (it carries the final message).
        for line in reversed(raw.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        else:
            return raw

    if isinstance(payload, dict):
        msg = payload.get("message")
        if isinstance(msg, dict):
            return msg.get("content", "") or ""
    return ""


def openrouter_chat_stream(
    messages,
    *,
    api_key=None,
    model,
    endpoint=OPENROUTER_DEFAULT_ENDPOINT,
    instruction=None,
    timeout=REQUEST_TIMEOUT,
):
    """Streaming version of :func:`openrouter_chat`.

    Yields incremental assistant text deltas as they arrive, then ends. Uses
    Server-Sent Events (``stream: true``): each SSE ``data:`` line is a JSON
    chunk with ``choices[0].delta.content``; the ``[DONE]`` sentinel terminates
    the stream. Errors are raised as RuntimeError (same contract as the one-shot
    client) so the UI can surface them via st.error.
    """
    if api_key is None:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError(
            "No OpenRouter API key provided and OPENROUTER_API_KEY is unset."
        )
    if not model:
        raise ValueError("No OpenRouter model specified.")

    full = []
    if instruction:
        full.append({"role": "system", "content": instruction})
    full.extend(messages)

    body = json.dumps(
        {"model": model, "messages": full, "stream": True}
    ).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/local-transcriber",
        "X-Title": "Local Transcriber",
        "User-Agent": USER_AGENT,
    }
    request = urllib.request.Request(
        _join_url(endpoint, "/chat/completions"),
        data=body, headers=headers, method="POST",
    )
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenRouter returned HTTP {e.code}: {detail}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not connect to OpenRouter at {endpoint}: {e}"
        ) from e

    with response:
        for line in _iter_stream_lines(response):
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if not isinstance(chunk, dict):
                continue
            if "error" in chunk:
                err = chunk["error"]
                err_msg = err.get("message") if isinstance(err, dict) else str(err)
                raise RuntimeError(f"OpenRouter error: {err_msg}")
            choices = chunk.get("choices") or []
            if choices and isinstance(choices[0], dict):
                delta = choices[0].get("delta")
                if isinstance(delta, dict):
                    piece = delta.get("content") or ""
                    if piece:
                        yield piece


def ollama_chat_stream(
    messages,
    *,
    endpoint=DEFAULT_ENDPOINT,
    model,
    instruction=None,
    timeout=REQUEST_TIMEOUT,
):
    """Streaming version of :func:`ollama_chat`.

    Ollama's ``/api/chat`` with ``stream: true`` emits newline-delimited JSON
    objects, each carrying a ``message.content`` delta; the final object has
    ``"done": true``. Yields incremental text chunks as they arrive.
    """
    if not model:
        raise ValueError("No Ollama model specified.")

    full = []
    if instruction:
        full.append({"role": "system", "content": instruction})
    full.extend(messages)

    body = json.dumps(
        {"model": model, "messages": full, "stream": True}
    ).encode("utf-8")
    headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
    request = urllib.request.Request(
        _join_url(endpoint, "/api/chat"),
        data=body, headers=headers, method="POST",
    )
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama server returned HTTP {e.code}: {detail}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not connect to Ollama at {endpoint}: {e}"
        ) from e

    with response:
        for line in _iter_stream_lines(response):
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(chunk, dict):
                continue
            if chunk.get("error"):
                raise RuntimeError(f"Ollama error: {chunk['error']}")
            msg = chunk.get("message")
            if isinstance(msg, dict):
                piece = msg.get("content") or ""
                if piece:
                    yield piece
            if chunk.get("done"):
                break


def openrouter_generate(
    text,
    instruction=DEFAULT_INSTRUCTION,
    api_key=None,
    model=OPENROUTER_DEFAULT_MODEL,
    endpoint=OPENROUTER_DEFAULT_ENDPOINT,
    timeout=REQUEST_TIMEOUT,
):
    """Send `text` to OpenRouter's chat-completions API.

    The `instruction` is the user-facing task description (e.g. "概括") and is
    framed as the system message so it shapes the assistant's behaviour without
    being echoed back. `api_key` defaults to the OPENROUTER_API_KEY env var.

    Raises RuntimeError with a clear message on auth / connection / server
    failure so the UI can surface it via st.error.
    """
    if api_key is None:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError(
            "No OpenRouter API key provided and OPENROUTER_API_KEY is unset."
        )
    if not model:
        raise ValueError("No OpenRouter model specified.")

    # stream=false so we get one complete JSON response rather than a
    # newline-delimited stream of partial chunks — the UI shows only the
    # finished result and discards anything that didn't come back whole.
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": text},
            ],
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        _join_url(endpoint, "/chat/completions"),
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/local-transcriber",
            "X-Title": "Local Transcriber",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenRouter returned HTTP {e.code}: {detail}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not connect to OpenRouter at {endpoint}: {e}"
        ) from e

    response_text = raw.strip()
    if not response_text:
        return ""
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError:
        return response_text

    if isinstance(payload, dict):
        choices = payload.get("choices") or []
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message")
            if isinstance(msg, dict):
                return msg.get("content", "") or ""
        # Some providers return the OpenAI "error" envelope on failures.
        if "error" in payload:
            err = payload["error"]
            err_msg = err.get("message") if isinstance(err, dict) else str(err)
            raise RuntimeError(f"OpenRouter error: {err_msg}")
    return ""


def openai_chat(
    messages,
    *,
    api_key=None,
    model,
    endpoint=OPENAI_DEFAULT_ENDPOINT,
    instruction=None,
    timeout=REQUEST_TIMEOUT,
):
    """Multi-turn chat against any OpenAI-compatible /chat/completions API.

    Sibling of :func:`openrouter_chat`, minus OpenRouter's attribution headers.
    ``api_key`` defaults to the OPENAI_API_KEY env var. Point ``endpoint`` at
    OpenAI itself, Groq (``https://api.groq.com/openai/v1``), Together, etc.
    Returns the assistant's reply text.

    Raises RuntimeError with a clear message on auth / connection / server
    failure so the UI can surface it via st.error.
    """
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "No OpenAI API key provided and OPENAI_API_KEY is unset."
        )
    if not model:
        raise ValueError("No OpenAI model specified.")

    full = []
    if instruction:
        full.append({"role": "system", "content": instruction})
    full.extend(messages)

    body = {"model": model, "messages": full, "stream": False}
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    raw = _post_json(
        _join_url(endpoint, "/chat/completions"), body, headers, timeout
    ).strip()
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw

    if isinstance(payload, dict):
        choices = payload.get("choices") or []
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message")
            if isinstance(msg, dict):
                return msg.get("content", "") or ""
        if "error" in payload:
            err = payload["error"]
            err_msg = err.get("message") if isinstance(err, dict) else str(err)
            raise RuntimeError(f"OpenAI error: {err_msg}")
    return ""


def openai_chat_stream(
    messages,
    *,
    api_key=None,
    model,
    endpoint=OPENAI_DEFAULT_ENDPOINT,
    instruction=None,
    timeout=REQUEST_TIMEOUT,
):
    """Streaming version of :func:`openai_chat`.

    Sibling of :func:`openrouter_chat_stream`, minus OpenRouter's attribution
    headers. Yields incremental assistant text deltas as they arrive via SSE
    (``stream: true``); each ``data:`` line is a JSON chunk with
    ``choices[0].delta.content``; the ``[DONE]`` sentinel terminates the
    stream. Errors are raised as RuntimeError.
    """
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "No OpenAI API key provided and OPENAI_API_KEY is unset."
        )
    if not model:
        raise ValueError("No OpenAI model specified.")

    full = []
    if instruction:
        full.append({"role": "system", "content": instruction})
    full.extend(messages)

    body = json.dumps(
        {"model": model, "messages": full, "stream": True}
    ).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": USER_AGENT,
    }
    request = urllib.request.Request(
        _join_url(endpoint, "/chat/completions"),
        data=body, headers=headers, method="POST",
    )
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenAI endpoint returned HTTP {e.code}: {detail}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not connect to OpenAI endpoint at {endpoint}: {e}"
        ) from e

    with response:
        for line in _iter_stream_lines(response):
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if not isinstance(chunk, dict):
                continue
            if "error" in chunk:
                err = chunk["error"]
                err_msg = err.get("message") if isinstance(err, dict) else str(err)
                raise RuntimeError(f"OpenAI error: {err_msg}")
            choices = chunk.get("choices") or []
            if choices and isinstance(choices[0], dict):
                delta = choices[0].get("delta")
                if isinstance(delta, dict):
                    piece = delta.get("content") or ""
                    if piece:
                        yield piece


def openai_generate(
    text,
    instruction=DEFAULT_INSTRUCTION,
    api_key=None,
    model=OPENAI_DEFAULT_MODEL,
    endpoint=OPENAI_DEFAULT_ENDPOINT,
    timeout=REQUEST_TIMEOUT,
):
    """Send `text` to any OpenAI-compatible /chat/completions API.

    Sibling of :func:`openrouter_generate`, minus OpenRouter's attribution
    headers. The `instruction` is framed as the system message; `api_key`
    defaults to the OPENAI_API_KEY env var. Point ``endpoint`` at OpenAI
    itself, Groq (``https://api.groq.com/openai/v1``), Together, etc., and
    pass the host's model id (e.g. ``gpt-4o-mini``, ``qwen/qwen3-32b``).

    Raises RuntimeError with a clear message on auth / connection / server
    failure so the UI can surface it via st.error.
    """
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "No OpenAI API key provided and OPENAI_API_KEY is unset."
        )
    if not model:
        raise ValueError("No OpenAI model specified.")

    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": text},
            ],
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        _join_url(endpoint, "/chat/completions"),
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenAI endpoint returned HTTP {e.code}: {detail}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not connect to OpenAI endpoint at {endpoint}: {e}"
        ) from e

    response_text = raw.strip()
    if not response_text:
        return ""
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError:
        return response_text

    if isinstance(payload, dict):
        choices = payload.get("choices") or []
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message")
            if isinstance(msg, dict):
                return msg.get("content", "") or ""
        # Some providers return the OpenAI "error" envelope on failures.
        if "error" in payload:
            err = payload["error"]
            err_msg = err.get("message") if isinstance(err, dict) else str(err)
            raise RuntimeError(f"OpenAI error: {err_msg}")
    return ""


# ---------------------------------------------------------------------------
# OpenCode (coding agent, subprocess path)
# ---------------------------------------------------------------------------
#
# `opencode run --format json --auto` emits one JSON object per line (JSONL) on
# stdout. Each line has a ``type`` field:
#   - "text"        → part.text is a piece of the assistant's reply (yield it)
#   - "tool_use"    → a tool finished (part.tool, part.state.title); surfaced
#                     inline as a one-line marker so the user sees agent activity
#   - "step_start" / "step_finish" → step boundaries (ignored here)
#   - "error"       → part.error.data.message; raised as RuntimeError
# A non-zero process exit (with no prior error event) is also raised, using the
# collected stderr. Note: token-level streaming is NOT available on this path —
# text arrives per OpenCode text-part, which still streams progressively across
# agent steps.


def list_opencode_models(binary=OPENCODE_BIN, timeout=20):
    """Return model ids advertised by ``opencode models`` (``provider/model``).

    Returns an empty list on any failure (binary missing, non-zero exit, parse
    error) so a UI selectbox degrades gracefully to manual entry — mirrors
    :func:`list_ollama_models`.
    """
    try:
        proc = subprocess.run(
            [binary, "models"],
            capture_output=True, text=True, timeout=timeout,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return []
    if proc.returncode != 0:
        return []
    models = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # `opencode models` prints a table; the model id is the first
        # whitespace-delimited token and always contains a "/" (provider/model).
        first = line.split()[0]
        if "/" in first:
            models.append(first)
    return models


def _parse_opencode_verbose_models(text):
    """Parse ``opencode models --verbose`` output into (id, metadata) pairs.

    The output interleaves one ``provider/model`` id line per model with a
    pretty-printed JSON metadata object, and JSON string values may themselves
    contain '/'-bearing tokens (api urls), so blocks are consumed positionally
    with raw_decode instead of line-splitting. Malformed blocks are skipped
    (association resumes at the next id line); the parser is total — it never
    raises on any input.
    """
    decoder = json.JSONDecoder()
    pairs = []
    current = None
    pos = 0
    n = len(text)
    while pos < n:
        nl = text.find("\n", pos)
        end_of_line = n if nl == -1 else nl
        stripped = text[pos:end_of_line].strip()
        brace = text.find("{", pos, end_of_line) if stripped.startswith("{") else -1
        if brace != -1:
            try:
                obj, pos = decoder.raw_decode(text, brace)
                if isinstance(obj, dict):
                    pairs.append((current, obj))
                continue
            except json.JSONDecodeError:
                pos = end_of_line + 1
                continue
        first = stripped.split()[0] if stripped else ""
        if first and not first.startswith('"') and "/" in first:
            current = first
        pos = end_of_line + 1
    return pairs


def list_opencode_model_details(binary=OPENCODE_BIN, timeout=20):
    """Return per-model metadata from ``opencode models --verbose``.

    Maps each ``provider/model`` id to its metadata dict; that dict's
    ``variants`` field lists the model's reasoning-effort variants (name →
    provider options). One subprocess call serves both model discovery and
    per-model variant discovery. Returns {} on any failure (binary missing,
    non-zero exit, unparsable output) so callers degrade gracefully — mirrors
    :func:`list_opencode_models`.
    """
    try:
        proc = subprocess.run(
            [binary, "models", "--verbose"],
            capture_output=True, text=True, timeout=timeout,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return {}
    if proc.returncode != 0:
        return {}
    details = {}
    for model_id, meta in _parse_opencode_verbose_models(proc.stdout):
        if model_id:
            details[model_id] = meta
    return details


def _variant_effort_label(name, spec):
    """The effort label a variant ranks by: its ``reasoningEffort`` when the
    spec carries one, else the variant name (the catalog names variants after
    their effort level; some specs use nested ``reasoning.effort`` instead)."""
    if isinstance(spec, dict) and spec.get("reasoningEffort"):
        return str(spec["reasoningEffort"])
    return name


def _effort_rank(label):
    try:
        return OPENCODE_EFFORT_ORDER.index(label)
    except ValueError:
        return -1


def opencode_variants_for(details, model_id):
    """Return ``model_id``'s variants map from :func:`list_opencode_model_details`
    output ({} when the model is unknown or its variants are absent/malformed)."""
    meta = details.get(model_id) if isinstance(details, dict) else None
    variants = meta.get("variants") if isinstance(meta, dict) else None
    return variants if isinstance(variants, dict) else {}


def order_opencode_variants(variants):
    """Return ``variants``' names sorted least → most effort for a dropdown.

    Names outside :data:`OPENCODE_EFFORT_ORDER` (non-effort toggles like
    ``thinking``) keep catalog order at the end.
    """
    floor = len(OPENCODE_EFFORT_ORDER)
    decorated = []
    for index, (name, spec) in enumerate((variants or {}).items()):
        rank = _effort_rank(_variant_effort_label(name, spec))
        decorated.append((rank if rank >= 0 else floor, index, name))
    decorated.sort()
    return [name for _, _, name in decorated]


def highest_opencode_variant(variants):
    """Return the highest-effort variant name in ``variants``, or None.

    Ranks each name by its effort label against :data:`OPENCODE_EFFORT_ORDER`
    and returns the first top-ranked entry. Returns None when nothing ranks
    (no variants, or only non-effort names like ``thinking``) — the caller
    should then omit ``--variant``.
    """
    best, best_rank = None, -1
    for name, spec in (variants or {}).items():
        rank = _effort_rank(_variant_effort_label(name, spec))
        if rank > best_rank:
            best, best_rank = name, rank
    return best


def opencode_chat_stream(
    prompt,
    *,
    model=None,
    workdir=None,
    attach=None,
    agent=None,
    variant=None,
    binary=OPENCODE_BIN,
    instruction=None,
    hardened=False,
    timeout=REQUEST_TIMEOUT,
):
    """Run ``opencode run --format json --auto`` and stream assistant text.

    OpenCode is a coding AGENT, not a plain chat API: with ``--auto`` it may run
    bash/read/edit/etc. in ``workdir``. The JSONL event stream is parsed line by
    line; ``text`` events are yielded as assistant chunks, ``tool_use`` events
    are surfaced inline as a one-line marker, and an ``error`` event (or a
    non-zero exit) raises :class:`RuntimeError`. If ``instruction`` is given it
    is prepended to the prompt. If ``variant`` is given it is forwarded as
    ``--variant`` (provider-specific reasoning effort).

    With ``hardened=True`` (macOS) the subprocess runs under a generated
    Seatbelt profile (:mod:`md_llm.sandbox`): file writes are confined to the
    workdir + scratch space, reads of the host's data tree and credential
    stores are denied, network stays open for the model API.

    Auth + model routing are OpenCode's own (configure via ``opencode auth
    login`` / env). Token-level streaming is unavailable on this path; text
    arrives per OpenCode text-part, which still streams progressively across
    agent steps. Very large prompts may hit the OS argv length limit (consider
    ``--attach`` to a running server with file context for huge documents).

    Raises :class:`RuntimeError` with a clear message on a missing binary /
    server / agent failure so the UI can surface it via ``st.error``.
    """
    if not prompt:
        raise ValueError("No prompt for opencode run.")

    full_prompt = prompt
    if instruction:
        full_prompt = f"{instruction}\n\n{prompt}"

    args = [binary, "run", "--format", "json", "--auto"]
    if model:
        args += ["--model", model]
    if variant:
        args += ["--variant", variant]
    if workdir:
        args += ["--dir", workdir]
    if attach:
        args += ["--attach", attach]
    if agent:
        args += ["--agent", agent]
    args.append(full_prompt)

    # opencode's --dir does a chdir, so the directory must already exist.
    # Create it (best-effort) — this also gives a fresh sandbox a place to land.
    if workdir:
        try:
            os.makedirs(workdir, exist_ok=True)
        except OSError as e:
            raise RuntimeError(
                f"Could not create opencode working directory {workdir!r}: {e}"
            ) from e

    # Wrap in a macOS Seatbelt profile when hardened mode is requested and the
    # host can enforce it; elsewhere (or without sandbox-exec) run unconfined.
    profile_path = None
    if hardened and sandbox.seatbelt_available():
        try:
            profile_path = sandbox.write_seatbelt_profile(workdir or ".")
            args = ["sandbox-exec", "-f", profile_path] + args
        except OSError:
            profile_path = None  # degrade to unconfined rather than fail

    try:
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered so yielded lines arrive as produced
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Could not find the {binary!r} executable on PATH. Install "
            "opencode (see https://opencode.ai) to use this provider."
        ) from e
    except OSError as e:
        raise RuntimeError(f"Could not start opencode: {e}") from e

    # Drain stderr on a background thread so a chatty agent can't fill the OS
    # pipe buffer (64 KiB) and deadlock the stdout reader.
    stderr_lines: list[str] = []

    def _drain_stderr():
        if proc.stderr is not None:
            for ln in proc.stderr:
                stderr_lines.append(ln)

    drainer = threading.Thread(target=_drain_stderr, daemon=True)
    drainer.start()

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(evt, dict):
                continue
            etype = evt.get("type")
            if etype == "text":
                part = evt.get("part") or {}
                piece = part.get("text") or ""
                if piece:
                    yield piece
            elif etype == "tool_use":
                part = evt.get("part") or {}
                tool = part.get("tool") or "tool"
                state = part.get("state") or {}
                title = (state.get("title") or "").strip()
                label = f" — {title}" if title else ""
                yield f"\n\n_🔧 {tool}{label}_\n\n"
            elif etype == "error":
                err = evt.get("error") or {}
                data = err.get("data") or {}
                msg = (
                    data.get("message")
                    or err.get("name")
                    or "opencode error"
                )
                raise RuntimeError(f"opencode error: {msg}")
        proc.wait(timeout=timeout)
    finally:
        try:
            proc.kill()
        except OSError:
            pass
        drainer.join(timeout=5)
        if profile_path:
            _unlink_quietly(profile_path)

    if proc.returncode not in (0, None):
        stderr_text = "".join(stderr_lines).strip()
        raise RuntimeError(
            f"opencode run exited {proc.returncode}: {stderr_text[-500:]}"
        )


# ---------------------------------------------------------------------------
# Cline (coding agent, subprocess path)
# ---------------------------------------------------------------------------
#
# `cline --json "prompt"` emits newline-delimited JSON (NDJSON) on stdout, one
# object per line, with a top-level ``type`` field:
#   - "agent_event" → event.type further qualifies it:
#       * "content_start" (contentType "text") → event.text is a streaming
#         delta of the assistant's reply (yield it)
#       * "content_end"   (contentType "text") → event.text is the finished
#         text of the whole block; only the part not already streamed is
#         yielded (deltas usually covered it entirely)
#       * "content_end"   (contentType "tool") → a tool finished
#         (event.toolName, event.output); surfaced inline as a one-line marker
#         so the user sees agent activity
#       * iteration / usage / reasoning events → ignored here
#   - "run_result"   → the final summary; its ``finishReason`` ("completed" /
#     "error") and ``text`` decide success.
# Failures are reported as ``{"type":"error","message":...}`` NDJSON lines on
# STDERR (plus finishReason "error" and a non-zero exit); the last such message
# is the real failure. Reasoning (``contentType: "reasoning"``) is not yielded.
# As with OpenCode, token-level streaming comes from the content deltas, and
# text still lands per content block across agent steps.


def _cline_tool_label(output):
    """One-line label for a cline tool event's output payload ('' when none).

    Tool outputs are either a single object (``{"query": "edit:/path/…",
    "result": …, "success": …}``) or a list of such objects (read_files et al.);
    the first ``query`` found names what the tool acted on. ``result`` only
    serves as the fallback when NO item carries a query, and anything else
    (missing / non-dict shapes) yields ''.
    """
    if isinstance(output, dict):
        output = [output]
    if isinstance(output, list):
        fallback = ""
        for item in output:
            if not isinstance(item, dict):
                continue
            query = item.get("query")
            if query:
                return " ".join(str(query).split())
            if not fallback:
                fallback = str(item.get("result") or "")
        return " ".join(fallback.split())
    return ""


def cline_chat_stream(
    prompt,
    *,
    model=None,
    workdir=None,
    thinking=None,
    binary=CLINE_BIN,
    instruction=None,
    hardened=False,
    timeout=REQUEST_TIMEOUT,
):
    """Run ``cline --json`` and stream assistant text.

    Cline is a coding AGENT, not a plain chat API: a positional prompt runs it
    headless in act mode with tools auto-approved (``--auto-approve true``), so
    it may run bash/read/edit/etc. in ``workdir`` (``--cwd``). The NDJSON event
    stream is parsed line by line; text deltas are yielded as assistant chunks,
    finished tools are surfaced inline as a one-line marker, and a ``run_result``
    with ``finishReason: "error"`` (or a non-zero exit) raises
    :class:`RuntimeError` — preferring the last ``{"type":"error"}`` message
    cline wrote to stderr. If ``instruction`` is given it is prepended to the
    prompt; if ``model`` is given it is forwarded as ``--model`` (omit it to use
    the model Cline was configured with via ``cline auth``); if ``thinking`` is
    given it is forwarded as ``--thinking`` (none|low|medium|high|xhigh).

    Two CLI quirks are handled here (observed on cline 3.0.60): a
    whitespace-free prompt argument is parsed as a command lookup (``cline hi``
    → "Unknown command or unquoted prompt"), so a trailing newline is appended
    when the prompt carries no whitespace at all; and passing ``--model``
    persists that model as Cline's own new default, so later runs outside this
    app will use it too.

    With ``hardened=True`` (macOS) the subprocess runs under a generated
    Seatbelt profile (:mod:`md_llm.sandbox`): file writes are confined to the
    workdir + scratch space, reads of the host's data tree and credential
    stores are denied, network stays open for the model API.

    Auth + model routing are Cline's own (configure via ``cline auth`` / env).
    Very large prompts may hit the OS argv length limit (cline 3.0.60's piped
    stdin mode is broken, so the argument-passing route is the only one).

    Raises :class:`RuntimeError` with a clear message on a missing binary /
    server / agent failure so the UI can surface it via ``st.error``.
    """
    if not prompt:
        raise ValueError("No prompt for cline run.")

    full_prompt = prompt
    if instruction:
        full_prompt = f"{instruction}\n\n{prompt}"
    # See docstring: cline parses a whitespace-free positional as a command.
    if not re.search(r"\s", full_prompt):
        full_prompt += "\n"

    args = [binary, "--json", "--auto-approve", "true"]
    if model:
        args += ["--model", model]
    if thinking:
        args += ["--thinking", thinking]
    if workdir:
        args += ["--cwd", workdir]
    args.append(full_prompt)

    # cline's --cwd chdirs, so the directory must already exist. Create it
    # (best-effort) — this also gives a fresh sandbox a place to land.
    if workdir:
        try:
            os.makedirs(workdir, exist_ok=True)
        except OSError as e:
            raise RuntimeError(
                f"Could not create cline working directory {workdir!r}: {e}"
            ) from e

    # Wrap in a macOS Seatbelt profile when hardened mode is requested and the
    # host can enforce it; elsewhere (or without sandbox-exec) run unconfined.
    profile_path = None
    if hardened and sandbox.seatbelt_available():
        try:
            profile_path = sandbox.write_seatbelt_profile(workdir or ".")
            args = ["sandbox-exec", "-f", profile_path] + args
        except OSError:
            profile_path = None  # degrade to unconfined rather than fail

    try:
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered so yielded lines arrive as produced
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Could not find the {binary!r} executable on PATH. Install "
            "cline (see https://docs.cline.bot) to use this provider."
        ) from e
    except OSError as e:
        raise RuntimeError(f"Could not start cline: {e}") from e

    # Drain stderr on a background thread so a chatty agent can't fill the OS
    # pipe buffer (64 KiB) and deadlock the stdout reader.
    stderr_lines: list[str] = []

    def _drain_stderr():
        if proc.stderr is not None:
            for ln in proc.stderr:
                stderr_lines.append(ln)

    drainer = threading.Thread(target=_drain_stderr, daemon=True)
    drainer.start()

    finish_reason = None  # run_result.finishReason ("completed"/"error"/…)
    finish_text = ""      # run_result.text (the error message on failures)
    streamed = ""         # text already yielded for the current text block
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(evt, dict):
                continue
            if evt.get("type") != "agent_event":
                if evt.get("type") == "run_result":
                    finish_reason = evt.get("finishReason")
                    finish_text = evt.get("text") or ""
                continue
            event = evt.get("event") or {}
            etype = event.get("type")
            if etype == "iteration_start":
                # Content blocks complete within their iteration; dropping any
                # un-ended block's delta prefix here keeps the dedup check in
                # content_end honest for the next block.
                streamed = ""
            elif etype == "content_start" and event.get("contentType") == "text":
                piece = event.get("text") or ""
                if piece:
                    streamed += piece
                    yield piece
            elif etype == "content_end":
                ctype = event.get("contentType")
                if ctype == "text":
                    # Deltas already streamed via content_start; yield only
                    # whatever the finished text adds beyond that prefix.
                    full = event.get("text") or ""
                    if full.startswith(streamed):
                        rest = full[len(streamed):]
                        if rest:
                            yield rest
                    streamed = ""
                elif ctype == "tool":
                    tool = event.get("toolName") or "tool"
                    label = _cline_tool_label(event.get("output"))
                    suffix = f" — {label}" if label else ""
                    yield f"\n\n_🔧 {tool}{suffix}_\n\n"
        proc.wait(timeout=timeout)
    finally:
        try:
            proc.kill()
        except OSError:
            pass
        drainer.join(timeout=5)
        if profile_path:
            _unlink_quietly(profile_path)

    if finish_reason == "error" or proc.returncode not in (0, None):
        # cline writes {"type":"error","message":…} NDJSON lines to stderr; the
        # LAST one is the real failure (earlier ones can be hook noise). Fall
        # back to the run_result text, then the raw stderr tail (ANSI-stripped:
        # CLI parse errors arrive colourized, e.g. "\x1b[31merror:\x1b[0m …").
        err_msg = None
        for ln in stderr_lines:
            ln = ln.strip()
            if not ln:
                continue
            try:
                s_evt = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if isinstance(s_evt, dict) and s_evt.get("type") == "error":
                err_msg = s_evt.get("message") or err_msg
        if not err_msg:
            err_msg = (
                finish_text
                or _ANSI_ESCAPE_RE.sub("", "".join(stderr_lines)).strip()[-500:]
                or f"exited {proc.returncode}"
            )
        raise RuntimeError(f"cline error: {err_msg}")
