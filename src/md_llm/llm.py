"""Lightweight LLM clients for post-processing transcripts (summarize, etc.).

Four providers are supported:
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

Uses only the standard library (urllib + json + subprocess) to match the
no-extra-deps style of transcribe_local.py's remote-Whisper client.
"""

import os
import json
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

# Suggested model variants for `opencode run --variant` (provider-specific
# reasoning effort). Not exhaustive — values vary by provider; the UI also lets
# the user type a custom one. Used as the dropdown options in controls.
OPENCODE_VARIANTS = ["low", "medium", "high", "minimal", "max"]


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
