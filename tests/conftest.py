"""Suite-wide test setup: keep provider env vars out of the code under test.

``chat._build_stream`` falls back to ``OPENROUTER_API_KEY`` / ``OPENAI_API_KEY``
when session state holds no key, so a developer with one exported locally flips
"no key configured" tests into happy paths (``test_chat_save``'s
``test_validation_failure_before_send_memorizes_nothing`` failed exactly that
way). The tests that exercise the env-var fallback patch it themselves
(``mock.patch.dict("os.environ", ...)`` in test_llm), so clearing here for the
whole suite can only remove accidental leakage.
"""

import os

for _var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
    os.environ.pop(_var, None)
