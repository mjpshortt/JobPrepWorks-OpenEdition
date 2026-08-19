"""Provider that shells out to the Claude Code CLI (`claude -p`) instead of
calling a hosted API with a key — for a machine where the Anthropic Console
is unreachable (e.g. an org-restricted account) but Claude Code itself is
already logged in.

Every invocation is hard-locked to isolated, tool-free text generation:
`--safe-mode` skips this repo's own CLAUDE.md/hooks/plugins/MCP servers so
they don't leak into unrelated prompts, while (unlike `--bare`) still using
normal OAuth/keychain auth rather than requiring a raw ANTHROPIC_API_KEY —
the entire reason for this provider is not having one. `--tools ""` +
`--permission-mode dontAsk` fully disable Bash/Read/Write. This is not
configurable — résumé and job-posting text ride in every prompt and is
attacker-controlled from the model's perspective, so a prompt injection must
not be able to reach a tool. Structured extraction uses the CLI's native
`--json-schema` flag rather than embedding schema instructions in the
prompt. `max_tokens` is accepted for Protocol compatibility but unused: the
CLI has no per-call output-token flag.
"""

import json
import logging
import shutil
import subprocess
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.llm.base import LLMError
from app.llm.openai_compat_provider import _strip_fences

T = TypeVar("T", bound=BaseModel)

log = logging.getLogger(__name__)


class ClaudeCliProvider:
    def __init__(self, model: str, timeout: float):
        self.model = model
        self.timeout = timeout
        self.cli_path = shutil.which("claude")
        if not self.cli_path:
            raise LLMError("Claude CLI not found — install Claude Code and run `claude login`.")

    def _build_command(
        self, *, system: str, prompt: str, json_schema: dict | None = None
    ) -> list[str]:
        command = [
            self.cli_path,
            "--safe-mode",
            "-p",
            prompt,
            "--system-prompt",
            system,
            "--model",
            self.model,
            "--tools",
            "",
            "--permission-mode",
            "dontAsk",
        ]
        if json_schema is not None:
            command += ["--json-schema", json.dumps(json_schema)]
        return command

    def _run(self, *, system: str, prompt: str, json_schema: dict | None = None) -> str:
        command = self._build_command(system=system, prompt=prompt, json_schema=json_schema)
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            log.warning("claude CLI timed out after %ss", self.timeout)
            raise LLMError("The AI service took too long to respond — try again.") from exc
        if result.returncode != 0:
            log.warning("claude CLI exited %s: %s", result.returncode, result.stderr)
            raise LLMError(f"The AI service returned an error (exit {result.returncode}) — try again.")
        return result.stdout.strip()

    def complete(self, *, system: str, prompt: str, max_tokens: int = 16000) -> str:
        return self._run(system=system, prompt=prompt)

    def extract(self, *, system: str, prompt: str, schema: type[T], max_tokens: int = 16000) -> T:
        json_schema = schema.model_json_schema()
        last_exc: Exception | None = None
        for _attempt in range(2):  # invalid output gets exactly one retry
            content = self._run(system=system, prompt=prompt, json_schema=json_schema)
            try:
                return schema.model_validate(json.loads(_strip_fences(content)))
            except (json.JSONDecodeError, ValidationError) as exc:
                last_exc = exc
        log.warning("model output failed validation for %s", schema.__name__, exc_info=last_exc)
        raise LLMError("The AI returned an unusable response — try again.") from last_exc
