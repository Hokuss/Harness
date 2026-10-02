import os
import json
import re
import sys
import time
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
from google import genai
from openai import OpenAI
from pathlib import Path

# Windows consoles often default to cp1252, which can't encode the emoji
# used in status prints below (🌲, ⚠️, 🛠️) — force UTF-8 so this doesn't
# crash the whole agent loop on a plain `python`/PowerShell invocation.
for _stream in (sys.stdout, sys.stderr):
    if getattr(_stream, "encoding", "").lower() != "utf-8":
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

# Import tools from functions.py / tools.py
from . import functions
from . import tools
from .repository_tree_sitter_scanner import RepositoryScanner

load_dotenv()

# ==========================================
# Configuration
# ==========================================
TARGET_DIR = "test"

# Request-level timeout for every provider. The OpenAI SDK default is
# 600s and google-genai is similarly generous; without an explicit cap a
# slow request looks identical to a deadlock.
REQUEST_TIMEOUT_S = 90.0

# ==========================================
# 1. Client Setup
# ==========================================
# Built lazily/defensively: a missing API key must NOT crash at import
# time (it used to — genai.Client() raises immediately without a key).
# Instead the client is left as None and chat_with_fallback() skips it.

def _make_openai_client(base_url: str, api_key_env: str) -> Optional[OpenAI]:
    key = os.getenv(api_key_env)
    if not key:
        return None
    return OpenAI(
        base_url=base_url,
        api_key=key,
        timeout=REQUEST_TIMEOUT_S,
        # Disable the SDK's own retry loop — we run our own retry + failover
        # and stacked retries silently multiply wall-clock time on 429s.
        max_retries=0,
    )


def _make_google_client() -> Optional[genai.Client]:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    return genai.Client(
        api_key=key,
        # google-genai takes millisecond timeouts via http_options.
        http_options={"timeout": int(REQUEST_TIMEOUT_S * 1000)},
    )


groq_client = _make_openai_client("https://api.groq.com/openai/v1", "GROQ_API_KEY")
openrouter_client = _make_openai_client("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY")
# Fireworks: $1 free credit, no card required at signup — easiest provider
# to add for smoke-testing this harness. OpenAI-schema compatible.
fireworks_client = _make_openai_client("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY")
# Cerebras: fastest tokens/sec of the free-tier options, but now requires
# a verified card and has tighter RPM limits. Optional bonus provider.
cerebras_client = _make_openai_client("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY")
google_client = _make_google_client()

OPENAI_STYLE_CLIENTS: Dict[str, Optional[OpenAI]] = {
    "groq": groq_client,
    "openrouter": openrouter_client,
    "fireworks": fireworks_client,
    "cerebras": cerebras_client,
}

# Combine repo tools (functions.py) with general-purpose tools (tools.py)
ALL_TOOLS: List[Dict[str, Any]] = functions.TOOLS + tools.TOOLS

# Convert into standard OpenAI JSON schemas
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"]
        }
    }
    for tool in ALL_TOOLS
]


def call_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch to functions.py (repo tools) or tools.py (general tools)."""
    if name in functions._DISPATCH:
        return functions.call_tool(name, arguments)
    if name in tools._DISPATCH:
        return tools.call_tool(name, arguments)
    return {"error": f"unknown tool: {name}"}


# Priority rotation pipeline: (provider, model_name)
MODEL_PIPELINE = [
    # ==========================================
    # Groq (Free Developer Tier)
    # Note: No credit card required for free tier.
    # Limits: ~30 RPM, 6,000 TPM, 14,400 RPD.
    # ==========================================
    ("groq", "openai/gpt-oss-20b"),
    ("groq", "openai/gpt-oss-120b"),
    ("groq", "qwen/qwen3.8-27b"),
    # ("openrouter", "thinkingmachines/inkling-small:free"),
    ("openrouter", "google/gemma-4-26b-a4b-it:free"),
    ("openrouter", "google/gemma-4-31b-it:free"),
    ("openrouter", "cohere/north-mini-code:free"),

    # ==========================================
    # Google GenAI (Free Tier)
    # Note: These models show "$0.00" for input/output
    # on the Free Tier, but are strictly rate-limited
    # (e.g., 15-60 RPM, 1M tokens/day) and data
    # may be used to improve Google's products.
    # ==========================================
    ("google", "gemini-3.6-flash"),
    ("google", "gemini-3.8-flash"),
    ("google", "gemini-3.7-flash"),
    ("google", "gemini-3.5-flash"),
    ("google", "gemini-3-flash"),
    ("google", "gemini-3.5-flash-lite"),
    ("google", "gemini-2.5-flash"),
    ("google", "gemini-2.5-flash-lite"),
    ("google", "gemma-4"), # Open model, free via API
]

# ==========================================
# 2. Model Health & Failure Tracker
# ==========================================
class ModelTracker:
    """Tracks failed, rate-limited, or broken models to avoid retrying them during a run."""
    def __init__(self):
        self.failed_models: Dict[Tuple[str, str], str] = {}

    def mark_failed(self, provider: str, model: str, reason: str):
        key = (provider, model)
        if key not in self.failed_models:
            print(f"  ⚠️ [Tracker] Disabling model for this run: {provider} -> {model} (Reason: {reason})")
            self.failed_models[key] = reason

    def is_failed(self, provider: str, model: str) -> bool:
        return (provider, model) in self.failed_models

    def status_summary(self) -> str:
        if not self.failed_models:
            return "All models active."
        return ", ".join([f"{p}:{m} ({r})" for (p, m), r in self.failed_models.items()])

TRACKER = ModelTracker()

# ==========================================
# 3. Universal API Runner with Fallback
# ==========================================
MAX_TRANSIENT_RETRIES = 2      # extra attempts (beyond the first) for retryable errors
RETRY_BACKOFF_BASE_S = 1.0     # doubles each retry: 1s, 2s

ModelInfo = Dict[str, str]     # {"provider": ..., "model": ...}

# Substrings that indicate the *payload* is too big, not that we're being
# throttled for request rate. A 429 mentioning tokens-per-minute is a
# size problem: waiting won't help, sending a smaller prompt will.
_CONTEXT_LENGTH_MARKERS = (
    "context length",
    "context_length_exceeded",
    "context window",
    "maximum context",
    "max context",
    "too many tokens",
    "tokens per minute",
    "token quota",
    "reduce the length",
    "reduce your prompt",
    "request too large",
    "payload too large",
    "input is too long",
    "prompt is too long",
    "exceeds the maximum",
)


def _error_status(exc: Exception) -> Optional[int]:
    """Pull an HTTP-ish status code off an SDK exception, if it has one."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "code", None)
    return status if isinstance(status, int) else None


def _classify_error(exc: Exception) -> str:
    """
    Return one of:
      'context_length' — payload too large; retrying identical bytes is useless.
      'transient'      — 429 (rate), 5xx, or network-level; worth a quick retry.
      'permanent'      — auth/bad-request/etc.; fail over immediately.
    """
    msg = str(exc).lower()
    status = _error_status(exc)

    if any(marker in msg for marker in _CONTEXT_LENGTH_MARKERS):
        return "context_length"
    if status == 429 or (status is not None and status >= 500):
        return "transient"
    if status is not None:
        return "permanent"
    # No status attached: network timeout / connection drop — transient.
    return "transient"


def _compact_messages(
    messages: List[Dict[str, str]],
    keep_tail: int = 4,
    tool_output_chars: int = 1500,
) -> None:
    """
    Shrink an oversized conversation *in place* so the next model in the
    pipeline has a fighting chance. Destructive on purpose: the caller's
    self.messages is truncated, which also stops the loop from growing
    without bound across steps.

    Strategy:
      - If there are enough messages to spare, keep the system prompt and
        the last `keep_tail` messages, replacing everything before them
        with a one-line note.
      - Otherwise truncate the longest messages in place.
    """
    if len(messages) > keep_tail + 1:
        system = messages[0] if messages and messages[0]["role"] == "system" else None
        tail = messages[-keep_tail:]
        new_tail = []
        for m in tail:
            c = m.get("content", "") or ""
            if len(c) > tool_output_chars:
                c = c[:tool_output_chars] + "\n...[truncated to fit context window]"
            new_tail.append({**m, "content": c})
        messages[:] = (
            ([system] if system else [])
            + [{
                "role": "user",
                "content": "[Earlier tool outputs were elided to fit the model's context window.]",
            }]
            + new_tail
        )
        return

    # Too few messages to drop — truncate the big ones in place.
    for m in messages:
        c = m.get("content", "") or ""
        if len(c) > tool_output_chars:
            m["content"] = c[:tool_output_chars] + "\n...[truncated]"


def chat_with_fallback(
    messages: List[Dict[str, str]]
) -> Tuple[str, List[Dict[str, Any]], ModelInfo]:
    """
    Try each (provider, model) in MODEL_PIPELINE in order, skipping ones
    already blacklisted this run. Returns (text, tool_calls, model_info)
    where tool_calls is a list (possibly empty — a model can request
    several tool calls in one turn, not just one) and model_info records
    which (provider, model) actually answered.

    Failure handling:
      - Output-token exhaustion ("finish_reason == 'length'" / MAX_TOKENS)
        is treated as a failure, not a valid answer: the reply may be
        truncated mid-tool-call, so the model is blacklisted, the context
        is compacted, and the next model is tried.
      - Context-length rejections compact messages before failing over.
      - Plain 429/5xx errors retry in place with exponential backoff.
    """
    last_error: Optional[Exception] = None

    for provider, model in MODEL_PIPELINE:
        if TRACKER.is_failed(provider, model):
            continue

        if provider in OPENAI_STYLE_CLIENTS:
            if OPENAI_STYLE_CLIENTS[provider] is None:
                TRACKER.mark_failed(provider, model, "no API key configured")
                continue
        elif provider == "google" and google_client is None:
            TRACKER.mark_failed(provider, model, "no API key configured")
            continue

        attempt = 0
        while True:
            try:
                if provider in OPENAI_STYLE_CLIENTS:
                    client = OPENAI_STYLE_CLIENTS[provider]
                    res = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=OPENAI_TOOLS,
                        tool_choice="auto",
                        temperature=0.2,
                    )
                    choice = res.choices[0]

                    # Output budget exhausted mid-reply. Anything the
                    # model emitted — especially tool calls — may be
                    # truncated. Do NOT return it as a valid answer.
                    if choice.finish_reason == "length":
                        print(f"  [Truncated] {provider} -> {model} hit its output "
                              f"token cap; compacting context and failing over.")
                        TRACKER.mark_failed(provider, model, "output token limit")
                        _compact_messages(messages)
                        break

                    text = choice.message.content or ""

                    tool_calls: List[Dict[str, Any]] = []
                    for tc in (choice.message.tool_calls or []):
                        try:
                            args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            args = {}
                        tool_calls.append({"name": tc.function.name, "arguments": args})

                    return text, tool_calls, {"provider": provider, "model": model}

                elif provider == "google":
                    system_instruction = None
                    contents = []
                    for msg in messages:
                        if msg["role"] == "system":
                            system_instruction = msg["content"]
                        else:
                            role = "user" if msg["role"] == "user" else "model"
                            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

                    res = google_client.models.generate_content(
                        model=model,
                        contents=contents,
                        config={
                            "system_instruction": system_instruction,
                            "temperature": 0.2,
                        },
                    )

                    # Detect output truncation. The google-genai SDK
                    # surfaces this on candidates[0].finish_reason.
                    finish = None
                    try:
                        finish = str(res.candidates[0].finish_reason).upper()
                    except (AttributeError, IndexError, TypeError):
                        pass

                    if finish and "MAX_TOKENS" in finish:
                        print(f"  [Truncated] google -> {model} hit its output "
                              f"token cap; compacting context and failing over.")
                        TRACKER.mark_failed(provider, model, "output token limit")
                        _compact_messages(messages)
                        break

                    # res.text raises if the response has no text part
                    # (e.g. safety block, or a function-call-only reply).
                    try:
                        text = res.text or ""
                    except (ValueError, AttributeError):
                        text = ""

                    return text, [], {"provider": provider, "model": model}

            except Exception as e:
                err_type = type(e).__name__
                kind = _classify_error(e)
                last_error = e

                if kind == "context_length":
                    print(f"  [Context] {provider} -> {model} rejected the payload "
                          f"({err_type}: {e}); compacting and failing over.")
                    TRACKER.mark_failed(provider, model, "context length")
                    _compact_messages(messages)
                    break

                if kind == "transient" and attempt < MAX_TRANSIENT_RETRIES:
                    attempt += 1
                    wait = RETRY_BACKOFF_BASE_S * (2 ** (attempt - 1))
                    print(f"  [Retry] {provider} -> {model} transient error "
                          f"({err_type}); retrying in {wait:.1f}s "
                          f"(attempt {attempt}/{MAX_TRANSIENT_RETRIES})")
                    time.sleep(wait)
                    continue

                print(f"  [API Error] {provider} -> {model} failed ({err_type}: {e})")
                TRACKER.mark_failed(provider, model, err_type)
                break  # give up on this (provider, model); try the next one

    raise RuntimeError(
        f"All models failed or were rate-limited. Summary: {TRACKER.status_summary()}"
        + (f" | Last error: {last_error}" if last_error else "")
    )

# ==========================================
# 4. Harness Agent
# ==========================================
class HarnessAgent:
    def __init__(self, target_dir: str = TARGET_DIR, max_iterations: int = 15):
        self.target_dir = target_dir
        self.max_iterations = max_iterations
        self.messages = []

        print(f"🌲 Initializing Tree-sitter scanner for '{self.target_dir}'...")
        scanner = RepositoryScanner()
        target_path = Path(self.target_dir).resolve()

        repo_tree = scanner.get_repo_tree(target_path, force_refresh=True)

        for f in repo_tree["files"]:
            f["path"] = f"{self.target_dir}/{f['path']}"
        repo_tree["root_path"] = str(Path(".").resolve())

        functions._TREE = repo_tree
        functions.REPO_ROOT = Path(".").resolve()

        stats = repo_tree.get("stats", {})
        files_count = stats.get("total_files", 0)
        langs = ", ".join(stats.get("files_by_language", {}).keys())

        dir_structure = set()
        for f in repo_tree["files"]:
            parts = Path(f["path"]).parts
            if len(parts) > 1:
                dir_structure.add(parts[1])

        architecture_context = (
            f"### Project Architecture Summary\n"
            f"- Total tracked files: {files_count}\n"
            f"- Languages detected: {langs}\n"
            f"- Primary subdirectories: {', '.join(sorted(dir_structure)) or 'Flat structure'}\n"
        )

        tools_info = []
        for t in ALL_TOOLS:
            tools_info.append(f"- **{t['name']}**: {t['description']}")
        tools_str = "\n".join(tools_info)

        self.system_prompt = f"""You are an autonomous AI coding assistant running inside a repository harness.

{architecture_context}

### Available Tools:
{tools_str}

### Instructions:
1. You can call tools natively, or output one or more structured tool call
   blocks in a single reply:
<tool_call>
{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}
</tool_call>
2. All tool calls in a turn are executed before you see any results.
3. Always search or read files before attempting edits.
4. The repo's syntax tree is large — prefer search_symbols / get_file_outline
   first, then get_adjacent_nodes or get_node(depth=1) to zoom in on a
   specific match rather than dumping whole subtrees.
5. When finished or answering non-code questions, reply with plain text without tool calls.
"""
        self.messages.append({"role": "system", "content": self.system_prompt})

    def enforce_target_dir(self, args: Dict[str, Any]) -> Dict[str, Any]:
        path_keys = {"path", "directory", "file_pattern", "file_path", "file_glob"}
        for k, v in args.items():
            if k in path_keys and isinstance(v, str):
                clean_path = v.lstrip("./\\")
                if clean_path == "":
                    args[k] = self.target_dir
                elif not clean_path.startswith(f"{self.target_dir}/") and clean_path != self.target_dir:
                    args[k] = f"{self.target_dir}/{clean_path}"
        return args

    def parse_text_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Parse zero or more <tool_call>...</tool_call> blocks from a text reply."""
        if not text:
            return []

        calls: List[Dict[str, Any]] = []
        for match in re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
            try:
                calls.append(json.loads(match.group(1).strip()))
            except json.JSONDecodeError:
                pass
        if calls:
            return calls

        match_json = re.search(r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}', text, re.DOTALL)
        if match_json:
            try:
                return [json.loads(match_json.group(0).strip())]
            except json.JSONDecodeError:
                pass

        return []

    def run_stream(self, user_task: str):
        """
        Generator version of the agent loop: yields one event dict per step
        so a caller (CLI printer, web GUI via SSE) can render progressively
        instead of blocking until the whole multi-step task finishes.

        Event shapes:
          {"type": "user", "content": str}
          {"type": "step_start", "step": int, "prompt_role": str, "prompt_content": str}
          {"type": "assistant", "content": str, "provider": str, "model": str}
          {"type": "tool_call", "name": str, "arguments": dict}
          {"type": "tool_result", "name": str, "result": Any}
          {"type": "final", "content": str}
          {"type": "error", "content": str}
          {"type": "max_iterations"}
        """
        self.messages.append({"role": "user", "content": user_task})
        yield {"type": "user", "content": user_task}

        for step in range(self.max_iterations):
            latest_msg = self.messages[-1]
            yield {
                "type": "step_start",
                "step": step + 1,
                "prompt_role": latest_msg["role"],
                "prompt_content": latest_msg["content"].strip(),
            }

            try:
                text_response, native_tool_calls, model_info = chat_with_fallback(self.messages)
            except RuntimeError as exc:
                yield {"type": "error", "content": str(exc)}
                return

            tool_calls = native_tool_calls or self.parse_text_tool_calls(text_response)

            assistant_content = text_response if text_response else (
                f"Calling tools: {', '.join(tc.get('name', '?') for tc in tool_calls)}"
                if tool_calls else ""
            )
            self.messages.append({"role": "assistant", "content": assistant_content})
            yield {
                "type": "assistant",
                "content": text_response or "",
                "provider": model_info["provider"],
                "model": model_info["model"],
            }

            if tool_calls:
                observations = []
                for tool_call in tool_calls:
                    tool_name = tool_call.get("name")
                    raw_args = tool_call.get("arguments", {})
                    tool_args = self.enforce_target_dir(raw_args)

                    yield {"type": "tool_call", "name": tool_name, "arguments": tool_args}

                    try:
                        result = call_tool(tool_name, tool_args)
                        # Truncate *before* building the observation string,
                        # not just at display time. Otherwise the full
                        # untruncated dump goes into self.messages and the
                        # context balloons silently across steps.
                        observation = json.dumps(result, indent=2)[:functions.MAX_OUTPUT_BYTES]
                    except Exception as e:
                        result = {"error": f"{type(e).__name__}: {e}"}
                        observation = f"Error executing {tool_name}: {e}"

                    yield {"type": "tool_result", "name": tool_name, "result": result}
                    observations.append(f"Tool Output for {tool_name}:\n{observation}")

                obs_message = "\n\n".join(observations)
                self.messages.append({"role": "user", "content": obs_message})
            else:
                yield {"type": "final"}
                return

        yield {"type": "max_iterations"}

    def run(self, user_task: str):
        """CLI entry point: drives run_stream() and prints each event."""
        print("\n" + "=" * 50)
        print(f"[Initial Task] {user_task}")
        print("=" * 50)

        for event in self.run_stream(user_task):
            et = event["type"]

            if et == "user":
                continue  # already printed above

            elif et == "step_start":
                print("\n" + "-" * 20 + f" Step {event['step']} " + "-" * 20)
                print(f"\n[Prompt Sent to Model ({event['prompt_role']})]")
                print(event["prompt_content"])
                print("-" * 50)

            elif et == "assistant":
                print(f"\n[Model Raw Reply] (answered by {event['provider']}:{event['model']})")
                if event["content"]:
                    print(f"Message: {event['content'].strip()}")
                else:
                    print("Message: <Empty (Model chose to emit a native tool call only)>")
                print("-" * 50)

            elif et == "tool_call":
                print(f"\n🛠️  Executing Tool: {event['name']}")
                print(f"Arguments: {json.dumps(event['arguments'], indent=2)}")

            elif et == "tool_result":
                print("\n[Tool Output Result]")
                print(json.dumps(event["result"], indent=2)[:functions.MAX_OUTPUT_BYTES])
                print("=" * 50)

            elif et == "final":
                print("\n[Final Answer Reached]")

            elif et == "error":
                print(f"\n[Error] {event['content']}")

            elif et == "max_iterations":
                print(f"\n[Terminated] Hit max iteration limit ({self.max_iterations}).")


if __name__ == "__main__":
    agent = HarnessAgent()
    agent.run("List the files in the repository and summarize what this project does.")