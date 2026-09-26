import os
import json
import re
import time
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
from google import genai
from openai import OpenAI
from pathlib import Path

# Import tools from functions.py
from . import functions 
from .repository_tree_sitter_scanner import RepositoryScanner

load_dotenv()

# ==========================================
# Configuration
# ==========================================
TARGET_DIR = "test"

# ==========================================
# 1. Client Setup
# ==========================================
google_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# Convert functions.TOOLS into standard OpenAI JSON schemas
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"]
        }
    }
    for tool in functions.TOOLS
]

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
    # ==========================================
    # OpenRouter (Free Tier)
    # Note: The ":free" suffix is MANDATORY. 
    # OpenRouter rotates these based on sponsorships.
    # ==========================================
    ("openrouter", "qwen/qwen3.8-27b:free"),
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
def chat_with_fallback(messages: List[Dict[str, str]]) -> Tuple[str, Optional[Dict[str, Any]]]:
    for provider, model in MODEL_PIPELINE:
        if TRACKER.is_failed(provider, model):
            continue

        try:
            if provider in ["groq", "openrouter"]:
                client = groq_client if provider == "groq" else openrouter_client
                res = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=OPENAI_TOOLS,
                    tool_choice="auto",
                    temperature=0.2
                )
                choice = res.choices[0]
                text = choice.message.content or ""

                if choice.message.tool_calls:
                    tc = choice.message.tool_calls[0]
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    return text, {"name": tc.function.name, "arguments": args}

                return text, None

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
                        "temperature": 0.2
                    }
                )
                return res.text or "", None

        except Exception as e:
            err_type = type(e).__name__
            print(f"  [API Error] {provider} -> {model} failed ({err_type}: {e})")
            TRACKER.mark_failed(provider, model, err_type)

    raise RuntimeError(f"All models failed or were rate-limited. Summary: {TRACKER.status_summary()}")

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
        for t in functions.TOOLS:
            tools_info.append(f"- **{t['name']}**: {t['description']}")
        tools_str = "\n".join(tools_info)

        self.system_prompt = f"""You are an autonomous AI coding assistant running inside a repository harness.

{architecture_context}

### Available Tools:
{tools_str}

### Instructions:
1. You can call tools natively or output a structured tool call block:
<tool_call>
{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}
</tool_call>
2. Execute ONE tool per turn and wait for the tool output observation.
3. Always search or read files before attempting edits.
4. When finished or answering non-code questions, reply with plain text without tool calls.
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

    def parse_text_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None

        match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        match_json = re.search(r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}', text, re.DOTALL)
        if match_json:
            try:
                return json.loads(match_json.group(0).strip())
            except json.JSONDecodeError:
                pass

        return None

    def run(self, user_task: str):
        self.messages.append({"role": "user", "content": user_task})
        
        print("\n" + "="*50)
        print(f"[Initial Task] {user_task}")
        print("="*50)

        for step in range(self.max_iterations):
            print(f"\n" + "-"*20 + f" Step {step + 1} " + "-"*20)
            
            # Print the latest prompt being sent to the model (excluding system prompt for brevity)
            latest_msg = self.messages[-1]
            print(f"\n[Prompt Sent to Model ({latest_msg['role']})]")
            print(latest_msg["content"].strip())
            print("-" * 50)

            text_response, native_tool_call = chat_with_fallback(self.messages)
            
            # Print raw model reply
            print(f"\n[Model Raw Reply]")
            if text_response:
                print(f"Message: {text_response.strip()}")
            else:
                print("Message: <Empty (Model chose to emit a native tool call only)>")
                
            if native_tool_call:
                print(f"Tool Call: {json.dumps(native_tool_call)}")
            print("-" * 50)

            tool_call = native_tool_call or self.parse_text_tool_call(text_response)

            assistant_content = text_response if text_response else f"Calling tool: {tool_call.get('name')}"
            self.messages.append({"role": "assistant", "content": assistant_content})

            if tool_call:
                tool_name = tool_call.get("name")
                raw_args = tool_call.get("arguments", {})
                
                tool_args = self.enforce_target_dir(raw_args)
                
                print(f"\n🛠️  Executing Tool: {tool_name}")
                print(f"Arguments: {json.dumps(tool_args, indent=2)}")

                try:
                    result = functions.call_tool(tool_name, tool_args)
                    observation = json.dumps(result, indent=2)[:functions.MAX_OUTPUT_BYTES]
                except Exception as e:
                    observation = f"Error executing {tool_name}: {e}"

                # Print the raw tool output
                print(f"\n[Tool Output Result]")
                print(observation)
                print("=" * 50)

                obs_message = f"Tool Output for {tool_name}:\n{observation}"
                self.messages.append({"role": "user", "content": obs_message})

            else:
                print("\n[Final Answer Reached]")
                break
        else:
            print(f"\n[Terminated] Hit max iteration limit ({self.max_iterations}).")


if __name__ == "__main__":
    agent = HarnessAgent()
    agent.run("List the files in the repository and summarize what this project does.")