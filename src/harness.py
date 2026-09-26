import os
import json
import re
import time
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
from google import genai
from openai import OpenAI

# Import tools from functions.py
import functions 

load_dotenv()

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
    """
    Tries each model in the pipeline.
    Passes tools natively to prevent OpenRouter/Groq 400 proxy validation errors.
    Returns: (text_content, tool_call_dict)
    """
    for provider, model in MODEL_PIPELINE:
        if TRACKER.is_failed(provider, model):
            continue

        try:
            print(f"  [API] Trying {provider} -> {model}...")

            if provider in ["groq", "openrouter"]:
                client = groq_client if provider == "groq" else openrouter_client
                res = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=OPENAI_TOOLS,       # Passed natively to satisfy API proxy rules
                    tool_choice="auto",
                    temperature=0.2
                )
                choice = res.choices[0]
                text = choice.message.content or ""

                # Handle native tool calls if generated
                if choice.message.tool_calls:
                    tc = choice.message.tool_calls[0]
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    return text, {"name": tc.function.name, "arguments": args}

                return text, None

            elif provider == "google":
                # Format messages for Google GenAI SDK
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
    def __init__(self, max_iterations: int = 15):
        self.max_iterations = max_iterations
        self.messages = []
        
        # Build tool context for system prompt
        tools_info = []
        for t in functions.TOOLS:
            tools_info.append(f"- **{t['name']}**: {t['description']}")
        tools_str = "\n".join(tools_info)

        self.system_prompt = f"""You are an autonomous AI coding assistant running inside a repository harness.

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

    def parse_text_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """Fallback parser for models that format tool calls in text."""
        if not text:
            return None

        # Look for <tool_call>...</tool_call> tags
        match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Look for raw JSON objects with "name" and "arguments"
        match_json = re.search(r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}', text, re.DOTALL)
        if match_json:
            try:
                return json.loads(match_json.group(0).strip())
            except json.JSONDecodeError:
                pass

        return None

    def run(self, user_task: str):
        self.messages.append({"role": "user", "content": user_task})
        print(f"\n[Task] {user_task}\n")

        for step in range(self.max_iterations):
            print(f"--- Step {step + 1} ---")

            # 1. Call API with model rotation
            text_response, native_tool_call = chat_with_fallback(self.messages)

            # 2. Determine tool call (Native takes priority, text parsing acts as fallback)
            tool_call = native_tool_call or self.parse_text_tool_call(text_response)

            # Append assistant's thoughts/text to message history
            assistant_content = text_response if text_response else f"Calling tool: {tool_call.get('name')}"
            self.messages.append({"role": "assistant", "content": assistant_content})

            if tool_call:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("arguments", {})
                print(f"🛠️  Executing: {tool_name}({json.dumps(tool_args)})")

                # 3. Execute tool via functions dispatch
                try:
                    result = functions.call_tool(tool_name, tool_args)
                    observation = json.dumps(result, indent=2)[:functions.MAX_OUTPUT_BYTES]
                except Exception as e:
                    observation = f"Error executing {tool_name}: {e}"

                # 4. Feed observation back to conversation history
                obs_message = f"Tool Output for {tool_name}:\n{observation}"
                self.messages.append({"role": "user", "content": obs_message})
                print(f"✅ Result received ({len(obs_message)} chars). Proceeding...\n")

            else:
                # No tool call means final response reached
                print("\n[Final Answer]")
                print(text_response)
                break
        else:
            print(f"\n[Terminated] Hit max iteration limit ({self.max_iterations}).")


if __name__ == "__main__":
    agent = HarnessAgent()
    agent.run("List the files in the repository and summarize what this project does.")