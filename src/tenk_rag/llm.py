"""LLM calls (Anthropic Messages API). Swap this file to use another provider."""
from __future__ import annotations

import json
import os
import re
import time


class LLM:
    def __init__(self, model: str, max_tokens: int = 800):
        import anthropic
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise EnvironmentError("Set ANTHROPIC_API_KEY to use the LLM steps.")
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.usage = {"input_tokens": 0, "output_tokens": 0}

    def complete(self, system: str, prompt: str, temperature: float = 0.0) -> str:
        for attempt in range(4):
            try:
                resp = self.client.messages.create(
                    model=self.model, max_tokens=self.max_tokens, temperature=temperature,
                    system=system, messages=[{"role": "user", "content": prompt}],
                )
                self.usage["input_tokens"] += resp.usage.input_tokens
                self.usage["output_tokens"] += resp.usage.output_tokens
                return "".join(b.text for b in resp.content if b.type == "text")
            except Exception as e:  # rate limits / transient errors
                if attempt == 3:
                    raise
                print(f"[llm] retrying after error: {e}")
                time.sleep(2 ** attempt * 2)
        return ""

    def complete_json(self, system: str, prompt: str):
        text = self.complete(system, prompt)
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON in model output: {text[:200]}")
        return json.loads(match.group(1))
