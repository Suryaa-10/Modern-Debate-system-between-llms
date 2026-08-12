import os
import json
import re
import random
import asyncio
from typing import Dict, Any, Optional
from openai import AsyncOpenAI
from .model_config import (
    MODEL_1, MODEL_2, MODEL_3, MODEL_JUDGE,
    AGENT_MODEL_CHAINS
)

def extract_and_repair_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extracts and repairs JSON objects from raw model responses using brace-matching and regex cleanup.
    """
    if not text:
        return None

    # Step 1: Strip markdown code blocks
    cleaned = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
    cleaned = re.sub(r'```\s*', '', cleaned).strip()

    # Step 2: Attempt direct json.loads
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Step 3: Extract first complete {...} block using brace matching
    start_idx = cleaned.find('{')
    if start_idx != -1:
        brace_count = 0
        end_idx = -1
        in_string = False
        escape = False
        
        for i in range(start_idx, len(cleaned)):
            char = cleaned[i]
            if escape:
                escape = False
                continue
            if char == '\\':
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i
                        break
        
        if end_idx != -1:
            json_candidate = cleaned[start_idx:end_idx + 1]
            try:
                data = json.loads(json_candidate)
                if isinstance(data, dict):
                    return data
            except Exception:
                # Step 4: Repair common JSON syntax flaws (trailing commas)
                repaired = re.sub(r',\s*([\}\]])', r'\1', json_candidate)
                try:
                    data = json.loads(repaired)
                    if isinstance(data, dict):
                        return data
                except Exception:
                    pass

    # Step 5: Fallback regex search for outermost braces
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        candidate = match.group(0)
        repaired = re.sub(r',\s*([\}\]])', r'\1', candidate)
        try:
            data = json.loads(repaired)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return None


class ModelManager:
    """Manages interactions with language models using native clients with multi-level fallbacks, JSON repair, and adaptive response_format."""
    
    def __init__(self):
        self.models = {
            "agent_1": MODEL_1,
            "agent_2": MODEL_2,
            "agent_3": MODEL_3,
            "agent_judge": MODEL_JUDGE,
        }
        self.model_chains = AGENT_MODEL_CHAINS
        
        # Concurrency semaphore to prevent hammering free tier APIs
        self.semaphore = asyncio.Semaphore(4)
        
        # Initialize native clients
        self.groq_client = AsyncOpenAI(
            api_key=os.getenv('GROQ_API_KEY'), 
            base_url='https://api.groq.com/openai/v1'
        )
        self.or_client = AsyncOpenAI(
            api_key=os.getenv('OPENROUTER_API_KEY'), 
            base_url='https://openrouter.ai/api/v1'
        )

    def _get_client_for_model(self, model_id: str) -> AsyncOpenAI:
        return self.or_client if "/" in model_id else self.groq_client

    async def generate_response(
        self, 
        agent_name: str, 
        messages: list, 
        response_schema: dict = None, 
        retries: int = 5,
        **kwargs
    ) -> str:
        if agent_name not in self.models and agent_name not in self.model_chains:
            raise ValueError(f"Unknown agent: {agent_name}")
            
        model_chain = self.model_chains.get(agent_name, [self.models.get(agent_name)])
        
        # Prepare JSON instructions if a schema is provided
        current_messages = list(messages)
        if response_schema:
            schema_str = json.dumps(response_schema)
            sys_msg = {
                "role": "system",
                "content": f"You MUST respond ONLY with valid JSON that matches this schema:\n{schema_str}\nDo not include markdown blocks like ```json."
            }
            # Insert system message at the beginning if not already present
            if not any(m.get("role") == "system" and "schema" in m.get("content", "").lower() for m in current_messages):
                current_messages.insert(0, sys_msg)

        last_exception = None
        use_response_format = True if response_schema else False
        
        async with self.semaphore:
            for attempt in range(retries):
                # Cycle through the model fallback chain
                model_id = model_chain[attempt % len(model_chain)]
                client = self._get_client_for_model(model_id)

                call_kwargs = {
                    "model": model_id,
                    "messages": current_messages,
                    "temperature": kwargs.get('temperature', 0.7),
                }
                if kwargs.get('max_tokens'):
                    call_kwargs["max_tokens"] = kwargs.get('max_tokens')

                # Native structured output / JSON mode support if enabled and supported
                if use_response_format:
                    call_kwargs["response_format"] = {"type": "json_object"}

                try:
                    resp = await client.chat.completions.create(**call_kwargs)
                    raw_response = resp.choices[0].message.content or ""
                    
                    if response_schema:
                        parsed = extract_and_repair_json(raw_response)
                        if parsed is None:
                            raise ValueError(f"Malformed or unparseable JSON response from {model_id}: {raw_response[:100]}...")
                        return json.dumps(parsed)
                    
                    return raw_response
                except Exception as e:
                    last_exception = e
                    err_str = str(e)
                    print(f"[Warning] Attempt {attempt + 1}/{retries} failed for {agent_name} using model {model_id}: {e}")
                    
                    # Fallback feature flag: if provider returns 400 or rejects response_format json_object, disable response_format parameter
                    if "json_object" in err_str.lower() or "response_format" in err_str.lower() or "400" in err_str:
                        use_response_format = False
                    
                    if attempt < retries - 1:
                        backoff = (1.5 ** attempt) + random.uniform(0.1, 1.0)
                        if "429" in err_str or "rate limit" in err_str.lower():
                            backoff += 2.0  # Extra backoff for rate limits
                        await asyncio.sleep(backoff)

        raise RuntimeError(f"Error generating response from {agent_name} (chain: {model_chain}) after {retries} retries: {last_exception}")

    def get_model_name(self, agent_name: str) -> str:
        chain = self.model_chains.get(agent_name, [self.models.get(agent_name)])
        return chain[0] if chain else agent_name
