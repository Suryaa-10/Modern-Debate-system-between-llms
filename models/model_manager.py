import os
import json
import re
import asyncio
from openai import AsyncOpenAI
from .model_config import MODEL_1, MODEL_2, MODEL_3

class ModelManager:
    """Manages interactions with language models using native clients to avoid compilation dependencies."""
    
    def __init__(self):
        self.models = {
            "agent_1": MODEL_1,
            "agent_2": MODEL_2,
            "agent_3": MODEL_3,
        }
        
        # Initialize native clients
        self.groq_client = AsyncOpenAI(
            api_key=os.getenv('GROQ_API_KEY'), 
            base_url='https://api.groq.com/openai/v1'
        )
        self.or_client = AsyncOpenAI(
            api_key=os.getenv('OPENROUTER_API_KEY'), 
            base_url='https://openrouter.ai/api/v1'
        )

    async def generate_response(
        self, 
        agent_name: str, 
        messages: list, 
        response_schema: dict = None, 
        retries: int = 3,
        **kwargs
    ) -> str:
        if agent_name not in self.models:
            raise ValueError(f"Unknown agent: {agent_name}")
            
        model_id = self.models[agent_name]
        
        # Prepare JSON instructions if a schema is provided
        current_messages = list(messages)
        if response_schema:
            schema_str = json.dumps(response_schema)
            sys_msg = {
                "role": "system",
                "content": f"You MUST respond ONLY with valid JSON that matches this schema:\n{schema_str}\nDo not include markdown blocks like ```json."
            }
            # Insert system message at the beginning
            current_messages.insert(0, sys_msg)

        # Simple routing logic: if model ID has a slash, it's an OpenRouter model. Otherwise, Groq.
        client = self.or_client if "/" in model_id else self.groq_client

        last_exception = None
        for attempt in range(retries):
            try:
                resp = await client.chat.completions.create(
                    model=model_id,
                    messages=current_messages,
                    temperature=kwargs.get('temperature', 0.7),
                )
                raw_response = resp.choices[0].message.content
                    
                # Cleanup potential markdown wrapper for JSON
                if response_schema:
                    raw_response = re.sub(r'```json\n|```\n|```', '', raw_response).strip()
                    
                return raw_response
            except Exception as e:
                last_exception = e
                print(f"[Warning] Attempt {attempt + 1}/{retries} failed for {agent_name} ({model_id}): {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)  # 1s, 2s backoff

        raise RuntimeError(f"Error generating response from {model_id} after {retries} retries: {last_exception}")

    def get_model_name(self, agent_name: str) -> str:
        return self.models.get(agent_name)
