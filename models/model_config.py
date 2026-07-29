"""
Model Configuration for Multi-LLM Debate Framework

This module sets up provider-independent LLM configuration.
"""

import os
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

# Define the models to be used in the debate
MODEL_1 = os.getenv("MODEL_1", "llama-3.1-8b-instant")
MODEL_2 = os.getenv("MODEL_2", "google/gemma-4-26b-a4b-it:free")
MODEL_3 = os.getenv("MODEL_3", "meta-llama/llama-3.1-8b-instruct")

# Optional: Add custom API base if using local servers like Ollama or LMStudio
CUSTOM_API_BASE = os.getenv("CUSTOM_API_BASE", None)
