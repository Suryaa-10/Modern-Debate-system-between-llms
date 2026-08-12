"""
Model Configuration for Multi-LLM Debate Framework

This module sets up provider-independent LLM configuration.
"""

import os
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

# Define primary models to be used in the debate (diverse model families)
MODEL_1 = os.getenv("MODEL_1", "llama-3.3-70b-versatile")
MODEL_2 = os.getenv("MODEL_2", "qwen/qwen-2.5-72b-instruct")
MODEL_3 = os.getenv("MODEL_3", "llama-3.1-8b-instant")
MODEL_JUDGE = os.getenv("MODEL_JUDGE", "llama-3.3-70b-versatile")

# Multi-level fallback chains for each agent slot (tries each in order on 400/404/429 errors)
AGENT_MODEL_CHAINS = {
    "agent_1": [
        MODEL_1,
        "llama-3.1-8b-instant",
        "llama-3.2-3b-preview",
        "llama3-8b-8192"
    ],
    "agent_2": [
        MODEL_2,
        "llama-3.1-8b-instant",       # Fast, reliable failover on Groq
        "llama-3.3-70b-versatile",    # 70B Groq failover
        "meta-llama/llama-3.3-70b-instruct",
        "qwen/qwen-2.5-72b-instruct"
    ],
    "agent_3": [
        MODEL_3,
        "llama3-8b-8192",
        "llama-3.3-70b-versatile"
    ],
    "agent_judge": [
        MODEL_JUDGE,
        "llama-3.1-8b-instant",
        "llama-3.2-3b-preview"
    ]
}

# Optional: Add custom API base if using local servers like Ollama or LMStudio
CUSTOM_API_BASE = os.getenv("CUSTOM_API_BASE", None)

