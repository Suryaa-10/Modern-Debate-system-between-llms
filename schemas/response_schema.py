from pydantic import BaseModel, Field
from typing import List, Optional

class ArgumentSchema(BaseModel):
    """Schema for an initial argument presented by an LLM agent."""
    agent_name: str = Field(..., description="Name of the agent presenting the argument.")
    reasoning: str = Field(..., description="Step-by-step reasoning supporting the conclusion.")
    final_answer: str = Field(..., description="Single direct answer (e.g., MCQ option like 'A' or concise direct answer).")
    confidence_score: float = Field(..., description="Confidence in the answer between 0.0 and 1.0.", ge=0.0, le=1.0)

class CritiqueSchema(BaseModel):
    """Schema for a critique and position refinement during debate rounds."""
    agent_name: str = Field(..., description="Name of the agent providing the critique.")
    reasoning: str = Field(..., description="Detailed critique of other agents' logic and step-by-step updated reasoning.")
    final_answer: str = Field(..., description="Single direct answer (e.g., MCQ option like 'A' or concise direct answer).")
    confidence_score: float = Field(..., description="Confidence in the answer between 0.0 and 1.0.", ge=0.0, le=1.0)

class ConsensusSchema(BaseModel):
    """Schema for the final synthesized consensus output."""
    query: str = Field(..., description="The original query that was debated.")
    reasoning: str = Field(..., description="Comprehensive reasoning synthesizing all points of agreement and disagreement across 7 rounds.")
    final_answer: str = Field(..., description="Single direct consensus answer (e.g., MCQ option like 'A' or concise direct answer).")
    confidence_score: float = Field(..., description="Consensus confidence score between 0.0 and 1.0.", ge=0.0, le=1.0)

