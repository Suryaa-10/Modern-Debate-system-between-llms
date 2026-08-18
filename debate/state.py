from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class DebateState(BaseModel):
    """
    Manages the state of a multi-agent debate session with context window efficiency and early exit tracking.
    """
    query: str = Field(..., description="The original query or topic of debate.")
    round_number: int = Field(0, description="The current round number.")
    max_rounds: int = Field(..., description="Maximum number of debate rounds.")
    
    # Store history as a list of round summaries
    history: List[Dict[str, Any]] = Field(default_factory=list, description="Record of the debate rounds.")
    
    # Keep track of individual agent histories
    agent_contexts: Dict[str, List[Dict[str, str]]] = Field(
        default_factory=dict, 
        description="Message history for each agent."
    )
    
    is_completed: bool = Field(False, description="True if the debate has concluded.")
    early_exit: bool = Field(False, description="True if debate terminated early via dynamic exit tripwire.")
    final_consensus: Optional[Dict[str, Any]] = Field(None, description="The final consensus output.")

    def add_agent_message(self, agent_name: str, role: str, content: str):
        """Adds a message to a specific agent's context history."""
        if agent_name not in self.agent_contexts:
            self.agent_contexts[agent_name] = []
        self.agent_contexts[agent_name].append({"role": role, "content": content})

    def get_agent_history(self, agent_name: str, max_messages: int = 6) -> List[Dict[str, str]]:
        """
        Retrieves optimized message history for a specific agent.
        Keeps system messages & initial prompt, plus the most recent N context messages.
        Prevents O(R^2) context window overflow and token waste.
        """
        msgs = self.agent_contexts.get(agent_name, [])
        if len(msgs) <= max_messages:
            return msgs

        # Preserve system / setup messages (usually the first 2 msgs)
        prefix = [m for m in msgs[:2] if m.get("role") in ["system", "user"]]
        suffix = msgs[-(max_messages - len(prefix)):]
        return prefix + suffix
    
    def advance_round(self):
        """Increments the round number and checks for completion."""
        self.round_number += 1
        if self.round_number >= self.max_rounds:
            self.is_completed = True

    def record_round_data(self, round_data: Dict[str, Any]):
        """Saves a summary or full data dump of the current round to history."""
        self.history.append({
            "round": self.round_number,
            "data": round_data
        })
