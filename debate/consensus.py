import json
from typing import Dict, Any
from .state import DebateState
from models.model_manager import ModelManager
from schemas.response_schema import ConsensusSchema

class ConsensusEngine:
    """Uses a third agent to evaluate the debate and reach a consensus."""
    
    def __init__(self, manager: ModelManager):
        self.manager = manager

    async def evaluate_and_conclude(self, state: DebateState) -> Dict[str, Any]:
        """
        Evaluates the entire debate history using the consensus agent (agent_3).
        """
        print(f"\n--- Final Round: Consensus Evaluation ---")
        
        agent = "agent_3"
        
        # Prepare the full history summary for the consensus agent
        debate_transcript = f"Original Query: {state.query}\n\n"
        for i, round_record in enumerate(state.history):
            debate_transcript += f"--- Round {i+1} ---\n"
            debate_transcript += json.dumps(round_record["data"], indent=2) + "\n\n"
            
        system_prompt = (
            "You are an impartial judge and consensus synthesizer. "
            "Review the full multi-round debate transcript between 3 AI agents. "
            "Identify key points of agreement, remaining disagreements, and synthesize the final consensus result."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here is the full debate transcript across all 3 models:\n{debate_transcript}"}
        ]
        
        response_json_str = await self.manager.generate_response(
            agent_name=agent,
            messages=messages,
            response_schema=ConsensusSchema.model_json_schema(),
            temperature=0.2 # Lower temperature for grounded synthesis
        )
        
        response_data = json.loads(response_json_str)
        
        # Save to state
        state.final_consensus = response_data
        state.is_completed = True
        
        print(f"\n[{agent}] Final Consensus Synthesized | Confidence Score: {response_data.get('confidence_score')}")
        return response_data
