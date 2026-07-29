import asyncio
import json
from typing import Dict, Any, Optional
from models.model_manager import ModelManager
from debate.state import DebateState
from debate.engine import DebateEngine
from debate.consensus import ConsensusEngine

class ExecutionController:
    """Controls the overall flow of the debate for a given query."""
    
    def __init__(self, manager: Optional[ModelManager] = None):
        self.manager = manager or ModelManager()
        self.engine = DebateEngine(self.manager)
        self.consensus = ConsensusEngine(self.manager)
        
    async def run_full_debate(self, query: str, max_rounds: int = 7) -> Dict[str, Any]:
        """
        Executes a full multi-round debate followed by consensus.
        """
        state = DebateState(query=query, max_rounds=max_rounds)
        
        # Round 1: Initial Arguments
        await self.engine.run_initial_arguments(state)
        
        # Subsequent Rounds: Critiques
        while not state.is_completed and state.round_number < max_rounds:
            await self.engine.run_critique_round(state)
            
        # Final Phase: Consensus
        final_result = await self.consensus.evaluate_and_conclude(state)
        
        return {
            "query": query,
            "rounds_completed": state.round_number,
            "history": state.history,
            "final_consensus": final_result
        }

async def execute_query(query: str, output_file: Optional[str] = None):
    """Utility function to run a debate from CLI or simple scripts."""
    controller = ExecutionController()
    print(f"Starting debate for query: '{query}'")
    
    try:
        result = await controller.run_full_debate(query)
        print("\n=== FINAL DEBATE RESULTS ===")
        print(json.dumps(result["final_consensus"], indent=2))
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nSaved full debate trace to {output_file}")
            
        return result
    except Exception as e:
        print(f"An error occurred during debate execution: {e}")
        raise e

if __name__ == "__main__":
    test_query = "What is the most effective renewable energy source for the next decade?"
    asyncio.run(execute_query(test_query))
