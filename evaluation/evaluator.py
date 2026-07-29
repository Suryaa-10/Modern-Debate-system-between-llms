from typing import Dict, Any, List

class DebateEvaluator:
    """Evaluates the quality and outcome of a debate."""
    
    def __init__(self):
        pass

    def evaluate(self, debate_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes the output of a debate execution and scores it based on various metrics.
        
        Args:
            debate_result (Dict): The full result dictionary from ExecutionController.
            
        Returns:
            Dict: Evaluation metrics.
        """
        history = debate_result.get("history", [])
        final_consensus = debate_result.get("final_consensus", {})
        
        metrics = {
            "total_rounds": debate_result.get("rounds_completed", 0),
            "confidence_score": final_consensus.get("confidence_score", 0.0),
            "final_answer": final_consensus.get("final_answer", "N/A"),
            "agent_participation": self._measure_participation(history)
        }
        
        metrics["overall_score"] = metrics["confidence_score"]
        return metrics

    def _measure_participation(self, history: List[Dict[str, Any]]) -> Dict[str, int]:
        """Counts how many arguments/critiques each agent made."""
        participation = {}
        for round_data in history:
            data = round_data.get("data", {})
            for agent, payload in data.items():
                participation[agent] = participation.get(agent, 0) + 1
        return participation
