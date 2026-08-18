import json
from typing import Dict, Any
from .state import DebateState
from models.model_manager import ModelManager
from schemas.response_schema import ConsensusSchema

class ConsensusEngine:
    """Uses a dedicated, decoupled consensus judge agent (agent_judge) and confidence-weighted majority voting to synthesize consensus."""
    
    def __init__(self, manager: ModelManager):
        self.manager = manager

    def _compute_confidence_weighted_majority(self, state: DebateState) -> Dict[str, Any]:
        """
        Item 6: Computes confidence-weighted vote tally across agents in the final round.
        Returns the top answer, total confidence weight, and average confidence.
        """
        if not state.history:
            return {"top_answer": "N/A", "total_weight": 0.0, "avg_confidence": 0.0, "is_unanimous": False}

        final_round_data = state.history[-1]["data"]
        tally = {}
        conf_sums = {}
        counts = {}

        for agent, data in final_round_data.items():
            ans = str(data.get("final_answer", "")).strip()
            conf = float(data.get("confidence_score", 0.0))
            if ans and ans != "N/A":
                tally[ans] = tally.get(ans, 0.0) + conf
                conf_sums[ans] = conf_sums.get(ans, 0.0) + conf
                counts[ans] = counts.get(ans, 0) + 1

        if not tally:
            return {"top_answer": "N/A", "total_weight": 0.0, "avg_confidence": 0.0, "is_unanimous": False}

        top_answer = max(tally, key=tally.get)
        avg_confidence = conf_sums[top_answer] / counts[top_answer] if counts.get(top_answer) else 0.0
        is_unanimous = (len(tally) == 1 and counts[top_answer] >= 2)

        return {
            "top_answer": top_answer,
            "total_weight": tally[top_answer],
            "avg_confidence": avg_confidence,
            "is_unanimous": is_unanimous,
            "vote_tally": tally
        }

    async def evaluate_and_conclude(self, state: DebateState) -> Dict[str, Any]:
        """
        Evaluates the debate transcript using an impartial, dedicated consensus judge (agent_judge).
        Incorporates confidence-weighted vote tallies.
        """
        print(f"\n--- Final Phase: Impartial Consensus Evaluation ---")
        
        # Item 5: Use decoupled dedicated judge (agent_judge)
        judge_agent = "agent_judge"
        
        majority_info = self._compute_confidence_weighted_majority(state)
        
        # Prepare structured debate transcript for judge synthesis
        debate_transcript = f"Original Query: {state.query}\n\n"
        for i, round_record in enumerate(state.history):
            debate_transcript += f"--- Round {round_record.get('round', i+1)} ---\n"
            round_data = round_record.get("data", {})
            for agent_name, payload in round_data.items():
                debate_transcript += f"[{agent_name}] Answer: {payload.get('final_answer')} (Conf: {payload.get('confidence_score')})\n"
                debate_transcript += f"Reasoning: {payload.get('reasoning')}\n\n"
            
        system_prompt = (
            "You are an impartial, independent judge and consensus synthesizer. "
            "Review the multi-round debate transcript between AI agents. "
            "Identify key points of agreement, resolve remaining discrepancies, and produce the definitive final consensus. "
            "Ensure the final_answer field contains ONLY the exact, concise target answer (e.g. 'A', 'B', '42', or concise direct answer)."
        )
        
        user_prompt = (
            f"Here is the full debate transcript:\n{debate_transcript}\n"
            f"Confidence-Weighted Majority Tally Hint: Top Candidate = '{majority_info['top_answer']}' "
            f"with confidence weight = {majority_info['total_weight']:.2f}.\n\n"
            "Provide your impartial synthesis matching the required JSON schema."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response_json_str = await self.manager.generate_response(
                agent_name=judge_agent,
                messages=messages,
                response_schema=ConsensusSchema.model_json_schema(),
                temperature=0.2 # Lower temperature for grounded synthesis
            )
            response_data = json.loads(response_json_str)
        except Exception as e:
            print(f"[Warning] Judge synthesis encountered error: {e}. Falling back to confidence-weighted majority.")
            response_data = {
                "query": state.query,
                "reasoning": f"Consensus derived via confidence-weighted majority vote due to judge API timeout: {e}",
                "final_answer": majority_info["top_answer"],
                "confidence_score": round(majority_info["avg_confidence"], 2)
            }
        
        # Save to state
        state.final_consensus = response_data
        state.is_completed = True
        
        print(f"\n[{judge_agent}] Impartial Consensus Synthesized | Final Answer: '{response_data.get('final_answer')}' | Confidence Score: {response_data.get('confidence_score')}")
        return response_data
