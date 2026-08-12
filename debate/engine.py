import json
import asyncio
from typing import Dict, Any, Tuple
from .state import DebateState
from models.model_manager import ModelManager
from schemas.response_schema import ArgumentSchema, CritiqueSchema

# Item 8: Distinct Role-Differentiated Personas for Debaters
AGENT_ROLES = {
    "agent_1": (
        "You are Agent 1 (Analytical & Fact-Focused Debater). "
        "Present a logically rigorous, step-by-step argument based strictly on facts and first-principles reasoning."
    ),
    "agent_2": (
        "You are Agent 2 (Skeptical & Adversarial Debater). "
        "Act as a critical reviewer. Probe potential flaws, edge cases, fallacies, counter-arguments, and hidden assumptions."
    ),
    "agent_3": (
        "You are Agent 3 (Synthesizing & Methodical Debater). "
        "Focus on double-checking mathematical and logical calculations, balance, and formal verification."
    )
}

class DebateEngine:
    """Orchestrates the 3-agent multi-round debate process with role differentiation, failure isolation, and dynamic early-exit routing."""
    
    def __init__(self, manager: ModelManager):
        self.manager = manager
        self.agents = ["agent_1", "agent_2", "agent_3"]

    async def _generate_agent_argument(self, agent: str, state: DebateState) -> Tuple[str, Dict[str, Any]]:
        role_prompt = AGENT_ROLES.get(
            agent, 
            "You are an expert AI participating in a 3-way debate. Present a strong initial argument."
        )
        
        state.add_agent_message(agent, "system", role_prompt)
        state.add_agent_message(agent, "user", state.query)
        
        history = state.get_agent_history(agent)
        
        response_json_str = await self.manager.generate_response(
            agent_name=agent,
            messages=history,
            response_schema=ArgumentSchema.model_json_schema(),
            temperature=0.7
        )
        
        response_data = json.loads(response_json_str)
        state.add_agent_message(agent, "assistant", response_json_str)
        return agent, response_data

    async def run_initial_arguments(self, state: DebateState) -> Dict[str, Any]:
        """Runs Round 1 where all 3 models independently present their initial arguments with failure isolation."""
        print(f"\n--- Round 1: Initial Arguments for query: '{state.query}' ---")
        
        tasks = [self._generate_agent_argument(agent, state) for agent in self.agents]
        # Item 1 & 3: return_exceptions=True so one agent failure doesn't sink the round
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        round_data = {}
        for idx, agent in enumerate(self.agents):
            res = results[idx]
            if isinstance(res, Exception):
                print(f"[Warning] {agent} failed in Round 1: {res}. Using low-confidence placeholder.")
                response_data = {
                    "agent_name": agent,
                    "reasoning": f"Agent unavailable: {str(res)}",
                    "final_answer": "N/A",
                    "confidence_score": 0.0
                }
            else:
                _, response_data = res

            round_data[agent] = response_data
            model_name = self.manager.get_model_name(agent)
            print(f"[{agent} ({model_name})] Final Answer: '{response_data.get('final_answer')}' | Confidence: {response_data.get('confidence_score')}")

        state.record_round_data(round_data)
        state.advance_round()

        # Item 11: Dynamic Early-Exit Routing Check
        self._check_early_exit(state, round_data)
        return round_data

    async def _generate_agent_critique(self, agent: str, state: DebateState, previous_round_data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        other_agents_data = {
            other: previous_round_data[other] 
            for other in self.agents if other != agent and other in previous_round_data
        }
        
        role_prompt = AGENT_ROLES.get(agent, "")
        prompt = (
            f"Round {state.round_number + 1} of Debate:\n"
            f"Role directive: {role_prompt}\n\n"
            "Review the arguments and final answers presented by your fellow debaters in the previous round:\n"
            f"{json.dumps(other_agents_data, indent=2)}\n\n"
            "Critique their reasoning, defend or refine your own answer, and provide your step-by-step reasoning and single direct final_answer."
        )
        
        state.add_agent_message(agent, "user", prompt)
        history = state.get_agent_history(agent)
        
        response_json_str = await self.manager.generate_response(
            agent_name=agent,
            messages=history,
            response_schema=CritiqueSchema.model_json_schema(),
            temperature=0.7
        )
        
        response_data = json.loads(response_json_str)
        state.add_agent_message(agent, "assistant", response_json_str)
        return agent, response_data

    async def run_critique_round(self, state: DebateState) -> Dict[str, Any]:
        """Runs a debate round where all 3 models exchange critiques with failure isolation."""
        print(f"\n--- Round {state.round_number + 1}: Iterative Debate Round ---")
        
        previous_round_data = state.history[-1]["data"]
        
        tasks = [self._generate_agent_critique(agent, state, previous_round_data) for agent in self.agents]
        # Item 1 & 3: return_exceptions=True
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        round_data = {}
        for idx, agent in enumerate(self.agents):
            res = results[idx]
            if isinstance(res, Exception):
                print(f"[Warning] {agent} failed in Round {state.round_number + 1}: {res}. Using low-confidence placeholder.")
                response_data = {
                    "agent_name": agent,
                    "reasoning": f"Agent unavailable: {str(res)}",
                    "final_answer": "N/A",
                    "confidence_score": 0.0
                }
            else:
                _, response_data = res

            round_data[agent] = response_data
            model_name = self.manager.get_model_name(agent)
            print(f"[{agent} ({model_name})] Refined Answer: '{response_data.get('final_answer')}' | Confidence: {response_data.get('confidence_score')}")

        state.record_round_data(round_data)
        state.advance_round()

        # Item 11: Dynamic Early-Exit Routing Check
        self._check_early_exit(state, round_data)
        return round_data

    def _check_early_exit(self, state: DebateState, round_data: Dict[str, Any]):
        """
        Item 7 & 11: Dynamic Early-Exit Routing
        Terminates the debate loop immediately upon multi-agent agreement.
        """
        valid_answers = []
        confidences = []
        for agent, data in round_data.items():
            ans = str(data.get("final_answer", "")).strip()
            conf = float(data.get("confidence_score", 0.0))
            if ans and ans != "N/A" and conf > 0.0:
                valid_answers.append(ans)
                confidences.append(conf)

        if len(valid_answers) >= 2:
            # Check if all valid agents agree on the exact same answer
            first_ans = valid_answers[0]
            all_agree = all(ans == first_ans for ans in valid_answers)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

            if all_agree and avg_conf >= 0.60:
                print(f"\n⚡ [Dynamic Early Exit Tripwire] All active agents agreed on '{first_ans}' with average confidence {avg_conf:.2f} in Round {state.round_number}. Terminating debate loop early.")
                state.is_completed = True
                state.early_exit = True
