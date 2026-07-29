import json
import asyncio
from typing import Dict, Any
from .state import DebateState
from models.model_manager import ModelManager
from schemas.response_schema import ArgumentSchema, CritiqueSchema

class DebateEngine:
    """Orchestrates the 3-agent multi-round debate process as defined in the system workflow."""
    
    def __init__(self, manager: ModelManager):
        self.manager = manager
        self.agents = ["agent_1", "agent_2", "agent_3"]

    async def _generate_agent_argument(self, agent: str, state: DebateState, system_prompt: str) -> tuple[str, Dict[str, Any]]:
        state.add_agent_message(agent, "system", system_prompt)
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
        """Runs Round 1 where all 3 models independently present their initial arguments."""
        print(f"\n--- Round 1: Initial Arguments for query: '{state.query}' ---")
        
        system_prompt = (
            "You are an expert AI participating in a 3-way debate. "
            "Present a strong, well-reasoned initial argument based on the user's query."
        )

        tasks = [self._generate_agent_argument(agent, state, system_prompt) for agent in self.agents]
        results = await asyncio.gather(*tasks)
        
        round_data = {}
        for agent, response_data in results:
            round_data[agent] = response_data
            model_name = self.manager.get_model_name(agent)
            print(f"[{agent} ({model_name})] Final Answer: '{response_data.get('final_answer')}' | Confidence: {response_data.get('confidence_score')}")

        state.record_round_data(round_data)
        state.advance_round()
        return round_data

    async def _generate_agent_critique(self, agent: str, state: DebateState, previous_round_data: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        other_agents_data = {
            other: previous_round_data[other] 
            for other in self.agents if other != agent and other in previous_round_data
        }
        
        prompt = (
            f"Round {state.round_number + 1} of Debate:\n"
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
        """Runs a debate round where all 3 models exchange arguments & counter-arguments."""
        print(f"\n--- Round {state.round_number + 1}: Iterative Debate Round ---")
        
        previous_round_data = state.history[-1]["data"]
        
        tasks = [self._generate_agent_critique(agent, state, previous_round_data) for agent in self.agents]
        results = await asyncio.gather(*tasks)
        
        round_data = {}
        for agent, response_data in results:
            round_data[agent] = response_data
            model_name = self.manager.get_model_name(agent)
            print(f"[{agent} ({model_name})] Refined Answer: '{response_data.get('final_answer')}' | Confidence: {response_data.get('confidence_score')}")

        state.record_round_data(round_data)
        state.advance_round()
        return round_data

