import asyncio
from typing import Dict, Any
from .model_manager import ModelManager

async def check_model_health(agent_name: str, manager: ModelManager) -> Dict[str, Any]:
    """
    Checks if a specific model is responsive.
    """
    model_name = manager.get_model_name(agent_name)
    messages = [{"role": "user", "content": "Ping. Reply with 'pong'."}]
    
    try:
        response = await manager.generate_response(
            agent_name=agent_name,
            messages=messages,
            max_tokens=5,
            timeout=10 # Short timeout for health check
        )
        is_healthy = "pong" in response.lower()
        return {
            "agent": agent_name,
            "model": model_name,
            "status": "Healthy" if is_healthy else "Unhealthy (Invalid Response)",
            "response": response
        }
    except Exception as e:
        return {
            "agent": agent_name,
            "model": model_name,
            "status": f"Unhealthy (Error: {str(e)})",
            "response": None
        }

async def run_all_health_checks() -> None:
    """
    Runs health checks for all configured agents.
    """
    manager = ModelManager()
    print(f"{'='*40}")
    print("Running LLM Health Checks...")
    print(f"{'='*40}")
    
    tasks = [check_model_health(agent, manager) for agent in manager.models]
    results = await asyncio.gather(*tasks)
    
    for result in results:
        print(f"Agent : {result['agent']}")
        print(f"Model : {result['model']}")
        print(f"Status: {result['status']}")
        if result['status'].startswith("Unhealthy"):
            print(f"Details: {result['response']}")
        print("-" * 40)

if __name__ == "__main__":
    asyncio.run(run_all_health_checks())
