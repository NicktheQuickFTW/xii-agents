from typing import Any, Dict, List, Optional
from core.base_agent import BaseAgent
import os
from dotenv import load_dotenv

class BaseMetaAgent(BaseAgent):
    """Base class for meta agents that can perform specialized tasks."""
    
    def __init__(self, agent_id: str, agent_type: str, memory_type: str = "mem0"):
        """Initialize the meta agent.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of meta agent (prompt, scraper, etc.)
            memory_type: Type of memory to use
        """
        super().__init__(agent_id, memory_type)
        self.agent_type = agent_type
        
    def generate_prompt(self, instruction: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate a prompt based on instruction and context.
        
        Args:
            instruction: The instruction for prompt generation
            context: Optional context information
            
        Returns:
            str: Generated prompt
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a specialized task.
        
        Args:
            task: Task specification with parameters
            
        Returns:
            Dict[str, Any]: Task results
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def compose_agents(self, agents: List[BaseAgent], task: Dict[str, Any]) -> Dict[str, Any]:
        """Compose multiple agents to complete a complex task.
        
        Args:
            agents: List of agents to compose
            task: The complex task to complete
            
        Returns:
            Dict[str, Any]: Task results
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def evaluate_result(self, result: Dict[str, Any], criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a result against specified criteria.
        
        Args:
            result: The result to evaluate
            criteria: Evaluation criteria
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        raise NotImplementedError("Subclasses must implement this method") 