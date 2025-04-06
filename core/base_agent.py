from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
from datetime import datetime
import os
from dotenv import load_dotenv

class BaseAgent(ABC):
    """Base class for all agents with core functionality."""
    
    def __init__(self, agent_id: str, memory_type: str = "mem0"):
        """Initialize the agent.
        
        Args:
            agent_id: Unique identifier for the agent
            memory_type: Type of memory to use (mem0, supabase, sqlite, etc.)
        """
        self.agent_id = agent_id
        self.memory_type = memory_type
        self.memory = self._initialize_memory()
        load_dotenv()
        
    def _initialize_memory(self):
        """Initialize the appropriate memory store."""
        if self.memory_type == "mem0":
            from utils.mem0_adapter import Mem0MemoryStore
            return Mem0MemoryStore(self.agent_id)
        elif self.memory_type == "supabase":
            from utils.supabase_memory import SupabaseMemoryStore
            return SupabaseMemoryStore(self.agent_id)
        else:
            from utils.memory_store import MemoryStore
            return MemoryStore(self.agent_id, self.memory_type)
    
    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """Main execution method for the agent.
        
        Args:
            **kwargs: Additional arguments specific to the agent
            
        Returns:
            Dict[str, Any]: Results of the agent's execution
        """
        pass
    
    def remember(self, data: Dict[str, Any], key: Optional[str] = None) -> str:
        """Store data in memory.
        
        Args:
            data: The data to store
            key: Optional key to associate with the data
            
        Returns:
            str: The ID of the stored memory
        """
        return self.memory.remember(data, key)
    
    def recall(self, key: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve data from memory.
        
        Args:
            key: Optional key to filter memories by
            
        Returns:
            List[Dict[str, Any]]: List of matching memories
        """
        return self.memory.recall(key)
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search memories using semantic search.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of matching memories
        """
        return self.memory.search(query, limit)
    
    def forget(self, memory_id: str) -> bool:
        """Remove a specific memory.
        
        Args:
            memory_id: The ID of the memory to remove
            
        Returns:
            bool: True if successful
        """
        return self.memory.forget(memory_id)
    
    def clear(self) -> bool:
        """Clear all memories for this agent.
        
        Returns:
            bool: True if successful
        """
        return self.memory.clear() 