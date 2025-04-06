#!/usr/bin/env python3
"""
Supabase Memory Store for Single-File Agents
Purpose: Provides persistent memory storage for agents using Supabase PostgreSQL
Version: 1.0.0

Usage:
  Import into agent scripts to provide Supabase-backed memory capabilities
  
Requirements:
  - Python 3.8+
  - pip install supabase
"""

import os
import sys
import json
import time
import hashlib
import pickle
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# Constants
DEFAULT_TABLE_NAME = "agent_memories"

class SupabaseMemoryStore:
    """Memory store implementation using Supabase PostgreSQL."""
    
    def __init__(self, agent_id: str):
        """Initialize the Supabase memory store.
        
        Args:
            agent_id: Unique identifier for the agent
        """
        self.agent_id = agent_id
        self.supabase: Client = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )
        
    def remember(self, data: Dict[str, Any], key: Optional[str] = None) -> str:
        """Store data in memory.
        
        Args:
            data: The data to store
            key: Optional key to associate with the data
            
        Returns:
            str: The ID of the stored memory
        """
        memory = {
            "agent_id": self.agent_id,
            "data": data,
            "key": key,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        result = self.supabase.table("agent_memories").insert(memory).execute()
        return result.data[0]["id"]
    
    def recall(self, key: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve data from memory.
        
        Args:
            key: Optional key to filter memories by
            
        Returns:
            List[Dict[str, Any]]: List of matching memories
        """
        query = self.supabase.table("agent_memories").select("*").eq("agent_id", self.agent_id)
        
        if key:
            query = query.eq("key", key)
            
        result = query.order("created_at", desc=True).execute()
        return result.data
    
    def forget(self, memory_id: str) -> bool:
        """Remove a specific memory.
        
        Args:
            memory_id: The ID of the memory to remove
            
        Returns:
            bool: True if successful
        """
        result = self.supabase.table("agent_memories").delete().eq("id", memory_id).execute()
        return len(result.data) > 0
    
    def clear(self) -> bool:
        """Clear all memories for this agent.
        
        Returns:
            bool: True if successful
        """
        result = self.supabase.table("agent_memories").delete().eq("agent_id", self.agent_id).execute()
        return len(result.data) > 0
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search memories using semantic search.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of matching memories
        """
        result = self.supabase.rpc(
            "search_memories",
            {
                "query": query,
                "agent_id": self.agent_id,
                "limit": limit
            }
        ).execute()
        return result.data

# Factory function to register with memory_store
def get_supabase_memory_store(agent_id: str, **kwargs) -> SupabaseMemoryStore:
    """Create a Supabase memory store for the given agent"""
    return SupabaseMemoryStore(agent_id, **kwargs)

# Example usage
if __name__ == "__main__":
    # Check if Supabase is available
    if not os.environ.get("SUPABASE_URL") or not os.environ.get("SUPABASE_KEY"):
        print("Please set SUPABASE_URL and SUPABASE_KEY environment variables")
        sys.exit(1)
    
    # Create a memory store
    memory = SupabaseMemoryStore("test_agent")
    
    # Save some data
    key = memory.remember({"test": "data"}, tags=["example"])
    print(f"Saved data with key: {key}")
    
    # Load the data
    data = memory.recall(key)
    print(f"Loaded data: {data}")
    
    # List all keys
    keys = memory.recall()
    print(f"All keys: {keys}")
    
    # Search by tags
    results = memory.search("example")
    print(f"Search results: {results}")
    
    # Delete the data
    memory.forget(key)
    print(f"Deleted key: {key}") 