#!/usr/bin/env python3
"""
Mem0 Adapter for Single-File Agents
Purpose: Provides integration with Mem0 AI for agent memory and state persistence
Version: 1.0.0

Usage:
  Import into agent scripts to provide Mem0-based memory capabilities
  
Requirements:
  - Python 3.8+
  - Mem0 AI installed (git clone https://github.com/mem0ai/mem0.git)
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
import traceback

# Add mem0 to path if available
MEM0_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "memory-store/mem0")
if os.path.exists(MEM0_PATH):
    sys.path.append(MEM0_PATH)

# Import Mem0 modules
try:
    from mem0.models import EntityMetadata, Memory, Entity, Relation
    from mem0.client import Mem0Client
    from mem0.query import Query
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    print("Warning: Mem0 AI not found. Install it with: git clone https://github.com/mem0ai/mem0.git")

# Import local memory store if available
try:
    from memory_store import MemoryStore
    MEMORY_STORE_AVAILABLE = True
except ImportError:
    MEMORY_STORE_AVAILABLE = False

class Mem0MemoryStore:
    """Memory store implementation using Mem0 AI"""
    
    def __init__(self, agent_id: str, mem0_path: str = MEM0_PATH, db_path: Optional[str] = None):
        """
        Initialize the Mem0 memory store
        
        Args:
            agent_id: Unique identifier for the agent
            mem0_path: Path to the Mem0 installation
            db_path: Optional path to the database file
        """
        if not MEM0_AVAILABLE:
            raise ImportError("Mem0 AI is not installed. Install it with: git clone https://github.com/mem0ai/mem0.git")
        
        self.agent_id = agent_id
        
        # Set up the database path
        if db_path is None:
            # Use a default location in the memory-store directory
            db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                   f"memory-store/mem0_data/{agent_id}.sqlite")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize Mem0 client
        self.client = Mem0Client(url=f"sqlite:///{db_path}")
        
        # Create agent entity if it doesn't exist
        self._ensure_agent_entity()
    
    def _ensure_agent_entity(self):
        """Create the agent entity in Mem0 if it doesn't exist"""
        # Check if agent entity exists
        query = Query().filter(
            Entity.metadata.name == self.agent_id,
            Entity.metadata.category == "agent"
        )
        
        agents = self.client.search_entities(query=query)
        
        if not agents:
            # Create agent entity
            metadata = EntityMetadata(
                name=self.agent_id,
                category="agent",
                description=f"Agent with ID {self.agent_id}",
                created_at=datetime.now()
            )
            
            agent_entity = Entity(metadata=metadata)
            self.client.create_entity(agent_entity)
            
            return agent_entity
        else:
            return agents[0]
    
    def _get_agent_entity(self):
        """Get the agent entity from Mem0"""
        query = Query().filter(
            Entity.metadata.name == self.agent_id,
            Entity.metadata.category == "agent"
        )
        
        agents = self.client.search_entities(query=query)
        
        if agents:
            return agents[0]
        else:
            return self._ensure_agent_entity()
    
    def save(self, data: Any, key: Optional[str] = None, tags: List[str] = None) -> str:
        """
        Save data to Mem0
        
        Args:
            data: Data to save
            key: Optional identifier for the data
            tags: Optional tags for the data
            
        Returns:
            Generated or provided key
        """
        # Generate key if not provided
        if key is None:
            timestamp = int(time.time())
            key = f"memory_{timestamp}"
        
        # Get agent entity
        agent_entity = self._get_agent_entity()
        
        # Prepare content
        if isinstance(data, dict):
            content = json.dumps(data)
        elif isinstance(data, (list, tuple)):
            content = json.dumps(data)
        else:
            content = str(data)
        
        # Create memory entity
        metadata = EntityMetadata(
            name=key,
            category="memory",
            tags=tags or [],
            description=f"Memory for agent {self.agent_id}",
            created_at=datetime.now()
        )
        
        memory_entity = Entity(metadata=metadata)
        
        # Create memory
        memory_content = Memory(
            content=content,
            source=f"agent:{self.agent_id}"
        )
        
        # Save memory entity and content
        memory_id = self.client.create_entity(memory_entity)
        self.client.add_memory(memory_id, memory_content)
        
        # Create relation from agent to memory
        relation = Relation(
            source_id=agent_entity.id, 
            target_id=memory_id,
            type="has_memory",
            metadata={
                "timestamp": datetime.now().isoformat(),
                "key": key
            }
        )
        
        self.client.create_relation(relation)
        
        return key
    
    def load(self, key: str) -> Any:
        """
        Load data from Mem0 by key
        
        Args:
            key: Key for the data to load
            
        Returns:
            Loaded data or None if not found
        """
        # Get agent entity
        agent_entity = self._get_agent_entity()
        
        # Query for memory entities with the given key
        query = Query().filter(
            Entity.metadata.name == key,
            Entity.metadata.category == "memory"
        )
        
        memory_entities = self.client.search_entities(query=query)
        
        if not memory_entities:
            return None
        
        # Get the memory content
        memory_entity = memory_entities[0]
        memories = self.client.get_memories(memory_entity.id)
        
        if not memories:
            return None
        
        # Parse the content
        content = memories[0].content
        
        try:
            # Try to parse as JSON
            return json.loads(content)
        except:
            # Return as string if not JSON
            return content
    
    def search_by_tags(self, tags: List[str]) -> List[Dict[str, Any]]:
        """
        Search for memories by tags
        
        Args:
            tags: List of tags to search for
            
        Returns:
            List of memories matching the tags
        """
        results = []
        
        for tag in tags:
            query = Query().filter(
                Entity.metadata.category == "memory",
                Entity.metadata.tags.contains(tag)
            )
            
            memory_entities = self.client.search_entities(query=query)
            
            for entity in memory_entities:
                memories = self.client.get_memories(entity.id)
                
                if memories:
                    try:
                        content = json.loads(memories[0].content)
                    except:
                        content = memories[0].content
                    
                    results.append({
                        "key": entity.metadata.name,
                        "data": content,
                        "tags": entity.metadata.tags,
                        "timestamp": entity.metadata.created_at.isoformat()
                    })
        
        return results
    
    def list_keys(self) -> List[str]:
        """
        List all memory keys for this agent
        
        Returns:
            List of memory keys
        """
        # Get agent entity
        agent_entity = self._get_agent_entity()
        
        # Get all relations from agent to memories
        relations = self.client.get_relations(
            source_id=agent_entity.id,
            type="has_memory"
        )
        
        # Extract keys from relations
        keys = []
        for relation in relations:
            if relation.metadata and "key" in relation.metadata:
                keys.append(relation.metadata["key"])
            
            # If key not in metadata, try to get the memory entity
            else:
                memory_entity = self.client.get_entity(relation.target_id)
                if memory_entity and memory_entity.metadata.name:
                    keys.append(memory_entity.metadata.name)
        
        return keys
    
    def delete(self, key: str) -> bool:
        """
        Delete a memory by key
        
        Args:
            key: Key of the memory to delete
            
        Returns:
            True if deleted, False otherwise
        """
        # Find memory entity
        query = Query().filter(
            Entity.metadata.name == key,
            Entity.metadata.category == "memory"
        )
        
        memory_entities = self.client.search_entities(query=query)
        
        if not memory_entities:
            return False
        
        # Delete memory entity
        for entity in memory_entities:
            self.client.delete_entity(entity.id)
        
        return True
    
    def clear(self) -> bool:
        """
        Clear all memories for this agent
        
        Returns:
            True if successful
        """
        # Get agent entity
        agent_entity = self._get_agent_entity()
        
        # Get all relations from agent to memories
        relations = self.client.get_relations(
            source_id=agent_entity.id,
            type="has_memory"
        )
        
        # Delete all memory entities
        for relation in relations:
            try:
                self.client.delete_entity(relation.target_id)
            except:
                pass
        
        return True
    
    def semantic_search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search for memories
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of matching memories
        """
        # Use Mem0 search
        search_query = Query().search(query)
        search_query = search_query.filter(Entity.metadata.category == "memory")
        
        memory_entities = self.client.search_entities(
            query=search_query, 
            limit=n_results
        )
        
        results = []
        for entity in memory_entities:
            memories = self.client.get_memories(entity.id)
            
            if memories:
                try:
                    content = json.loads(memories[0].content)
                except:
                    content = memories[0].content
                
                results.append({
                    "key": entity.metadata.name,
                    "data": content,
                    "tags": entity.metadata.tags,
                    "timestamp": entity.metadata.created_at.isoformat()
                })
        
        return results
    
    def save_interaction(self, user_input: str, agent_response: str, tags: List[str] = None) -> str:
        """
        Save an interaction between user and agent
        
        Args:
            user_input: User input text
            agent_response: Agent response text
            tags: Optional tags for the interaction
            
        Returns:
            Key of the saved interaction
        """
        # Create interaction data
        timestamp = int(time.time())
        key = f"interaction_{timestamp}"
        
        interaction_data = {
            "user_input": user_input,
            "agent_response": agent_response,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save using standard save method
        return self.save(interaction_data, key, tags=tags or ["interaction"])

class Mem0AgentAdapter:
    """Adapter class for integrating Mem0 with agents"""
    
    def __init__(self, agent_id: str, mem0_store: Optional[Mem0MemoryStore] = None):
        """Initialize the Mem0 agent adapter"""
        if not MEM0_AVAILABLE:
            raise ImportError("Mem0 AI is not installed. Install it with: git clone https://github.com/mem0ai/mem0.git")
        
        self.agent_id = agent_id
        
        # Create or use provided Mem0 store
        if mem0_store:
            self.memory = mem0_store
        else:
            self.memory = Mem0MemoryStore(agent_id)
        
        # Initialize conversation history
        self.conversation_history = []
    
    def remember(self, data: Any, key: Optional[str] = None, tags: List[str] = None) -> str:
        """
        Remember (save) data to memory
        
        Args:
            data: Data to remember
            key: Optional identifier
            tags: Optional tags
            
        Returns:
            Memory key
        """
        return self.memory.save(data, key, tags)
    
    def recall(self, key: str) -> Any:
        """
        Recall (load) data from memory
        
        Args:
            key: Memory key
            
        Returns:
            Retrieved data
        """
        return self.memory.load(key)
    
    def recall_by_tags(self, tags: List[str]) -> List[Dict[str, Any]]:
        """
        Recall memories by tags
        
        Args:
            tags: List of tags
            
        Returns:
            List of matching memories
        """
        return self.memory.search_by_tags(tags)
    
    def forget(self, key: str) -> bool:
        """
        Forget (delete) a memory
        
        Args:
            key: Memory key
            
        Returns:
            True if forgotten
        """
        return self.memory.delete(key)
    
    def forget_all(self) -> bool:
        """
        Forget all memories
        
        Returns:
            True if successful
        """
        return self.memory.clear()
    
    def reflect(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search on memories
        
        Args:
            query: Search query
            n_results: Number of results
            
        Returns:
            Matching memories
        """
        return self.memory.semantic_search(query, n_results)
    
    def add_conversation(self, user_input: str, agent_response: str) -> str:
        """
        Add a conversation exchange to memory
        
        Args:
            user_input: User input text
            agent_response: Agent response text
            
        Returns:
            Memory key
        """
        # Add to conversation history
        self.conversation_history.append({
            "user": user_input,
            "agent": agent_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Save to Mem0
        return self.memory.save_interaction(user_input, agent_response)
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get the conversation history
        
        Returns:
            List of conversation exchanges
        """
        return self.conversation_history
    
    def save_agent_state(self, state: Dict[str, Any]) -> str:
        """
        Save the agent's current state
        
        Args:
            state: Agent state data
            
        Returns:
            Memory key
        """
        return self.memory.save(state, "agent_state", tags=["state"])
    
    def load_agent_state(self) -> Dict[str, Any]:
        """
        Load the agent's saved state
        
        Returns:
            Agent state data
        """
        state = self.memory.load("agent_state")
        return state if state else {}
    
    def save_learning_data(self, data: Dict[str, Any], key: Optional[str] = None) -> str:
        """
        Save learning data for the agent
        
        Args:
            data: Learning data
            key: Optional identifier
            
        Returns:
            Memory key
        """
        if key is None:
            timestamp = int(time.time())
            key = f"learning_{timestamp}"
        
        return self.memory.save(data, key, tags=["learning"])
    
    def get_learning_data(self) -> List[Dict[str, Any]]:
        """
        Get all learning data
        
        Returns:
            List of learning data entries
        """
        return self.memory.search_by_tags(["learning"])

# Example usage
if __name__ == "__main__":
    if not MEM0_AVAILABLE:
        print("Mem0 AI not available. Please install it first.")
        sys.exit(1)
    
    # Create Mem0 adapter
    adapter = Mem0AgentAdapter("test_agent")
    
    # Save some data
    key = adapter.remember({"test": "data"}, tags=["example"])
    print(f"Saved data with key: {key}")
    
    # Recall the data
    data = adapter.recall(key)
    print(f"Recalled data: {data}")
    
    # Add some conversation
    adapter.add_conversation("Hello, agent!", "Hello, human!")
    adapter.add_conversation("How are you?", "I'm doing well, thank you!")
    
    # Get conversation history
    history = adapter.get_conversation_history()
    print(f"Conversation history: {history}")
    
    # Search memories
    results = adapter.reflect("hello")
    print(f"Search results: {results}")
    
    # Save and load agent state
    adapter.save_agent_state({"mood": "happy", "last_action": "greeting"})
    state = adapter.load_agent_state()
    print(f"Agent state: {state}")
    
    # Clean up
    adapter.forget(key)
    print(f"Deleted key: {key}") 