#!/usr/bin/env python3
"""
Memory Store Utility for Single-File Agents
Purpose: Provides persistent memory storage for agents with support for different storage backends
Version: 1.0.0

Usage:
  Import into agent scripts to provide memory capabilities
  
Requirements:
  - Python 3.8+
  - pip install duckdb sqlite3 chromadb (or use uv)
"""

import os
import sys
import json
import time
import hashlib
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import sqlite3
import pickle

# Constants
DEFAULT_MEMORY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "memory-store")

# Try importing optional dependencies
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# Try importing Supabase store
try:
    from supabase_memory import SupabaseMemoryStore, get_supabase_memory_store
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

class MemoryStore:
    """Base class for agent memory storage systems"""
    
    def __init__(self, agent_id: str, memory_dir: str = DEFAULT_MEMORY_DIR):
        """Initialize the memory store"""
        self.agent_id = agent_id
        self.memory_dir = memory_dir
        
        # Ensure memory directory exists
        os.makedirs(memory_dir, exist_ok=True)
        
    def save(self, data: Any, key: Optional[str] = None) -> str:
        """Save data to memory with optional key"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def load(self, key: str) -> Any:
        """Load data from memory by key"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def list_keys(self) -> List[str]:
        """List all available memory keys"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def delete(self, key: str) -> bool:
        """Delete a memory entry by key"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def clear(self) -> bool:
        """Clear all memory for this agent"""
        raise NotImplementedError("Subclasses must implement this method")

class FileMemoryStore(MemoryStore):
    """Simple file-based memory store"""
    
    def __init__(self, agent_id: str, memory_dir: str = DEFAULT_MEMORY_DIR):
        super().__init__(agent_id, memory_dir)
        self.agent_dir = os.path.join(self.memory_dir, f"file_{self.agent_id}")
        os.makedirs(self.agent_dir, exist_ok=True)
    
    def _get_path(self, key: str) -> str:
        """Get the file path for a given key"""
        return os.path.join(self.agent_dir, f"{key}.json")
    
    def save(self, data: Any, key: Optional[str] = None) -> str:
        """Save data to a JSON file"""
        if key is None:
            # Generate key based on timestamp and content hash
            timestamp = int(time.time())
            content_hash = hashlib.md5(str(data).encode()).hexdigest()[:8]
            key = f"{timestamp}_{content_hash}"
        
        file_path = self._get_path(key)
        
        # Add metadata
        metadata = {
            "data": data,
            "timestamp": datetime.now().isoformat(),
            "agent_id": self.agent_id
        }
        
        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return key
    
    def load(self, key: str) -> Any:
        """Load data from a JSON file"""
        file_path = self._get_path(key)
        
        if not os.path.exists(file_path):
            return None
        
        with open(file_path, 'r') as f:
            metadata = json.load(f)
            
        return metadata.get("data")
    
    def list_keys(self) -> List[str]:
        """List all memory keys"""
        files = os.listdir(self.agent_dir)
        return [f.replace('.json', '') for f in files if f.endswith('.json')]
    
    def delete(self, key: str) -> bool:
        """Delete a memory entry"""
        file_path = self._get_path(key)
        
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        
        return False
    
    def clear(self) -> bool:
        """Clear all memory for this agent"""
        for key in self.list_keys():
            self.delete(key)
        return True

class SQLiteMemoryStore(MemoryStore):
    """SQLite-based memory store for more structured data"""
    
    def __init__(self, agent_id: str, memory_dir: str = DEFAULT_MEMORY_DIR):
        super().__init__(agent_id, memory_dir)
        self.db_path = os.path.join(self.memory_dir, f"sqlite_{self.agent_id}.db")
        self._init_db()
    
    def _init_db(self):
        """Initialize the SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create memory table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS memory (
            key TEXT PRIMARY KEY,
            data BLOB,
            timestamp TEXT,
            tags TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def save(self, data: Any, key: Optional[str] = None, tags: List[str] = None) -> str:
        """Save data to SQLite with optional tags"""
        if key is None:
            # Generate key based on timestamp and content hash
            timestamp = int(time.time())
            content_hash = hashlib.md5(str(data).encode()).hexdigest()[:8]
            key = f"{timestamp}_{content_hash}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Serialize data
        serialized_data = pickle.dumps(data)
        
        # Join tags if provided
        tags_str = ','.join(tags) if tags else ''
        
        # Insert or replace data
        cursor.execute(
            "INSERT OR REPLACE INTO memory (key, data, timestamp, tags) VALUES (?, ?, ?, ?)",
            (key, serialized_data, datetime.now().isoformat(), tags_str)
        )
        
        conn.commit()
        conn.close()
        
        return key
    
    def load(self, key: str) -> Any:
        """Load data from SQLite by key"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT data FROM memory WHERE key = ?", (key,))
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            return pickle.loads(result[0])
        
        return None
    
    def search_by_tags(self, tags: List[str]) -> List[Dict[str, Any]]:
        """Search for memory entries by tags"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        results = []
        for tag in tags:
            cursor.execute(
                "SELECT key, data, timestamp, tags FROM memory WHERE tags LIKE ?", 
                (f"%{tag}%",)
            )
            
            for row in cursor.fetchall():
                key, data_blob, timestamp, tags_str = row
                results.append({
                    "key": key,
                    "data": pickle.loads(data_blob),
                    "timestamp": timestamp,
                    "tags": tags_str.split(',') if tags_str else []
                })
        
        conn.close()
        return results
    
    def list_keys(self) -> List[str]:
        """List all memory keys"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT key FROM memory")
        keys = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return keys
    
    def delete(self, key: str) -> bool:
        """Delete a memory entry by key"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM memory WHERE key = ?", (key,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        
        return deleted
    
    def clear(self) -> bool:
        """Clear all memory for this agent"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM memory")
        
        conn.commit()
        conn.close()
        
        return True

class DuckDBMemoryStore(MemoryStore):
    """DuckDB-based memory store for analytical data"""
    
    def __init__(self, agent_id: str, memory_dir: str = DEFAULT_MEMORY_DIR):
        if not DUCKDB_AVAILABLE:
            raise ImportError("DuckDB is not installed. Install it with: pip install duckdb")
            
        super().__init__(agent_id, memory_dir)
        self.db_path = os.path.join(self.memory_dir, f"duckdb_{self.agent_id}.db")
        self._init_db()
    
    def _init_db(self):
        """Initialize the DuckDB database"""
        conn = duckdb.connect(self.db_path)
        
        # Create memory table if it doesn't exist
        conn.execute('''
        CREATE TABLE IF NOT EXISTS memory (
            key VARCHAR PRIMARY KEY,
            data BLOB,
            timestamp VARCHAR,
            metadata JSON
        )
        ''')
        
        conn.close()
    
    def save(self, data: Any, key: Optional[str] = None, metadata: Dict[str, Any] = None) -> str:
        """Save data to DuckDB with optional metadata"""
        if key is None:
            # Generate key based on timestamp and content hash
            timestamp = int(time.time())
            content_hash = hashlib.md5(str(data).encode()).hexdigest()[:8]
            key = f"{timestamp}_{content_hash}"
        
        conn = duckdb.connect(self.db_path)
        
        # Serialize data
        serialized_data = pickle.dumps(data)
        
        # Convert metadata to JSON string
        metadata_json = json.dumps(metadata) if metadata else '{}'
        
        # Insert or replace data
        conn.execute(
            "INSERT OR REPLACE INTO memory VALUES (?, ?, ?, ?)",
            (key, serialized_data, datetime.now().isoformat(), metadata_json)
        )
        
        conn.close()
        
        return key
    
    def load(self, key: str) -> Any:
        """Load data from DuckDB by key"""
        conn = duckdb.connect(self.db_path)
        
        result = conn.execute("SELECT data FROM memory WHERE key = ?", [key]).fetchone()
        
        conn.close()
        
        if result:
            return pickle.loads(result[0])
        
        return None
    
    def query(self, sql: str) -> List[Dict[str, Any]]:
        """Run a custom SQL query against the memory store"""
        conn = duckdb.connect(self.db_path)
        
        results = conn.execute(sql).fetchall()
        column_names = [desc[0] for desc in conn.description]
        
        conn.close()
        
        return [dict(zip(column_names, row)) for row in results]
    
    def list_keys(self) -> List[str]:
        """List all memory keys"""
        conn = duckdb.connect(self.db_path)
        
        keys = conn.execute("SELECT key FROM memory").fetchall()
        
        conn.close()
        
        return [key[0] for key in keys]
    
    def delete(self, key: str) -> bool:
        """Delete a memory entry by key"""
        conn = duckdb.connect(self.db_path)
        
        conn.execute("DELETE FROM memory WHERE key = ?", [key])
        
        conn.close()
        return True
    
    def clear(self) -> bool:
        """Clear all memory for this agent"""
        conn = duckdb.connect(self.db_path)
        
        conn.execute("DELETE FROM memory")
        
        conn.close()
        return True

class VectorMemoryStore(MemoryStore):
    """Vector database memory store for semantic search capabilities"""
    
    def __init__(self, agent_id: str, memory_dir: str = DEFAULT_MEMORY_DIR):
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is not installed. Install it with: pip install chromadb")
            
        super().__init__(agent_id, memory_dir)
        self.db_path = os.path.join(self.memory_dir, f"vector_{self.agent_id}")
        
        # Initialize ChromaDB client and collection
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # Create or get collection for this agent
        try:
            self.collection = self.client.get_collection(name=f"memory_{self.agent_id}")
        except:
            self.collection = self.client.create_collection(name=f"memory_{self.agent_id}")
    
    def save(self, data: Any, key: Optional[str] = None, text: str = None, metadata: Dict[str, Any] = None) -> str:
        """Save data to vector store with embeddings"""
        if key is None:
            # Generate key based on timestamp and content hash
            timestamp = int(time.time())
            content_hash = hashlib.md5(str(data).encode()).hexdigest()[:8]
            key = f"{timestamp}_{content_hash}"
        
        # If no text is provided, try to convert data to string
        if text is None:
            if isinstance(data, (str, int, float, bool)):
                text = str(data)
            elif isinstance(data, (dict, list)):
                text = json.dumps(data)
            else:
                text = str(data)
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        # Add timestamp
        metadata["timestamp"] = datetime.now().isoformat()
        
        # Store serialized data in metadata
        metadata["_serialized_data"] = pickle.dumps(data).hex()
        
        # Add to collection
        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[key]
        )
        
        return key
    
    def load(self, key: str) -> Any:
        """Load data from vector store by key"""
        result = self.collection.get(ids=[key])
        
        if result and result["metadatas"]:
            serialized_data = result["metadatas"][0].get("_serialized_data")
            if serialized_data:
                return pickle.loads(bytes.fromhex(serialized_data))
        
        return None
    
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar memories"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        if not results["metadatas"]:
            return []
        
        output = []
        for i, doc_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i].copy()
            
            # Extract and deserialize data
            serialized_data = metadata.pop("_serialized_data", None)
            data = None
            if serialized_data:
                data = pickle.loads(bytes.fromhex(serialized_data))
            
            output.append({
                "key": doc_id,
                "text": results["documents"][0][i],
                "metadata": metadata,
                "data": data,
                "distance": results.get("distances", [[]])[0][i] if "distances" in results else None
            })
        
        return output
    
    def list_keys(self) -> List[str]:
        """List all memory keys"""
        return self.collection.get()["ids"]
    
    def delete(self, key: str) -> bool:
        """Delete a memory entry by key"""
        self.collection.delete(ids=[key])
        return True
    
    def clear(self) -> bool:
        """Clear all memory for this agent"""
        self.collection.delete()
        return True

# Factory function to get the appropriate memory store
def get_memory_store(agent_id: str, store_type: str = "file", memory_dir: str = DEFAULT_MEMORY_DIR, **kwargs) -> MemoryStore:
    """
    Factory function to create the appropriate memory store
    
    Args:
        agent_id: Unique identifier for the agent
        store_type: Type of memory store (file, sqlite, duckdb, vector, mem0, supabase)
        memory_dir: Directory to store memory files
        **kwargs: Additional arguments to pass to the memory store constructor
        
    Returns:
        MemoryStore instance
    """
    if store_type == "file":
        return FileMemoryStore(agent_id, memory_dir)
    elif store_type == "sqlite":
        return SQLiteMemoryStore(agent_id, memory_dir)
    elif store_type == "duckdb":
        return DuckDBMemoryStore(agent_id, memory_dir)
    elif store_type == "vector":
        return VectorMemoryStore(agent_id, memory_dir)
    elif store_type == "supabase":
        if SUPABASE_AVAILABLE:
            return get_supabase_memory_store(agent_id, **kwargs)
        else:
            raise ImportError("Supabase memory store not available. Install it with: from utils import supabase_memory")
    else:
        raise ValueError(f"Unknown memory store type: {store_type}")

# Example usage
if __name__ == "__main__":
    # Example of using the memory store
    memory = get_memory_store("test_agent", "file")
    
    # Save some data
    key = memory.save({"test": "data"})
    print(f"Saved data with key: {key}")
    
    # Load the data
    data = memory.load(key)
    print(f"Loaded data: {data}")
    
    # List all keys
    keys = memory.list_keys()
    print(f"All keys: {keys}")
    
    # Delete the data
    memory.delete(key)
    print(f"Deleted key: {key}") 