from typing import Any, Dict, List, Optional
from core.base_agent import BaseAgent
import os
from dotenv import load_dotenv

class BaseDatabaseAgent(BaseAgent):
    """Base class for database agents with common database operations."""
    
    def __init__(self, agent_id: str, db_type: str, memory_type: str = "mem0"):
        """Initialize the database agent.
        
        Args:
            agent_id: Unique identifier for the agent
            db_type: Type of database (duckdb, sqlite, etc.)
            memory_type: Type of memory to use
        """
        super().__init__(agent_id, memory_type)
        self.db_type = db_type
        self.db = self._initialize_database()
        
    def _initialize_database(self):
        """Initialize the appropriate database connection."""
        if self.db_type == "duckdb":
            import duckdb
            return duckdb.connect(database=":memory:")
        elif self.db_type == "sqlite":
            import sqlite3
            return sqlite3.connect(":memory:")
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
    
    def execute_query(self, query: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """Execute a database query.
        
        Args:
            query: SQL query to execute
            params: Optional query parameters
            
        Returns:
            List[Dict[str, Any]]: Query results
        """
        cursor = self.db.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
            
        if query.strip().upper().startswith("SELECT"):
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        else:
            self.db.commit()
            return []
    
    def create_table(self, table_name: str, schema: Dict[str, str]) -> bool:
        """Create a new table.
        
        Args:
            table_name: Name of the table
            schema: Dictionary of column names and types
            
        Returns:
            bool: True if successful
        """
        columns = ", ".join([f"{col} {type_}" for col, type_ in schema.items()])
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})"
        self.execute_query(query)
        return True
    
    def insert_data(self, table_name: str, data: List[Dict[str, Any]]) -> bool:
        """Insert data into a table.
        
        Args:
            table_name: Name of the table
            data: List of dictionaries containing data
            
        Returns:
            bool: True if successful
        """
        if not data:
            return False
            
        columns = list(data[0].keys())
        placeholders = ", ".join(["?"] * len(columns))
        query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
        
        for row in data:
            self.execute_query(query, list(row.values()))
            
        return True
    
    def close(self):
        """Close the database connection."""
        self.db.close() 