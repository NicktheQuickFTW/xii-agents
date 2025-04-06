from typing import Any, Dict, List, Optional
from core.base_agent import BaseAgent
import os
import json
from pathlib import Path
from dotenv import load_dotenv

class BaseDataProcAgent(BaseAgent):
    """Base class for data processing agents."""
    
    def __init__(self, agent_id: str, data_format: str, memory_type: str = "mem0"):
        """Initialize the data processing agent.
        
        Args:
            agent_id: Unique identifier for the agent
            data_format: Type of data format (csv, json, etc.)
            memory_type: Type of memory to use
        """
        super().__init__(agent_id, memory_type)
        self.data_format = data_format
        self.data_path = os.getenv("DATA_PATH", os.path.join(os.getcwd(), "data"))
        
    def load_data(self, file_path: str) -> Any:
        """Load data from a file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Any: Loaded data
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def save_data(self, data: Any, file_path: str) -> bool:
        """Save data to a file.
        
        Args:
            data: Data to save
            file_path: Path to save the data
            
        Returns:
            bool: True if successful
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def transform_data(self, data: Any, transformation: Dict[str, Any]) -> Any:
        """Apply transformation to data.
        
        Args:
            data: Data to transform
            transformation: Transformation to apply
            
        Returns:
            Any: Transformed data
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def analyze_data(self, data: Any, analysis_type: str) -> Dict[str, Any]:
        """Analyze data.
        
        Args:
            data: Data to analyze
            analysis_type: Type of analysis to perform
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def filter_data(self, data: Any, filter_criteria: Dict[str, Any]) -> Any:
        """Filter data based on criteria.
        
        Args:
            data: Data to filter
            filter_criteria: Filtering criteria
            
        Returns:
            Any: Filtered data
        """
        raise NotImplementedError("Subclasses must implement this method") 