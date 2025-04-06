#!/usr/bin/env python3
"""
Autonomous Agent for Single-File Agents
Purpose: Provides a base class for self-running agents with memory and learning
Version: 1.0.0

Usage:
  Import and extend this class to create autonomous agents
  
Requirements:
  - Python 3.8+
  - memory_store.py, ml_utils.py, mem0_adapter.py (optional)
"""

import os
import sys
import json
import time
import logging
import threading
import traceback
import argparse
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs/agents.log"), mode='a')
    ]
)

# Create logs directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs"), exist_ok=True)

# Try importing memory store modules
MEMORY_STORE_AVAILABLE = False
try:
    from memory_store import get_memory_store, MemoryStore
    MEMORY_STORE_AVAILABLE = True
except ImportError:
    pass

# Try importing Mem0 adapter
MEM0_AVAILABLE = False
try:
    from mem0_adapter import Mem0AgentAdapter, Mem0MemoryStore
    MEM0_AVAILABLE = True
except ImportError:
    pass

# Try importing ML utilities
ML_UTILS_AVAILABLE = False
try:
    from ml_utils import LearningAgent, SimpleModel
    ML_UTILS_AVAILABLE = True
except ImportError:
    pass

class AutonomousAgent:
    """Base class for autonomous agents that can run independently"""
    
    def __init__(
        self, 
        agent_id: str, 
        memory_type: str = "sqlite",  # "sqlite", "file", "duckdb", "vector", "mem0"
        enable_learning: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the autonomous agent
        
        Args:
            agent_id: Unique identifier for the agent
            memory_type: Type of memory storage to use
            enable_learning: Whether to enable machine learning capabilities
            config: Additional configuration options
        """
        self.agent_id = agent_id
        self.config = config or {}
        self.logger = logging.getLogger(f"agent.{agent_id}")
        
        # Set up memory storage
        self.memory = self._setup_memory(memory_type)
        
        # Set up learning capabilities
        self.learning = None
        if enable_learning and ML_UTILS_AVAILABLE:
            self.learning = LearningAgent(agent_id, self.memory)
        
        # Conversation history
        self.conversation_history = []
        
        # State management
        self.state = self._load_state() or {
            "created_at": datetime.now().isoformat(),
            "last_active": None,
            "run_count": 0,
            "last_run_status": None,
            "settings": {}
        }
        
        # Running flag
        self.is_running = False
        self.run_thread = None
    
    def _setup_memory(self, memory_type: str) -> Any:
        """Set up memory storage based on type"""
        if memory_type == "mem0":
            if MEM0_AVAILABLE:
                self.logger.info(f"Using Mem0 for agent {self.agent_id} memory")
                return Mem0AgentAdapter(self.agent_id)
            else:
                self.logger.warning("Mem0 not available, falling back to SQLite")
                memory_type = "sqlite"
        
        if MEMORY_STORE_AVAILABLE:
            self.logger.info(f"Using {memory_type} for agent {self.agent_id} memory")
            return get_memory_store(self.agent_id, memory_type)
        else:
            self.logger.warning("Memory store not available, agent will not persist memory")
            return None
    
    def _save_state(self) -> None:
        """Save agent state to memory"""
        # Update last active time
        self.state["last_active"] = datetime.now().isoformat()
        
        if hasattr(self.memory, "save_agent_state"):
            # For Mem0 adapter
            self.memory.save_agent_state(self.state)
        elif hasattr(self.memory, "save"):
            # For regular memory store
            self.memory.save(self.state, "agent_state")
    
    def _load_state(self) -> Optional[Dict[str, Any]]:
        """Load agent state from memory"""
        if hasattr(self.memory, "load_agent_state"):
            # For Mem0 adapter
            return self.memory.load_agent_state()
        elif hasattr(self.memory, "load"):
            # For regular memory store
            return self.memory.load("agent_state")
        return None
    
    def remember(self, data: Any, key: Optional[str] = None, tags: Optional[List[str]] = None) -> Optional[str]:
        """Store data in memory"""
        if self.memory:
            if hasattr(self.memory, "remember"):
                # For Mem0 adapter
                return self.memory.remember(data, key, tags)
            elif hasattr(self.memory, "save"):
                # For regular memory store
                return self.memory.save(data, key)
        return None
    
    def recall(self, key: str) -> Any:
        """Retrieve data from memory"""
        if self.memory:
            if hasattr(self.memory, "recall"):
                # For Mem0 adapter
                return self.memory.recall(key)
            elif hasattr(self.memory, "load"):
                # For regular memory store
                return self.memory.load(key)
        return None
    
    def search_memory(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search memory (semantic search if available)"""
        if self.memory:
            if hasattr(self.memory, "reflect"):
                # For Mem0 adapter
                return self.memory.reflect(query, n_results)
            elif hasattr(self.memory, "search"):
                # For vector memory store
                return self.memory.search(query, n_results)
            elif hasattr(self.memory, "search_by_tags"):
                # For other memory stores
                return self.memory.search_by_tags([query])
        return []
    
    def add_conversation(self, user_input: str, agent_response: str) -> None:
        """Add conversation to history and memory"""
        # Add to in-memory history
        conversation = {
            "user": user_input,
            "agent": agent_response,
            "timestamp": datetime.now().isoformat()
        }
        self.conversation_history.append(conversation)
        
        # Store in persistent memory
        if self.memory:
            if hasattr(self.memory, "add_conversation"):
                # For Mem0 adapter
                self.memory.add_conversation(user_input, agent_response)
            elif hasattr(self.memory, "save"):
                # For regular memory store
                timestamp = int(time.time())
                self.memory.save(conversation, f"conversation_{timestamp}")
    
    def learn(self, model_name: str, features: List[List[float]], targets: List[Any]) -> Optional[float]:
        """Train a machine learning model with data"""
        if self.learning:
            # Get or create the model
            model = self.learning.get_model(model_name)
            if not model:
                # Default to random forest classifier/regressor based on target type
                if all(isinstance(t, (int, bool)) for t in targets):
                    model_type = "random_forest_classifier"
                else:
                    model_type = "random_forest_regressor"
                
                self.learning.create_model(model_name, model_type)
            
            # Train the model
            return self.learning.train_model(model_name, features, targets)
        return None
    
    def predict(self, model_name: str, features: Union[List[float], List[List[float]]]) -> Any:
        """Make a prediction using a trained model"""
        if self.learning:
            return self.learning.make_prediction(model_name, features)
        return None
    
    def record_feedback(self, prediction: Any, actual_result: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record feedback for a prediction to improve future predictions"""
        if self.learning:
            self.learning.record_feedback(prediction, actual_result, metadata)
    
    def improve_from_feedback(self, model_name: str, feature_extractor: Callable[[Dict[str, Any]], List[float]]) -> Optional[float]:
        """Learn from collected feedback data"""
        if self.learning:
            return self.learning.learn_from_feedback(model_name, feature_extractor)
        return None
    
    def save_models(self, directory: Optional[str] = None) -> None:
        """Save learned models to disk"""
        if self.learning:
            if directory is None:
                directory = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), 
                    f"models/{self.agent_id}"
                )
            
            os.makedirs(directory, exist_ok=True)
            self.learning.save_models(directory)
    
    def load_models(self, directory: Optional[str] = None) -> None:
        """Load learned models from disk"""
        if self.learning:
            if directory is None:
                directory = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), 
                    f"models/{self.agent_id}"
                )
            
            if os.path.exists(directory):
                self.learning.load_models(directory)
    
    def run(self, **kwargs) -> Any:
        """
        Main method to run the agent's task (override in subclasses)
        
        Args:
            **kwargs: Additional arguments
            
        Returns:
            Result of the agent's execution
        """
        raise NotImplementedError("Subclasses must implement the run method")
    
    def start(self, **kwargs) -> None:
        """Start the agent in a separate thread"""
        if self.is_running:
            self.logger.warning(f"Agent {self.agent_id} is already running")
            return
        
        # Update state
        self.state["run_count"] += 1
        self.state["last_run_start"] = datetime.now().isoformat()
        self._save_state()
        
        # Start agent in a new thread
        self.is_running = True
        self.run_thread = threading.Thread(target=self._run_thread, kwargs=kwargs)
        self.run_thread.daemon = True
        self.run_thread.start()
        
        self.logger.info(f"Agent {self.agent_id} started")
    
    def _run_thread(self, **kwargs) -> None:
        """Thread method to run the agent"""
        try:
            self.logger.info(f"Agent {self.agent_id} is running")
            
            # Run the agent
            result = self.run(**kwargs)
            
            # Update state
            self.state["last_run_status"] = "success"
            self.state["last_run_result"] = str(result)
            self.state["last_run_end"] = datetime.now().isoformat()
            
            self.logger.info(f"Agent {self.agent_id} completed successfully")
            
        except Exception as e:
            # Handle exception
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            
            self.logger.error(f"Agent {self.agent_id} error: {error_msg}")
            self.logger.error(stack_trace)
            
            # Update state
            self.state["last_run_status"] = "error"
            self.state["last_error"] = {
                "message": error_msg,
                "traceback": stack_trace,
                "timestamp": datetime.now().isoformat()
            }
            
        finally:
            # Save state and cleanup
            self._save_state()
            self.is_running = False
    
    def stop(self) -> None:
        """
        Stop the agent (override in subclasses if needed)
        
        Note: Since threads can't be forcefully terminated in Python,
        subclasses should implement a mechanism to check self.is_running
        and exit gracefully when it becomes False.
        """
        if not self.is_running:
            return
        
        self.logger.info(f"Stopping agent {self.agent_id}")
        self.is_running = False
        
        # Wait for thread to finish (with timeout)
        if self.run_thread and self.run_thread.is_alive():
            self.run_thread.join(timeout=5)
            if self.run_thread.is_alive():
                self.logger.warning(f"Agent {self.agent_id} did not stop gracefully")
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the agent"""
        return {
            "agent_id": self.agent_id,
            "is_running": self.is_running,
            "state": self.state,
            "memory_available": self.memory is not None,
            "learning_available": self.learning is not None,
            "conversation_count": len(self.conversation_history)
        }

class SimpleAutonomousAgent(AutonomousAgent):
    """Simple implementation of an autonomous agent for demonstration"""
    
    def __init__(
        self, 
        agent_id: str = "simple_agent",
        task: Optional[Callable[..., Any]] = None,
        memory_type: str = "sqlite",
        enable_learning: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a simple autonomous agent
        
        Args:
            agent_id: Unique identifier for the agent
            task: Function to run for each execution
            memory_type: Type of memory storage to use
            enable_learning: Whether to enable machine learning
            config: Additional configuration options
        """
        super().__init__(agent_id, memory_type, enable_learning, config)
        self.task = task
    
    def run(self, **kwargs) -> Any:
        """Run the agent's task if provided"""
        if self.task:
            result = self.task(self, **kwargs)
            # Store result in memory
            self.remember(
                {
                    "result": result,
                    "kwargs": kwargs,
                    "timestamp": datetime.now().isoformat()
                },
                f"task_result_{int(time.time())}",
                ["task_result"]
            )
            return result
        else:
            # Default behavior if no task provided
            self.logger.info(f"Agent {self.agent_id} running default behavior")
            
            # Record this execution
            self.remember(
                {
                    "message": "Agent executed with no specific task",
                    "timestamp": datetime.now().isoformat()
                },
                f"execution_{int(time.time())}",
                ["execution"]
            )
            
            return "No task defined"

# Example task function for SimpleAutonomousAgent
def example_task(agent, **kwargs):
    """Example task that demonstrates agent capabilities"""
    agent.logger.info("Running example task")
    
    # Remember some data
    agent.remember({"example": "data", "timestamp": datetime.now().isoformat()})
    
    # Simulate a conversation
    agent.add_conversation("Hello agent!", "Hello! I'm running autonomously!")
    
    # If learning is available, train a simple model
    if agent.learning:
        # Temperature prediction model (features: [hour_of_day, is_sunny])
        X = [
            [8, 1],   # 8am, sunny - expect warmer
            [12, 1],  # noon, sunny - expect hot
            [18, 1],  # 6pm, sunny - expect cooling
            [22, 0],  # 10pm, not sunny - expect cool
            [3, 0],   # 3am, not sunny - expect cold
        ]
        y = [22, 28, 25, 18, 15]  # temperatures in Celsius
        
        agent.learn("temperature_model", X, y)
        
        # Make and record a prediction
        prediction = agent.predict("temperature_model", [15, 1])  # 3pm, sunny
        agent.logger.info(f"Predicted temperature: {prediction}°C")
        
        # Record feedback (actual temperature was different)
        agent.record_feedback(prediction, 27, {"hour": 15, "is_sunny": True})
    
    # Return a result
    return {"status": "completed", "task_name": kwargs.get("task_name", "example_task")}

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Agent Example")
    parser.add_argument("--agent-id", type=str, default="simple_agent", help="Agent ID")
    parser.add_argument("--memory", type=str, default="sqlite", choices=["file", "sqlite", "duckdb", "vector", "mem0"], help="Memory type")
    parser.add_argument("--no-learning", action="store_true", help="Disable learning capabilities")
    parser.add_argument("--task", type=str, help="Task to run (example or none)")
    
    args = parser.parse_args()
    
    # Determine task function
    task_fn = None
    if args.task == "example":
        task_fn = example_task
    elif args.task != "none":
        print(f"Unknown task: {args.task}. Using default.")
    
    # Create agent
    agent = SimpleAutonomousAgent(
        agent_id=args.agent_id,
        task=task_fn,
        memory_type=args.memory,
        enable_learning=not args.no_learning
    )
    
    print(f"Created agent {agent.agent_id}")
    print(f"Memory type: {args.memory}")
    print(f"Learning enabled: {not args.no_learning}")
    
    # Run synchronously for command line
    print(f"Running agent {agent.agent_id}...")
    result = agent.run()
    print(f"Result: {result}")
    
    # Print status
    status = agent.get_status()
    print(f"Final status: {json.dumps(status, indent=2)}") 