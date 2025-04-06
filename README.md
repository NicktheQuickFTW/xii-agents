# XII Agents

A collection of autonomous AI agents for the XII-OS ecosystem, designed for easy deployment and maintenance.

## Context Priming
> Read README.md, ai_docs/*, and run git ls-files to understand this codebase.

# XII Agents Architecture

This repository contains the Agent architecture for XII-OS, which provides a standardized approach for creating autonomous agents with memory, learning capabilities, and self-execution.

## Core Components

### Core Agents
The foundation of our agent system, providing essential functionality:
```
core/
├── base_agent.py         # Base agent class with core functionality
├── memory_agent.py       # Agent with memory capabilities
├── learning_agent.py     # Agent with learning capabilities
└── autonomous_agent.py   # Agent with scheduling and autonomy
```

### Specialized Agents
Domain-specific agents built on core functionality:
```
specialized/
├── flextime/            # Sports scheduling agents
├── transfer-portal/     # NCAA transfer portal agents
└── [other domains]/     # Additional specialized agents
```

### Example Agents
Reference implementations and templates:
```
examples/
├── basic_agent.py       # Simple agent template
├── memory_agent.py      # Agent with memory example
├── learning_agent.py    # Agent with learning example
└── autonomous_agent.py  # Agent with scheduling example
```

### Utility Modules
Shared functionality across all agents:
```
utils/
├── memory_store.py      # Memory storage implementations
├── ml_utils.py          # Machine learning utilities
├── agent_utils.py       # Agent helper functions
└── autonomous_scheduler.py # Agent scheduling
```

## Memory and Learning Architecture

### Mem0 AI Integration (Primary Memory System)
Our primary memory system using Mem0 AI for advanced capabilities:

1. **Setup**
   ```bash
   git clone https://github.com/mem0ai/mem0.git single-file-agents/memory-store/mem0
   ```

2. **Features**
   - Enhanced semantic search
   - Entity relationship graphs
   - Long-term memory persistence
   - Contextual memory retrieval
   - Memory versioning

3. **Usage**
   ```python
   from utils.mem0_adapter import Mem0MemoryStore
   
   memory = Mem0MemoryStore("agent_id")
   memory.remember({"data": "important info"})
   results = memory.search("relevant context")
   ```

### Supabase Memory Storage (Secondary/Cloud Storage)
Cloud-based memory storage for distributed agents:

1. **Setup**
   ```bash
   # Set environment variables
   export SUPABASE_URL=your-project-url
   export SUPABASE_KEY=your-api-key
   
   # Initialize database
   psql -U postgres -d your_database -f supabase_schema.sql
   ```

2. **Features**
   - Cloud-based storage
   - PostgreSQL capabilities
   - Row-level security
   - Real-time updates
   - Analytics integration

3. **Usage**
   ```python
   from utils.supabase_memory import SupabaseMemoryStore
   
   memory = SupabaseMemoryStore("agent_id")
   memory.remember({"data": "cloud stored info"})
   results = memory.search("distributed context")
   ```

### Machine Learning Integration
Advanced learning capabilities for agents:

1. **Model Storage**
   ```
   models/
   ├── trained/          # Pre-trained models
   ├── training/         # Models in training
   └── evaluation/       # Model evaluation results
   ```

2. **Learning Workflow**
   ```python
   # Training
   agent.learn("model_name", training_data, labels)
   
   # Prediction
   prediction = agent.predict("model_name", new_data)
   
   # Feedback Loop
   agent.record_feedback(prediction, actual_result)
   agent.improve_from_feedback("model_name", feature_extractor)
   ```

3. **ML Utilities**
   - Feature extraction
   - Model evaluation
   - Hyperparameter tuning
   - Distributed training
   - Model versioning

## Getting Started

### Prerequisites
- Python 3.8+
- Required packages: `pip install -r requirements.txt`
- Optional: Mem0 AI for advanced memory
- Optional: Supabase for cloud storage

### Creating an Agent
1. Choose base class:
   ```python
   from core.base_agent import BaseAgent
   from core.memory_agent import MemoryAgent
   from core.learning_agent import LearningAgent
   from core.autonomous_agent import AutonomousAgent
   ```

2. Select memory system:
   ```python
   # Mem0 (recommended)
   agent = MyAgent(memory_type="mem0")
   
   # Supabase
   agent = MyAgent(memory_type="supabase")
   
   # Local storage
   agent = MyAgent(memory_type="sqlite")  # or "file", "duckdb", "vector"
   ```

3. Add learning capabilities:
   ```python
   class MyAgent(LearningAgent):
       def learn(self, data):
           self.train_model("my_model", data)
           self.evaluate_model("my_model")
   ```

4. Enable autonomy:
   ```python
   # Schedule agent
   scheduler = AutonomousScheduler()
   scheduler.add_agent("my_agent", MyAgent(), schedule="daily 08:00")
   ```

## Advanced Usage

### Agent Communication
```python
# Through shared memory
agent1.remember({"message": "Hello"}, "shared_message")
message = agent2.recall("shared_message")

# Through learning
agent1.learn("shared_model", data)
agent2.use_model("shared_model")
```

### Distributed Learning
```python
# Train across multiple agents
agent1.train_model("distributed_model", data1)
agent2.train_model("distributed_model", data2)
agent1.merge_models("distributed_model", agent2)
```

### Memory Synchronization
```python
# Sync memories between systems
mem0_memory = Mem0MemoryStore("agent_id")
supabase_memory = SupabaseMemoryStore("agent_id")
mem0_memory.sync_to(supabase_memory)
```

### Mem0 Autogen Integration
Advanced agent generation and orchestration using Mem0's Autogen cookbook:

1. **Setup**
   ```bash
   # Navigate to Mem0 cookbooks
   cd memory-store/mem0/cookbooks
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Features**
   - Automated agent generation
   - Multi-agent orchestration
   - Dynamic memory allocation
   - Agent-to-agent communication
   - Task decomposition and delegation

3. **Usage**
   ```python
   # Example from mem0-autogen.ipynb
   from mem0.autogen import AgentGenerator
   
   # Generate specialized agents
   generator = AgentGenerator()
   agent_config = {
       "type": "learning_agent",
       "capabilities": ["memory", "learning", "autonomy"],
       "memory_type": "mem0"
   }
   
   # Create and configure agent
   agent = generator.create_agent(agent_config)
   
   # Orchestrate multiple agents
   orchestrator = generator.create_orchestrator()
   orchestrator.add_agent(agent)
   orchestrator.delegate_task("analyze_data", agent)
   ```

4. **Advanced Orchestration**
   ```python
   # Create agent team
   team = orchestrator.create_team(
       roles=["researcher", "analyst", "executor"],
       memory_shared=True
   )
   
   # Delegate complex tasks
   task = {
       "type": "analysis",
       "data": "complex_dataset",
       "requirements": ["research", "analysis", "execution"]
   }
   team.execute_task(task)
   ```

5. **Memory Integration**
   ```python
   # Shared memory across team
   team_memory = team.get_shared_memory()
   team_memory.remember({"task_results": results})
   
   # Access team memories
   insights = team_memory.search("key_insights")
   ```

This integration allows for:
- Automated agent creation based on requirements
- Dynamic team formation and management
- Shared memory and knowledge across agents
- Complex task decomposition and execution
- Real-time agent communication and coordination

## License

[License information here]
