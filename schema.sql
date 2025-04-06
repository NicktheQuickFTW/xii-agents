-- Schema file for Supabase memory store setup
-- This file contains SQL to create the required tables and indexes in Supabase

-- Agent memories table
CREATE TABLE agent_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    key TEXT NOT NULL,
    data JSONB NOT NULL,
    tags TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(agent_id, key)
);

-- Create indexes
CREATE INDEX idx_agent_memories_agent_id ON agent_memories(agent_id);
CREATE INDEX idx_agent_memories_key ON agent_memories(key);
CREATE INDEX idx_agent_memories_tags ON agent_memories USING GIN(tags);
CREATE INDEX idx_agent_memories_metadata ON agent_memories USING GIN(metadata);

-- Agent models table for machine learning models
CREATE TABLE agent_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_type TEXT NOT NULL,
    model_data BYTEA NOT NULL,  -- Serialized model
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(agent_id, model_name)
);

-- Create indexes
CREATE INDEX idx_agent_models_agent_id ON agent_models(agent_id);
CREATE INDEX idx_agent_models_model_name ON agent_models(model_name);
CREATE INDEX idx_agent_models_model_type ON agent_models(model_type);

-- Agent feedback table for collecting feedback for ML models
CREATE TABLE agent_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    prediction JSONB NOT NULL,
    actual JSONB NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes
CREATE INDEX idx_agent_feedback_agent_id ON agent_feedback(agent_id);
CREATE INDEX idx_agent_feedback_model_name ON agent_feedback(model_name);
CREATE INDEX idx_agent_feedback_created_at ON agent_feedback(created_at);

-- Agent conversation history
CREATE TABLE agent_conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    user_input TEXT NOT NULL,
    agent_response TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes
CREATE INDEX idx_agent_conversations_agent_id ON agent_conversations(agent_id);
CREATE INDEX idx_agent_conversations_created_at ON agent_conversations(created_at);

-- Agent runs table for tracking agent executions
CREATE TABLE agent_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    status TEXT NOT NULL, -- 'success', 'error', 'running'
    result JSONB,
    error TEXT,
    runtime_seconds NUMERIC,
    metadata JSONB DEFAULT '{}',
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes
CREATE INDEX idx_agent_runs_agent_id ON agent_runs(agent_id);
CREATE INDEX idx_agent_runs_status ON agent_runs(status);
CREATE INDEX idx_agent_runs_started_at ON agent_runs(started_at);

-- Notification function to update updated_at on record changes
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Add triggers for updated_at columns
CREATE TRIGGER update_agent_memories_updated_at
BEFORE UPDATE ON agent_memories
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agent_models_updated_at 
BEFORE UPDATE ON agent_models
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Row Level Security (RLS) policies
-- Enable RLS
ALTER TABLE agent_memories ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_models ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_feedback ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_runs ENABLE ROW LEVEL SECURITY;

-- Create policies (adjust as needed for your auth setup)
-- For agent_memories
CREATE POLICY "Allow agent access to their own memories" ON agent_memories
  FOR ALL USING (auth.uid()::text = agent_id);

-- For agent_models
CREATE POLICY "Allow agent access to their own models" ON agent_models
  FOR ALL USING (auth.uid()::text = agent_id);

-- For agent_feedback
CREATE POLICY "Allow agent access to their own feedback" ON agent_feedback
  FOR ALL USING (auth.uid()::text = agent_id);

-- For agent_conversations
CREATE POLICY "Allow agent access to their own conversations" ON agent_conversations
  FOR ALL USING (auth.uid()::text = agent_id);

-- For agent_runs
CREATE POLICY "Allow agent access to their own runs" ON agent_runs
  FOR ALL USING (auth.uid()::text = agent_id);

## Context Priming
> Read README.md, ai_docs/*, and run git ls-files to understand this codebase. 