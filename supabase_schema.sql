-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create agent_memories table
CREATE TABLE IF NOT EXISTS agent_memories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id TEXT NOT NULL,
    data JSONB NOT NULL,
    key TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_agent_memories_agent_id ON agent_memories(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_memories_key ON agent_memories(key);

-- Create function for semantic search
CREATE OR REPLACE FUNCTION search_memories(
    query TEXT,
    agent_id TEXT,
    limit_count INTEGER DEFAULT 10
) RETURNS TABLE (
    id UUID,
    agent_id TEXT,
    data JSONB,
    key TEXT,
    created_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        m.id,
        m.agent_id,
        m.data,
        m.key,
        m.created_at,
        m.updated_at,
        similarity(m.data::TEXT, query) as similarity
    FROM agent_memories m
    WHERE m.agent_id = search_memories.agent_id
    ORDER BY similarity DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql; 