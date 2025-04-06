#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "anthropic>=0.45.2",
#   "rich>=13.7.0",
# ]
# ///

"""
/// Example Usage

# Run DuckDB agent with default compute loops (3)
uv run sfa_duckdb_anthropic_v2.py -d ./data/analytics.db -p "Show me all users with score above 80"

# Run with custom compute loops
uv run sfa_duckdb_anthropic_v2.py -d ./data/analytics.db -p "Show me all users with score above 80" -c 5

///
"""

import os
import sys
import json
import argparse
import subprocess
from typing import List, Dict, Any
from rich.console import Console
from rich.panel import Panel
from anthropic import Anthropic
from core.base_agent import BaseAgent

# Initialize rich console
console = Console()


AGENT_PROMPT = """<purpose>
    You are a world-class expert at crafting precise DuckDB SQL queries.
    Your goal is to generate accurate queries that exactly match the user's data needs.
</purpose>

<instructions>
    <instruction>Use the provided tools to explore the database and construct the perfect query.</instruction>
    <instruction>Start by listing tables to understand what's available.</instruction>
    <instruction>Describe tables to understand their schema and columns.</instruction>
    <instruction>Sample tables to see actual data patterns.</instruction>
    <instruction>Test queries before finalizing them.</instruction>
    <instruction>Only call run_final_sql_query when you're confident the query is perfect.</instruction>
    <instruction>Be thorough but efficient with tool usage.</instruction>
    <instruction>If you find your run_test_sql_query tool call returns an error or won't satisfy the user request, try to fix the query or try a different query.</instruction>
    <instruction>Think step by step about what information you need.</instruction>
    <instruction>Be sure to specify every parameter for each tool call.</instruction>
    <instruction>Every tool call should have a reasoning parameter which gives you a place to explain why you are calling the tool.</instruction>
</instructions>

<tools>
    <tool>
        <n>list_tables</n>
        <description>Returns list of available tables in database</description>
        <parameters>
            <parameter>
                <n>reasoning</n>
                <type>string</type>
                <description>Why we need to list tables relative to user request</description>
                <required>true</required>
            </parameter>
        </parameters>
    </tool>
    
    <tool>
        <n>describe_table</n>
        <description>Returns schema info for specified table</description>
        <parameters>
            <parameter>
                <n>reasoning</n>
                <type>string</type>
                <description>Why we need to describe this table</description>
                <required>true</required>
            </parameter>
            <parameter>
                <n>table_name</n>
                <type>string</type>
                <description>Name of table to describe</description>
                <required>true</required>
            </parameter>
        </parameters>
    </tool>
    
    <tool>
        <n>sample_table</n>
        <description>Returns sample rows from specified table, always specify row_sample_size</description>
        <parameters>
            <parameter>
                <n>reasoning</n>
                <type>string</type>
                <description>Why we need to sample this table</description>
                <required>true</required>
            </parameter>
            <parameter>
                <n>table_name</n>
                <type>string</type>
                <description>Name of table to sample</description>
                <required>true</required>
            </parameter>
            <parameter>
                <n>row_sample_size</n>
                <type>integer</type>
                <description>Number of rows to sample aim for 3-5 rows</description>
                <required>true</required>
            </parameter>
        </parameters>
    </tool>
    
    <tool>
        <n>run_test_sql_query</n>
        <description>Tests a SQL query and returns results (only visible to agent)</description>
        <parameters>
            <parameter>
                <n>reasoning</n>
                <type>string</type>
                <description>Why we're testing this specific query</description>
                <required>true</required>
            </parameter>
            <parameter>
                <n>sql_query</n>
                <type>string</type>
                <description>The SQL query to test</description>
                <required>true</required>
            </parameter>
        </parameters>
    </tool>
    
    <tool>
        <n>run_final_sql_query</n>
        <description>Runs the final validated SQL query and shows results to user</description>
        <parameters>
            <parameter>
                <n>reasoning</n>
                <type>string</type>
                <description>Final explanation of how query satisfies user request</description>
                <required>true</required>
            </parameter>
            <parameter>
                <n>sql_query</n>
                <type>string</type>
                <description>The validated SQL query to run</description>
                <required>true</required>
            </parameter>
        </parameters>
    </tool>
</tools>

<user-request>
    {{user_request}}
</user-request>
"""


class DuckDBAnthropic(BaseAgent):
    """DuckDB agent using Anthropic for SQL query generation."""
    
    def __init__(self, agent_id: str = "duckdb_anthropic", memory_type: str = "mem0", db_path: str = ":memory:"):
        """Initialize the DuckDB Anthropic agent.
        
        Args:
            agent_id: Unique identifier for the agent
            memory_type: Type of memory to use
            db_path: Path to the DuckDB database
        """
        super().__init__(agent_id, memory_type)
        # Initialize DuckDB connection
        import duckdb
        self.db = duckdb.connect(database=":memory:")
        
        # If a file path is provided, close the in-memory connection and use the file
        if db_path != ":memory:":
            self.db.close()
            self.db = duckdb.connect(database=db_path)
        
        # Set up Anthropic client
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
        self.max_compute_loops = 3
        self.user_request = ""
        
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a SQL query in DuckDB.
        
        Args:
            query: SQL query to execute
            
        Returns:
            List[Dict[str, Any]]: Query results
        """
        result = self.db.execute(query)
        column_names = [desc[0] for desc in result.description]
        return [dict(zip(column_names, row)) for row in result.fetchall()]
    
    def list_tables(self, reasoning: str) -> List[str]:
        """Returns a list of tables in the database.
    
        The agent uses this to discover available tables and make informed decisions.
    
        Args:
            reasoning: Explanation of why we're listing tables relative to user request
    
        Returns:
            List of table names as strings
        """
        console.log(f"[blue]List Tables Tool[/blue] - Reasoning: {reasoning}")
        return [table[0] for table in self.execute_query("SELECT name FROM sqlite_master WHERE type='table'")]
    
    def describe_table(self, reasoning: str, table_name: str) -> str:
        """Returns schema information about the specified table.
    
        The agent uses this to understand table structure and available columns.
    
        Args:
            reasoning: Explanation of why we're describing this table
            table_name: Name of table to describe
    
        Returns:
            String containing table schema information
        """
        console.log(f"[blue]Describe Table Tool[/blue] - Table: {table_name} - Reasoning: {reasoning}")
        result = self.execute_query(f"PRAGMA table_info({table_name})")
        return "\n".join([f"{col['name']} ({col['type']})" for col in result])
    
    def sample_table(self, reasoning: str, table_name: str, row_sample_size: int) -> str:
        """Returns sample rows from the specified table.
    
        The agent uses this to understand data patterns and make informed decisions.
    
        Args:
            reasoning: Explanation of why we're sampling this table
            table_name: Name of table to sample
            row_sample_size: Number of rows to sample
    
        Returns:
            String containing formatted sample rows
        """
        console.log(f"[blue]Sample Table Tool[/blue] - Table: {table_name} - Rows: {row_sample_size} - Reasoning: {reasoning}")
        result = self.execute_query(f"SELECT * FROM {table_name} LIMIT {row_sample_size}")
        return json.dumps(result, indent=2)
    
    def run_test_sql_query(self, reasoning: str, sql_query: str) -> str:
        """Tests a SQL query and returns results (only visible to agent).
    
        The agent uses this to validate queries before returning them to the user.
    
        Args:
            reasoning: Explanation of why we're testing this query
            sql_query: SQL query to test
    
        Returns:
            String containing query results or error message
        """
        console.log(f"[blue]Test SQL Query Tool[/blue] - Reasoning: {reasoning}")
        console.log(f"[dim]{sql_query}[/dim]")
        try:
            result = self.execute_query(sql_query)
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error: {str(e)}"
    
    def run_final_sql_query(self, reasoning: str, sql_query: str) -> str:
        """Runs the final validated SQL query and shows results to user.
    
        This is the final output shown to the user.
    
        Args:
            reasoning: Explanation of how query satisfies user request
            sql_query: The validated SQL query to run
    
        Returns:
            String containing query results or error message
        """
        console.log(f"[green]Final SQL Query Tool[/green] - Reasoning: {reasoning}")
        console.print(Panel(sql_query, title="SQL Query", border_style="green"))
        try:
            result = self.execute_query(sql_query)
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error: {str(e)}"
    
    def run(self, user_request: str, max_compute_loops: int = 3, db_path: str = None) -> Dict[str, Any]:
        """Main execution method for the DuckDB Anthropic agent.
        
        Args:
            user_request: Natural language request for SQL query
            max_compute_loops: Maximum number of model invocations
            db_path: Path to the DuckDB database (if different from initialized path)
            
        Returns:
            Dict[str, Any]: Results of the agent's execution
        """
        self.user_request = user_request
        self.max_compute_loops = max_compute_loops
        
        # Prepare the prompt with the user request
        prompt = AGENT_PROMPT.replace("{{user_request}}", user_request)
        
        # Initialize messages for the conversation
        messages = [{"role": "user", "content": prompt}]
        
        # Main agent loop
        for i in range(max_compute_loops):
            console.log(f"[bold]Compute loop {i+1}/{max_compute_loops}[/bold]")
            
            # Get response from Anthropic
            response = self.anthropic.messages.create(
                model=self.model,
                messages=messages,
                max_tokens=4000,
                temperature=0
            )
            
            # Extract the assistant's message
            assistant_message = response.content[0].text
            
            # Check if there's a tool call in the response
            if "<tool>" in assistant_message and "</tool>" in assistant_message:
                # Extract tool call
                tool_start = assistant_message.index("<tool>")
                tool_end = assistant_message.index("</tool>") + len("</tool>")
                tool_call = assistant_message[tool_start:tool_end]
                
                # Parse tool name and parameters
                tool_name = tool_call.split("<n>")[1].split("</n>")[0]
                params = {}
                
                # Extract parameters
                param_sections = tool_call.split("<parameter>")[1:]
                for section in param_sections:
                    if "</parameter>" in section:
                        param_name = section.split("<n>")[1].split("</n>")[0]
                        param_value = section.split("</value>")[0].split("<value>")[1]
                        params[param_name] = param_value
                
                # Call the appropriate method
                if tool_name == "list_tables":
                    result = self.list_tables(params["reasoning"])
                elif tool_name == "describe_table":
                    result = self.describe_table(params["reasoning"], params["table_name"])
                elif tool_name == "sample_table":
                    result = self.sample_table(params["reasoning"], params["table_name"], int(params["row_sample_size"]))
                elif tool_name == "run_test_sql_query":
                    result = self.run_test_sql_query(params["reasoning"], params["sql_query"])
                elif tool_name == "run_final_sql_query":
                    result = self.run_final_sql_query(params["reasoning"], params["sql_query"])
                    # Store the result in memory
                    self.remember({
                        "user_request": user_request,
                        "sql_query": params["sql_query"],
                        "reasoning": params["reasoning"],
                        "result": result
                    })
                    return {
                        "success": True,
                        "user_request": user_request,
                        "sql_query": params["sql_query"],
                        "reasoning": params["reasoning"],
                        "result": result
                    }
                
                # Add the result to the messages
                messages.append({"role": "assistant", "content": assistant_message})
                messages.append({
                    "role": "user", 
                    "content": f"<tool-result>\n{result}\n</tool-result>"
                })
            else:
                # If there's no tool call, just add the assistant's message and break
                messages.append({"role": "assistant", "content": assistant_message})
                break
        
        return {
            "success": False,
            "user_request": user_request,
            "error": "Max compute loops reached without final result"
        }


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="DuckDB Agent with Anthropic")
    parser.add_argument("-d", "--database", help="Path to DuckDB database", required=True)
    parser.add_argument("-p", "--prompt", help="Natural language query prompt", required=True)
    parser.add_argument("-c", "--compute-loops", help="Max compute loops", type=int, default=3)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize and run the agent
    agent = DuckDBAnthropic(db_path=args.database)
    result = agent.run(args.prompt, args.compute_loops)
    
    # Display results
    if result["success"]:
        console.print(Panel(result["result"], title="Query Results", border_style="green"))
    else:
        console.print(Panel(result["error"], title="Error", border_style="red"))


if __name__ == "__main__":
    main()
