# /// script
# dependencies = [
#   "openai>=1.63.0",
#   "rich>=13.7.0",
#   "pydantic>=2.0.0",
# ]
# ///


import os
import sys
import json
import argparse
import sqlite3
import subprocess
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.panel import Panel
import openai
from pydantic import BaseModel, Field, ValidationError
from openai import pydantic_function_tool
from core.base_agent import BaseAgent

# Initialize rich console
console = Console()


# Create our list of function tools from our pydantic models
class ListTablesArgs(BaseModel):
    reasoning: str = Field(
        ..., description="Explanation for listing tables relative to the user request"
    )


class DescribeTableArgs(BaseModel):
    reasoning: str = Field(..., description="Reason why the table schema is needed")
    table_name: str = Field(..., description="Name of the table to describe")


class SampleTableArgs(BaseModel):
    reasoning: str = Field(..., description="Explanation for sampling the table")
    table_name: str = Field(..., description="Name of the table to sample")
    row_sample_size: int = Field(
        ..., description="Number of rows to sample (aim for 3-5 rows)"
    )


class RunTestSQLQuery(BaseModel):
    reasoning: str = Field(..., description="Reason for testing this query")
    sql_query: str = Field(..., description="The SQL query to test")


class RunFinalSQLQuery(BaseModel):
    reasoning: str = Field(
        ...,
        description="Final explanation of how this query satisfies the user request",
    )
    sql_query: str = Field(..., description="The validated SQL query to run")


# Create tools list
tools = [
    pydantic_function_tool(ListTablesArgs),
    pydantic_function_tool(DescribeTableArgs),
    pydantic_function_tool(SampleTableArgs),
    pydantic_function_tool(RunTestSQLQuery),
    pydantic_function_tool(RunFinalSQLQuery),
]

AGENT_PROMPT = """<purpose>
    You are a world-class expert at crafting precise SQLite SQL queries.
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


class SQLiteAgent(BaseAgent):
    """SQLite agent using OpenAI for SQL query generation."""
    
    def __init__(self, agent_id: str = "sqlite_openai", memory_type: str = "mem0", db_path: str = ":memory:"):
        """Initialize the SQLite OpenAI agent.
        
        Args:
            agent_id: Unique identifier for the agent
            memory_type: Type of memory to use
            db_path: Path to the SQLite database
        """
        super().__init__(agent_id, memory_type)
        self.data_format = "sqlite"
        self.db_path = db_path
        
        # Set up OpenAI client
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "o3-mini")
        
        # Agent state
        self.max_compute_loops = 10
        self.user_request = ""
        
    def list_tables(self, reasoning: str) -> List[str]:
        """Returns a list of tables in the database.

        The agent uses this to discover available tables and make informed decisions.

        Args:
            reasoning: Explanation of why we're listing tables relative to user request

        Returns:
            List of table names as strings
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            console.log(f"[blue]List Tables Tool[/blue] - Reasoning: {reasoning}")
            
            # Store the result in memory
            self.remember({
                "tool": "list_tables",
                "reasoning": reasoning,
                "result": tables
            })
            
            return tables
        except Exception as e:
            console.log(f"[red]Error listing tables: {str(e)}[/red]")
            return []

    def describe_table(self, reasoning: str, table_name: str) -> str:
        """Returns schema information about the specified table.

        The agent uses this to understand table structure and available columns.

        Args:
            reasoning: Explanation of why we're describing this table
            table_name: Name of table to describe

        Returns:
            String containing table schema information
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info('{table_name}');")
            rows = cursor.fetchall()
            conn.close()
            output = "\n".join([str(row) for row in rows])
            console.log(f"[blue]Describe Table Tool[/blue] - Table: {table_name} - Reasoning: {reasoning}")
            
            # Store the result in memory
            self.remember({
                "tool": "describe_table",
                "reasoning": reasoning,
                "table_name": table_name,
                "result": output
            })
            
            return output
        except Exception as e:
            console.log(f"[red]Error describing table: {str(e)}[/red]")
            return ""

    def sample_table(self, reasoning: str, table_name: str, row_sample_size: int) -> str:
        """Returns a sample of rows from the specified table.

        The agent uses this to understand actual data content and patterns.

        Args:
            reasoning: Explanation of why we're sampling this table
            table_name: Name of table to sample from
            row_sample_size: Number of rows to sample aim for 3-5 rows

        Returns:
            String containing sample rows in readable format
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM {table_name} LIMIT {row_sample_size};")
            rows = cursor.fetchall()
            conn.close()
            output = "\n".join([str(row) for row in rows])
            console.log(
                f"[blue]Sample Table Tool[/blue] - Table: {table_name} - Rows: {row_sample_size} - Reasoning: {reasoning}"
            )
            
            # Store the result in memory
            self.remember({
                "tool": "sample_table",
                "reasoning": reasoning,
                "table_name": table_name,
                "row_sample_size": row_sample_size,
                "result": output
            })
            
            return output
        except Exception as e:
            console.log(f"[red]Error sampling table: {str(e)}[/red]")
            return ""

    def run_test_sql_query(self, reasoning: str, sql_query: str) -> str:
        """Executes a test SQL query and returns results.

        The agent uses this to validate queries before finalizing them.
        Results are only shown to the agent, not the user.

        Args:
            reasoning: Explanation of why we're running this test query
            sql_query: The SQL query to test

        Returns:
            Query results as a string
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            conn.commit()
            conn.close()
            output = "\n".join([str(row) for row in rows])
            console.log(f"[blue]Test Query Tool[/blue] - Reasoning: {reasoning}")
            console.log(f"[dim]Query: {sql_query}[/dim]")
            
            # Store the result in memory
            self.remember({
                "tool": "run_test_sql_query",
                "reasoning": reasoning,
                "sql_query": sql_query,
                "result": output
            })
            
            return output
        except Exception as e:
            console.log(f"[red]Error running test query: {str(e)}[/red]")
            return str(e)

    def run_final_sql_query(self, reasoning: str, sql_query: str) -> str:
        """Executes the final SQL query and returns results to user.

        This is the last tool call the agent should make after validating the query.

        Args:
            reasoning: Final explanation of how this query satisfies user request
            sql_query: The validated SQL query to run

        Returns:
            Query results as a string
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            conn.commit()
            conn.close()
            output = "\n".join([str(row) for row in rows])
            console.log(
                Panel(
                    f"[green]Final Query Tool[/green]\nReasoning: {reasoning}\nQuery: {sql_query}"
                )
            )
            
            # Store the result in memory
            self.remember({
                "tool": "run_final_sql_query",
                "reasoning": reasoning,
                "sql_query": sql_query,
                "result": output
            })
            
            return output
        except Exception as e:
            console.log(f"[red]Error running final query: {str(e)}[/red]")
            return str(e)

    def run(self, user_request: str, max_compute_loops: int = 10, db_path: Optional[str] = None) -> Dict[str, Any]:
        """Main execution method for the SQLite OpenAI agent.
        
        Args:
            user_request: Natural language request for SQL query
            max_compute_loops: Maximum number of model invocations
            db_path: Path to the SQLite database (if different from initialized path)
            
        Returns:
            Dict[str, Any]: Results of the agent's execution
        """
        self.user_request = user_request
        self.max_compute_loops = max_compute_loops
        
        if db_path:
            self.db_path = db_path
        
        # Create a single combined prompt based on the full template
        completed_prompt = AGENT_PROMPT.replace("{{user_request}}", user_request)
        messages = [{"role": "user", "content": completed_prompt}]
        
        compute_iterations = 0
        result = None
        
        # Main agent loop
        while True:
            console.rule(
                f"[yellow]Agent Loop {compute_iterations+1}/{max_compute_loops}[/yellow]"
            )
            compute_iterations += 1
            
            if compute_iterations >= max_compute_loops:
                console.print(
                    "[yellow]Warning: Reached maximum compute loops without final query[/yellow]"
                )
                return {
                    "success": False,
                    "error": f"Maximum compute loops reached: {compute_iterations}/{max_compute_loops}",
                    "iterations": compute_iterations
                }
            
            try:
                # Generate content with tool support
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="required",
                )
                
                if response.choices:
                    assert len(response.choices) == 1
                    message = response.choices[0].message
                    
                    if message.function_call:
                        func_call = message.function_call
                    elif message.tool_calls and len(message.tool_calls) > 0:
                        # If a tool_calls list is present, use the first call and extract its function details.
                        tool_call = message.tool_calls[0]
                        func_call = tool_call.function
                    else:
                        func_call = None
                    
                    if func_call:
                        func_name = func_call.name
                        func_args_str = func_call.arguments
                        
                        messages.append(
                            {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": tool_call.id,
                                        "type": "function",
                                        "function": func_call,
                                    }
                                ],
                            }
                        )
                        
                        console.print(
                            f"[blue]Function Call:[/blue] {func_name}({func_args_str})"
                        )
                        try:
                            # Validate and parse arguments using the corresponding pydantic model
                            if func_name == "ListTablesArgs":
                                args_parsed = ListTablesArgs.model_validate_json(
                                    func_args_str
                                )
                                result = self.list_tables(reasoning=args_parsed.reasoning)
                            elif func_name == "DescribeTableArgs":
                                args_parsed = DescribeTableArgs.model_validate_json(
                                    func_args_str
                                )
                                result = self.describe_table(
                                    reasoning=args_parsed.reasoning,
                                    table_name=args_parsed.table_name,
                                )
                            elif func_name == "SampleTableArgs":
                                args_parsed = SampleTableArgs.model_validate_json(
                                    func_args_str
                                )
                                result = self.sample_table(
                                    reasoning=args_parsed.reasoning,
                                    table_name=args_parsed.table_name,
                                    row_sample_size=args_parsed.row_sample_size,
                                )
                            elif func_name == "RunTestSQLQuery":
                                args_parsed = RunTestSQLQuery.model_validate_json(
                                    func_args_str
                                )
                                result = self.run_test_sql_query(
                                    reasoning=args_parsed.reasoning,
                                    sql_query=args_parsed.sql_query,
                                )
                            elif func_name == "RunFinalSQLQuery":
                                args_parsed = RunFinalSQLQuery.model_validate_json(
                                    func_args_str
                                )
                                result = self.run_final_sql_query(
                                    reasoning=args_parsed.reasoning,
                                    sql_query=args_parsed.sql_query,
                                )
                                console.print("\n[green]Final Results:[/green]")
                                console.print(result)
                                return {
                                    "success": True,
                                    "result": result,
                                    "iterations": compute_iterations,
                                    "sql_query": args_parsed.sql_query,
                                    "reasoning": args_parsed.reasoning
                                }
                            else:
                                raise Exception(f"Unknown tool call: {func_name}")
                            
                            console.print(
                                f"[blue]Function Call Result:[/blue] {func_name}(...) ->\n{result}"
                            )
                            
                            # Append the function call result into our messages as a tool response
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": json.dumps({"result": str(result)}),
                                }
                            )
                            
                        except Exception as e:
                            error_msg = f"Argument validation failed for {func_name}: {e}"
                            console.print(f"[red]{error_msg}[/red]")
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": json.dumps({"error": error_msg}),
                                }
                            )
                            continue
                    else:
                        raise Exception(
                            "No function call in this response - should never happen"
                        )
                    
            except Exception as e:
                console.print(f"[red]Error in agent loop: {str(e)}[/red]")
                return {
                    "success": False,
                    "error": str(e),
                    "iterations": compute_iterations
                }


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="SQLite Agent using OpenAI API")
    parser.add_argument(
        "-d", "--db", required=True, help="Path to SQLite database file"
    )
    parser.add_argument("-p", "--prompt", required=True, help="The user's request")
    parser.add_argument(
        "-c",
        "--compute",
        type=int,
        default=10,
        help="Maximum number of agent loops (default: 10)",
    )
    args = parser.parse_args()

    # Configure the API key
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        console.print(
            "[red]Error: OPENAI_API_KEY environment variable is not set[/red]"
        )
        console.print(
            "Please get your API key from https://platform.openai.com/api-keys"
        )
        console.print("Then set it with: export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)

    # Initialize the agent
    agent = SQLiteAgent(db_path=args.db)
    
    # Run the agent
    try:
        result = agent.run(args.prompt, args.compute)
        
        if not result["success"]:
            console.print(f"[red]Error: {result.get('error', 'Unknown error')}[/red]")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
