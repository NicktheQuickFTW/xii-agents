# /// script
# dependencies = [
#   "anthropic>=0.47.1",
#   "rich>=13.7.0",
#   "pydantic>=2.0.0",
#   "polars>=1.22.0",
# ]
# ///

"""
    Example Usage:
        uv run sfa_polars_csv_agent_anthropic_v3.py -i "data/analytics.csv" -p "What is the average age of the users?"
"""

import io
import os
import sys
import json
import argparse
import tempfile
import subprocess
import time
from typing import List, Optional, Dict, Any
from rich.console import Console
from rich.panel import Panel
import anthropic
from anthropic import Anthropic
import polars as pl
from pydantic import BaseModel, Field, ValidationError
from core.base_agent import BaseAgent

# Initialize rich console
console = Console()


class PolarsCsvAgent(BaseAgent):
    """Polars CSV agent using Anthropic for data analysis."""
    
    def __init__(self, agent_id: str = "polars_csv_anthropic", memory_type: str = "mem0"):
        """Initialize the Polars CSV Anthropic agent.
        
        Args:
            agent_id: Unique identifier for the agent
            memory_type: Type of memory to use
        """
        super().__init__(agent_id, memory_type)
        self.data_format = "csv"
        self.data_path = os.getenv("DATA_PATH", os.path.join(os.getcwd(), "data"))
        
        # Set up Anthropic client
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-20250219")
        self.csv_path = None
        
    def load_data(self, file_path: str) -> pl.DataFrame:
        """Load CSV data into a Polars DataFrame.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            pl.DataFrame: Loaded data
        """
        try:
            self.csv_path = file_path
            df = pl.scan_csv(file_path).collect()
            return df
        except Exception as e:
            console.log(f"[red]Error loading CSV: {str(e)}[/red]")
            raise e
    
    def save_data(self, data: pl.DataFrame, file_path: str) -> bool:
        """Save Polars DataFrame to a CSV file.
        
        Args:
            data: DataFrame to save
            file_path: Path to save the CSV
            
        Returns:
            bool: True if successful
        """
        try:
            data.write_csv(file_path)
            return True
        except Exception as e:
            console.log(f"[red]Error saving CSV: {str(e)}[/red]")
            return False
    
    def transform_data(self, data: pl.DataFrame, transformation: Dict[str, Any]) -> pl.DataFrame:
        """Apply transformation to Polars DataFrame.
        
        Args:
            data: DataFrame to transform
            transformation: Transformation to apply
            
        Returns:
            pl.DataFrame: Transformed data
        """
        if not transformation.get("code"):
            raise ValueError("Transformation must include 'code' parameter")
        
        # Execute the transformation code
        locals_dict = {"df": data, "pl": pl}
        try:
            exec(transformation["code"], globals(), locals_dict)
            return locals_dict.get("result", data)
        except Exception as e:
            console.log(f"[red]Error transforming data: {str(e)}[/red]")
            raise e
    
    def analyze_data(self, data: pl.DataFrame, analysis_type: str) -> Dict[str, Any]:
        """Analyze Polars DataFrame.
        
        Args:
            data: DataFrame to analyze
            analysis_type: Type of analysis to perform
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        if analysis_type == "summary":
            return {
                "shape": (data.height, data.width),
                "columns": data.columns,
                "dtypes": {col: str(data.schema[col]) for col in data.columns},
                "null_counts": {col: data[col].null_count() for col in data.columns},
                "sample": data.head(5).to_dict(as_series=False)
            }
        elif analysis_type == "statistics":
            numeric_cols = [col for col, dtype in data.schema.items() 
                          if pl.dtype_to_py_type(dtype) in (int, float)]
            return {
                "numeric_columns": numeric_cols,
                "min": {col: data[col].min() for col in numeric_cols},
                "max": {col: data[col].max() for col in numeric_cols},
                "mean": {col: data[col].mean() for col in numeric_cols},
                "median": {col: data[col].median() for col in numeric_cols},
                "std": {col: data[col].std() for col in numeric_cols}
            }
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
    
    def filter_data(self, data: pl.DataFrame, filter_criteria: Dict[str, Any]) -> pl.DataFrame:
        """Filter Polars DataFrame based on criteria.
        
        Args:
            data: DataFrame to filter
            filter_criteria: Filtering criteria
            
        Returns:
            pl.DataFrame: Filtered data
        """
        if not filter_criteria.get("code"):
            raise ValueError("Filter criteria must include 'code' parameter")
        
        # Execute the filter code
        locals_dict = {"df": data, "pl": pl}
        try:
            exec(filter_criteria["code"], globals(), locals_dict)
            return locals_dict.get("result", data)
        except Exception as e:
            console.log(f"[red]Error filtering data: {str(e)}[/red]")
            raise e
    
    def list_columns(self, reasoning: str) -> List[str]:
        """Returns a list of columns in the CSV file.
    
        The agent uses this to discover available columns and make informed decisions.
        This is typically the first tool called to understand the data structure.
    
        Args:
            reasoning: Explanation of why we're listing columns relative to user request
    
        Returns:
            List of column names as strings
    
        Example:
            columns = list_columns("Need to find age-related columns", "data.csv")
            # Returns: ['user_id', 'age', 'name', ...]
        """
        try:
            df = pl.scan_csv(self.csv_path).collect()
            columns = df.columns
            console.log(f"[blue]List Columns Tool[/blue] - Reasoning: {reasoning}")
            console.log(f"[dim]Columns: {columns}[/dim]")
            return columns
        except Exception as e:
            console.log(f"[red]Error listing columns: {str(e)}[/red]")
            return []
    
    
    def sample_csv(self, reasoning: str, row_count: int) -> str:
        """Returns a sample of rows from the CSV file.
    
        The agent uses this to understand actual data content and patterns.
        This helps validate data types and identify any potential data quality issues.
    
        Args:
            reasoning: Explanation of why we're sampling this data
            row_count: Number of rows to sample (aim for 3-5 rows)
    
        Returns:
            String containing sample rows in readable format
    
        Example:
            sample = sample_csv("Check age values and formats", "data.csv", 3)
            # Returns formatted string with 3 rows of data
        """
        try:
            df = pl.scan_csv(self.csv_path).limit(row_count).collect()
            # Convert to string representation
            output = df.select(pl.all()).write_csv(None)
            console.log(
                f"[blue]Sample CSV Tool[/blue] - Rows: {row_count} - Reasoning: {reasoning}"
            )
            console.log(f"[dim]Sample:\n{output}[/dim]")
            return output
        except Exception as e:
            console.log(f"[red]Error sampling CSV: {str(e)}[/red]")
            return ""
    
    
    def run_test_polars_code(self, reasoning: str, polars_python_code: str) -> str:
        """Executes test Polars Python code and returns results.
    
        The agent uses this to validate code before finalizing it.
        Results are only shown to the agent, not the user.
        The code should use Polars' lazy evaluation (LazyFrame) for better performance.
    
        Args:
            reasoning: Explanation of why we're running this test code
            polars_python_code: The Polars Python code to test. Should use pl.scan_csv() for lazy evaluation.
    
        Returns:
            Code execution results as a string
        """
        try:
            # Create a unique filename based on timestamp
            timestamp = int(time.time())
            filename = f"test_polars_{timestamp}.py"
    
            # Prepare code with proper CSV path
            modified_code = polars_python_code.replace("{{CSV_PATH}}", self.csv_path)
    
            # Write code to a real file
            with open(filename, "w") as f:
                f.write(modified_code)
    
            # Execute the code
            result = subprocess.run(
                ["uv", "run", "--with", "polars", filename],
                text=True,
                capture_output=True,
            )
            output = result.stdout + result.stderr
    
            # Clean up the file
            os.remove(filename)
    
            console.log(f"[blue]Test Code Tool[/blue] - Reasoning: {reasoning}")
            console.log(f"[dim]Code:\n{modified_code}[/dim]")
            return output
        except Exception as e:
            console.log(f"[red]Error running test code: {str(e)}[/red]")
            return str(e)
    
    
    def run_final_polars_code(self, reasoning: str, polars_python_code: str, 
                            output_file: Optional[str] = None) -> str:
        """Executes the final Polars code and returns results to user.
    
        This is the last tool call the agent should make after validating the code.
        The code should be fully tested and ready for production use.
        Results will be displayed to the user and optionally saved to a file.
    
        Args:
            reasoning: Final explanation of how this code satisfies user request
            polars_python_code: The validated Polars Python code to run
            output_file: Optional path to save results to
    
        Returns:
            Code execution results as a string
        """
        try:
            # Create a unique filename based on timestamp
            timestamp = int(time.time())
            filename = f"polars_code_{timestamp}.py"
    
            # Prepare code with proper CSV path
            modified_code = polars_python_code.replace("{{CSV_PATH}}", self.csv_path)
    
            # Write code to a real file
            with open(filename, "w") as f:
                f.write(modified_code)
    
            # Execute the code
            result = subprocess.run(
                ["uv", "run", "--with", "polars", filename],
                text=True,
                capture_output=True,
            )
            output = result.stdout + result.stderr
    
            # Clean up the file
            os.remove(filename)
    
            console.log(Panel(f"[green]Final Code Tool[/green]\nReasoning: {reasoning}\n"))
            console.log(f"[dim]Code:\n{modified_code}[/dim]")
            
            # Store the result in memory
            self.remember({
                "reasoning": reasoning,
                "code": polars_python_code,
                "result": output,
                "timestamp": time.time()
            })
            
            return output
        except Exception as e:
            console.log(f"[red]Error running final code: {str(e)}[/red]")
            return str(e)
    
    def run(self, prompt: str, csv_path: str, output_file: Optional[str] = None, 
          max_loops: int = 10) -> Dict[str, Any]:
        """Main execution method for the Polars CSV Anthropic agent.
        
        Args:
            prompt: The user's natural language request
            csv_path: Path to the CSV file
            output_file: Optional path to save results to
            max_loops: Maximum number of model invocations
            
        Returns:
            Dict[str, Any]: Results of the agent's execution
        """
        start_time = time.time()
        self.csv_path = csv_path
        
        console.log(f"[bold green]Running Polars CSV Agent[/bold green]")
        console.log(f"Prompt: {prompt}")
        console.log(f"CSV Path: {csv_path}")
        
        # Define tools
        tools = [
            {
                "name": "list_columns",
                "description": "Returns list of available columns in the CSV file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "Why we need to list columns relative to user request",
                        }
                    },
                    "required": ["reasoning"],
                },
            },
            {
                "name": "sample_csv",
                "description": "Returns sample rows from the CSV file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "Why we need to sample this data",
                        },
                        "row_count": {
                            "type": "integer",
                            "description": "Number of rows to sample (aim for 3-5 rows)",
                        },
                    },
                    "required": ["reasoning", "row_count"],
                },
            },
            {
                "name": "run_test_polars_code",
                "description": "Tests Polars Python code and returns results (only visible to agent)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "Why we're running this test code",
                        },
                        "polars_python_code": {
                            "type": "string",
                            "description": "The Polars Python code to test. Use {{CSV_PATH}} as placeholder for the CSV file path.",
                        },
                    },
                    "required": ["reasoning", "polars_python_code"],
                },
            },
            {
                "name": "run_final_polars_code",
                "description": "Runs the final validated Polars code and shows results to user",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "Final explanation of how code satisfies user request",
                        },
                        "polars_python_code": {
                            "type": "string",
                            "description": "The validated Polars Python code to run. Use {{CSV_PATH}} as placeholder for the CSV file path.",
                        },
                        "output_file": {
                            "type": "string",
                            "description": "Optional path to save results to",
                        },
                    },
                    "required": ["reasoning", "polars_python_code"],
                },
            },
        ]
        
        # Prepare messages
        messages = [{
            "role": "user",
            "content": f"""I need help analyzing a CSV file with Polars.

CSV file path: {csv_path}

Request: {prompt}

Use the tools to help me analyze this data. First, explore the data to understand its structure, then write and test Polars code to answer my question.

Please use pl.scan_csv() for lazy evaluation when possible for better performance.

In code use {{{{CSV_PATH}}}} as a placeholder for the actual CSV file path."""
        }]
        
        final_output = None
        
        # Main agent loop
        for i in range(max_loops):
            console.log(f"[bold]Loop {i+1}/{max_loops}[/bold]")
            
            # Call Claude with tools
            response = self.anthropic.messages.create(
                model=self.model,
                max_tokens=4000,
                temperature=0,
                messages=messages,
                tools=tools
            )
            
            # Extract the tool call if present
            content = response.content[0]
            messages.append({"role": "assistant", "content": [content]})
            
            if content.type == "tool_use":
                tool_name = content.name
                tool_input = content.input
                
                # Call the appropriate method
                if tool_name == "list_columns":
                    result = self.list_columns(tool_input["reasoning"])
                elif tool_name == "sample_csv":
                    result = self.sample_csv(
                        tool_input["reasoning"],
                        tool_input["row_count"]
                    )
                elif tool_name == "run_test_polars_code":
                    result = self.run_test_polars_code(
                        tool_input["reasoning"],
                        tool_input["polars_python_code"]
                    )
                elif tool_name == "run_final_polars_code":
                    output_path = tool_input.get("output_file", output_file)
                    result = self.run_final_polars_code(
                        tool_input["reasoning"],
                        tool_input["polars_python_code"],
                        output_path
                    )
                    final_output = {
                        "success": True,
                        "result": result,
                        "code": tool_input["polars_python_code"],
                        "reasoning": tool_input["reasoning"]
                    }
                    break
                
                # Add the result to messages
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_call_id": content.id,
                            "content": json.dumps(result, default=str)
                        }
                    ]
                })
            else:
                # If it's a text response and not a tool call, we're done
                final_output = {
                    "success": True, 
                    "result": content.text,
                    "final_response": True
                }
                break
        
        elapsed_time = time.time() - start_time
        console.log(f"[dim]Completed in {elapsed_time:.2f} seconds[/dim]")
        
        if final_output:
            return final_output
        else:
            return {
                "success": False,
                "error": "Maximum number of loops reached without producing final output"
            }


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Polars CSV Agent using Anthropic API")
    parser.add_argument("-i", "--input", required=True, help="Path to CSV file")
    parser.add_argument("-p", "--prompt", required=True, help="The user's request")
    parser.add_argument("-o", "--output", help="Path to save output to")
    parser.add_argument("-l", "--loops", type=int, default=10, help="Maximum number of loops")
    
    args = parser.parse_args()
    
    # Initialize and run the agent
    agent = PolarsCsvAgent()
    result = agent.run(args.prompt, args.input, args.output, args.loops)
    
    if result["success"]:
        if "final_response" in result:
            console.print(Panel(result["result"], title="Final Response", border_style="green"))
        else:
            console.print(Panel(result["result"], title="Analysis Result", border_style="green"))
    else:
        console.print(Panel(result["error"], title="Error", border_style="red"))


if __name__ == "__main__":
    main()
