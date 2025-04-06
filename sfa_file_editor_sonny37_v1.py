#!/usr/bin/env python3

# /// script
# dependencies = [
#   "anthropic>=0.49.0",
#   "rich>=13.7.0",
# ]
# ///

"""
/// Example Usage

# View a file
uv run sfa_file_editor_sonny37_v1.py --prompt "Show me the content of README.md"

# Use token-efficient tools (reduces token usage by ~14% on average)
uv run sfa_file_editor_sonny37_v1.py --prompt "Read the first 20 lines of content from README.md and summarize into a new README_SUMMARY.md" --efficiency

# Edit a file
uv run sfa_file_editor_sonny37_v1.py --prompt "Fix the syntax error in sfa_poc.py"

# Create a new file
uv run sfa_file_editor_sonny37_v1.py --prompt "Create a new file called hello.py with a function that prints Hello World"

# Add docstrings to functions
uv run sfa_file_editor_sonny37_v1.py --prompt "Add proper docstrings to all functions in sfa_poc.py"

# Insert code at specific location
uv run sfa_file_editor_sonny37_v1.py --prompt "Insert error handling code before the API call in sfa_duckdb_openai_v2.py"

# Modify multiple files
uv run sfa_file_editor_sonny37_v1.py --prompt "Update all print statements in agent_workspace directory to use f-strings"

# Refactor code
uv run sfa_file_editor_sonny37_v1.py --prompt "Refactor the factorial function in agent_workspace/test.py to use iteration instead of recursion"

# Create new test files
uv run sfa_file_editor_sonny37_v1.py --prompt "Create unit tests for the functions in sfa_file_editor_sonny37_v1.py and save them in agent_workspace/test_file_editor.py"

# Run with higher thinking tokens
uv run sfa_file_editor_sonny37_v1.py --prompt "Refactor README.md to make it more concise" --thinking 5000

# Increase max loops for complex tasks
uv run sfa_file_editor_sonny37_v1.py --prompt "Create a Python class that implements a binary search tree with insert, delete, and search methods" --max-loops 20

# Combine multiple flags

uv run sfa_file_editor_sonny37_v1.py --prompt "Create a Flask API with 3 endpoints inside of agent_workspace/api_server.py" --thinking 6000 --max-loops 25

uv run sfa_file_editor_sonny37_v1.py --prompt "Create a Flask API with 3 endpoints inside of agent_workspace/api_server.py" --efficiency --thinking 6000 --max-loops 25

///
"""

import os
import sys
import argparse
import time
import json
import traceback
from typing import List, Dict, Any, Optional, Tuple, Union
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table
from rich.style import Style
from rich.align import Align
from anthropic import Anthropic
from specialized.editor.base_editor_agent import BaseEditorAgent

# Initialize rich console
console = Console()

# Define constants
MODEL = "claude-3-7-sonnet-20250219"
DEFAULT_THINKING_TOKENS = 3000


def display_token_usage(input_tokens: int, output_tokens: int) -> None:
    """
    Display token usage information in a rich formatted table

    Args:
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens used
    """
    total_tokens = input_tokens + output_tokens
    token_ratio = output_tokens / input_tokens if input_tokens > 0 else 0

    # Create a table for token usage
    table = Table(title="Token Usage Statistics", expand=True)

    # Add columns with proper styling
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Count", style="magenta", justify="right")
    table.add_column("Percentage", justify="right")

    # Add rows with data
    table.add_row(
        "Input Tokens", f"{input_tokens:,}", f"{input_tokens/total_tokens:.1%}"
    )
    table.add_row(
        "Output Tokens", f"{output_tokens:,}", f"{output_tokens/total_tokens:.1%}"
    )
    table.add_row("Total Tokens", f"{total_tokens:,}", "100.0%")
    table.add_row("Output/Input Ratio", f"{token_ratio:.2f}", "")

    console.print()
    console.print(table)


def normalize_path(path: str) -> str:
    """
    Normalize file paths to handle various formats (absolute, relative, Windows paths, etc.)

    Args:
        path: The path to normalize

    Returns:
        The normalized path
    """
    if not path:
        return path

    # Handle Windows backslash paths if provided
    path = path.replace("\\", os.sep)

    is_windows_path = False
    if os.name == "nt" and len(path) > 1 and path[1] == ":":
        is_windows_path = True

    # Handle /repo/ paths from Claude (tool use convention)
    if path.startswith("/repo/"):
        path = os.path.join(os.getcwd(), path[6:])
        return path

    if path.startswith("/"):
        # Handle case when Claude provides paths with leading slash
        if path == "/" or path == "/.":
            # Special case for root directory
            path = os.getcwd()
        else:
            # Replace leading slash with current working directory
            path = os.path.join(os.getcwd(), path[1:])
    elif path.startswith("./"):
        # Handle relative paths starting with ./
        path = os.path.join(os.getcwd(), path[2:])
    elif not os.path.isabs(path) and not is_windows_path:
        # For non-absolute paths that aren't Windows paths either
        path = os.path.join(os.getcwd(), path)

    return path


class FileEditorAgent(BaseEditorAgent):
    """File editor agent using Anthropic Claude for text editing."""
    
    def __init__(self, agent_id: str = "file_editor_anthropic", memory_type: str = "mem0"):
        """Initialize the File Editor Anthropic agent.
        
        Args:
            agent_id: Unique identifier for the agent
            memory_type: Type of memory to use
        """
        super().__init__(agent_id, memory_type)
        
        # Set up Anthropic client
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = os.getenv("ANTHROPIC_MODEL", MODEL)
        self.max_thinking_tokens = DEFAULT_THINKING_TOKENS
        self.max_loops = 10
        self.use_token_efficiency = False
        
        # Dict to store original file contents for undo functionality
        self.file_backups = {}
        
    def view_file(self, path: str, view_range=None) -> Dict[str, Any]:
        """
        View the contents of a file.
    
        Args:
            path: The path to the file to view
            view_range: Optional start and end lines to view [start, end]
    
        Returns:
            Dictionary with content or error message
        """
        try:
            # Normalize the path
            path = normalize_path(path)
    
            if not os.path.exists(path):
                error_msg = f"File {path} does not exist"
                console.log(f"[view_file] Error: {error_msg}")
                return {"error": error_msg}
    
            content = self.read_file(path)
            lines = content.splitlines(True)
    
            if view_range:
                start, end = view_range
                # Convert to 0-indexed for Python
                start = max(0, start - 1)
                if end == -1:
                    end = len(lines)
                else:
                    end = min(len(lines), end)
                lines = lines[start:end]
                content = "".join(lines)
    
            # Display the file content (only for console, not returned to Claude)
            file_extension = os.path.splitext(path)[1][1:]  # Get extension without the dot
            syntax = Syntax(content, file_extension or "text", line_numbers=True)
            console.print(Panel(syntax, title=f"File: {path}"))
    
            return {"result": content}
        except Exception as e:
            error_msg = f"Error viewing file: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            console.log(f"[view_file] Error: {str(e)}")
            console.log(traceback.format_exc())
            return {"error": error_msg}
    
    def str_replace(self, path: str, old_str: str, new_str: str) -> Dict[str, Any]:
        """
        Replace a string in a file with a new string.
    
        Args:
            path: The path to the file
            old_str: The string to replace
            new_str: The new string
    
        Returns:
            Dictionary with result message or error
        """
        try:
            # Normalize the path
            path = normalize_path(path)
    
            if not os.path.exists(path):
                error_msg = f"File {path} does not exist"
                console.log(f"[str_replace] Error: {error_msg}")
                return {"error": error_msg}
    
            # Backup the file for undo functionality if not already backed up
            if path not in self.file_backups:
                self.file_backups[path] = self.read_file(path)
    
            # Read the current content
            content = self.read_file(path)
    
            # Check if old_str exists in the file
            if old_str not in content:
                error_msg = f"String not found in {path}"
                console.log(f"[str_replace] Error: {error_msg}")
                return {"error": error_msg}
    
            # Replace the string
            new_content = content.replace(old_str, new_str)
    
            # Write back to the file
            self.write_file(path, new_content)
    
            # For display purposes only
            import difflib
            diff = difflib.unified_diff(
                content.splitlines(), new_content.splitlines(), lineterm=""
            )
            console.print(Panel("\n".join(diff), title=f"Changes in {path}"))
    
            return {"result": f"Successfully replaced string in {path}"}
        except Exception as e:
            error_msg = f"Error replacing string: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            console.log(f"[str_replace] Error: {str(e)}")
            console.log(traceback.format_exc())
            return {"error": error_msg}
    
    def create_file(self, path: str, file_text: str) -> Dict[str, Any]:
        """
        Create a new file with the specified content.
    
        Args:
            path: The path for the new file
            file_text: The content to write to the file
    
        Returns:
            Dictionary with result message or error
        """
        try:
            # Normalize the path
            path = normalize_path(path)
    
            # Check if file already exists
            if os.path.exists(path):
                warning_msg = f"File {path} already exists. Overwriting."
                console.log(f"[create_file] Warning: {warning_msg}")
    
            # Create directory if it doesn't exist
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
    
            # Write the content to the file
            self.write_file(path, file_text)
    
            # Display a confirmation message
            console.print(
                Panel(
                    f"Created file {path} with {len(file_text.splitlines())} lines",
                    title="File Created",
                    style="green",
                )
            )
    
            # Show a preview of the file
            file_extension = os.path.splitext(path)[1][1:]
            syntax = Syntax(
                file_text[:1000] + ("..." if len(file_text) > 1000 else ""),
                file_extension or "text",
                line_numbers=True,
            )
            console.print(Panel(syntax, title=f"Preview of {path}"))
    
            return {"result": f"Successfully created file {path}"}
        except Exception as e:
            error_msg = f"Error creating file: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            console.log(f"[create_file] Error: {str(e)}")
            console.log(traceback.format_exc())
            return {"error": error_msg}
    
    def insert_text(self, path: str, insert_line: int, new_str: str) -> Dict[str, Any]:
        """
        Insert text at a specific line in a file.
    
        Args:
            path: The path to the file
            insert_line: The line number after which to insert text (0 for beginning of file)
            new_str: The text to insert
    
        Returns:
            Dictionary with result message or error
        """
        try:
            # Normalize the path
            path = normalize_path(path)
    
            if not os.path.exists(path):
                error_msg = f"File {path} does not exist"
                console.log(f"[insert_text] Error: {error_msg}")
                return {"error": error_msg}
    
            # Backup the file for undo functionality if not already backed up
            if path not in self.file_backups:
                self.file_backups[path] = self.read_file(path)
    
            # Read the current content
            content = self.read_file(path)
            lines = content.splitlines(True)  # Keep line endings
    
            # Validate insert_line
            if insert_line < 0 or insert_line > len(lines):
                error_msg = f"Invalid insert line {insert_line}. File has {len(lines)} lines."
                console.log(f"[insert_text] Error: {error_msg}")
                return {"error": error_msg}
    
            # Ensure new_str ends with a newline if inserted in the middle
            if insert_line < len(lines) and not new_str.endswith("\n"):
                new_str += "\n"
    
            # Insert the text
            before = lines[:insert_line]
            after = lines[insert_line:]
            new_content = "".join(before) + new_str + "".join(after)
    
            # Write back to the file
            self.write_file(path, new_content)
    
            # For display purposes only
            import difflib
            diff = difflib.unified_diff(
                content.splitlines(), new_content.splitlines(), lineterm=""
            )
            console.print(Panel("\n".join(diff), title=f"Changes in {path}"))
    
            return {
                "result": f"Successfully inserted text at line {insert_line} in {path}"
            }
        except Exception as e:
            error_msg = f"Error inserting text: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            console.log(f"[insert_text] Error: {str(e)}")
            console.log(traceback.format_exc())
            return {"error": error_msg}
    
    def undo_edit(self, path: str) -> Dict[str, Any]:
        """
        Undo the last edit to a file by restoring from backup.
    
        Args:
            path: The path to the file
    
        Returns:
            Dictionary with result message or error
        """
        try:
            # Normalize the path
            path = normalize_path(path)
    
            if not os.path.exists(path):
                error_msg = f"File {path} does not exist"
                console.log(f"[undo_edit] Error: {error_msg}")
                return {"error": error_msg}
    
            if path not in self.file_backups:
                error_msg = f"No edits to undo for {path}"
                console.log(f"[undo_edit] Error: {error_msg}")
                return {"error": error_msg}
    
            # Restore the backup
            backup_content = self.file_backups[path]
            self.write_file(path, backup_content)
    
            # Remove the backup
            del self.file_backups[path]
    
            console.print(
                Panel(f"Reverted changes to {path}", title="Undo Complete", style="green")
            )
    
            return {"result": f"Successfully reverted changes to {path}"}
        except Exception as e:
            error_msg = f"Error undoing edit: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            console.log(f"[undo_edit] Error: {str(e)}")
            console.log(traceback.format_exc())
            return {"error": error_msg}
    
    def handle_tool_use(self, tool_use: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle the tool use request from Claude.
    
        Args:
            tool_use: The tool use request
    
        Returns:
            Dictionary with the tool result
        """
        try:
            command = tool_use.get("command")
            console.log(f"[bold cyan]Tool command:[/bold cyan] {command}")
    
            if command == "view":
                path = tool_use.get("path", "")
                view_range = tool_use.get("view_range")
                return self.view_file(path, view_range)
            elif command == "str_replace":
                path = tool_use.get("path", "")
                old_str = tool_use.get("old_str", "")
                new_str = tool_use.get("new_str", "")
                return self.str_replace(path, old_str, new_str)
            elif command == "create":
                path = tool_use.get("path", "")
                file_text = tool_use.get("file_text", "")
                return self.create_file(path, file_text)
            elif command == "insert":
                path = tool_use.get("path", "")
                insert_line = tool_use.get("insert_line", 0)
                new_str = tool_use.get("new_str", "")
                return self.insert_text(path, insert_line, new_str)
            elif command == "undo_edit":
                path = tool_use.get("path", "")
                return self.undo_edit(path)
            else:
                error_msg = f"Unknown command: {command}"
                console.log(f"[handle_tool_use] Error: {error_msg}")
                return {"error": error_msg}
        except Exception as e:
            error_msg = f"Error handling tool use: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            console.log(f"[handle_tool_use] Error: {str(e)}")
            console.log(traceback.format_exc())
            return {"error": error_msg}
    
    def run(self, prompt: str, max_thinking_tokens: int = DEFAULT_THINKING_TOKENS, 
            max_loops: int = 10, use_token_efficiency: bool = False) -> Dict[str, Any]:
        """Main execution method for the File Editor Anthropic agent.
        
        Args:
            prompt: The user's natural language request
            max_thinking_tokens: Maximum tokens to allocate for thinking
            max_loops: Maximum number of model invocations
            use_token_efficiency: Whether to use token-efficient tools
            
        Returns:
            Dict[str, Any]: Results of the agent's execution
        """
        self.max_thinking_tokens = max_thinking_tokens
        self.max_loops = max_loops
        self.use_token_efficiency = use_token_efficiency
        
        console.log(f"[bold green]Running File Editor Agent[/bold green]")
        console.log(f"Prompt: {prompt}")
        console.log(f"Max thinking tokens: {max_thinking_tokens}")
        console.log(f"Max loops: {max_loops}")
        console.log(f"Use token efficiency: {use_token_efficiency}")
        
        start_time = time.time()
        
        # Set up tool definition (either efficient or standard)
        tool_type = "text_editor_20250124"  # Text editor tool for Claude 3.7 Sonnet
        
        # Initialize messages
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"You are a file editing assistant. Help me with the following task:\n\n{prompt}"
                    }
                ]
            }
        ]
        
        total_input_tokens = 0
        total_output_tokens = 0
        
        # Main agent loop
        for i in range(max_loops):
            console.log(f"[bold]Loop {i+1}/{max_loops}[/bold]")
            
            # Call Claude with the text editor tool
            response = self.anthropic.messages.create(
                model=self.model,
                max_tokens=max_thinking_tokens,
                messages=messages,
                tools=[{"type": tool_type}],
                tool_choice={"type": "any"},
            )
            
            # Update token counts
            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens
            
            # Process the response
            message = response.content[0]
            
            if message.type == "text":
                # If Claude doesn't use a tool, just return the response
                result = message.text
                console.print(Panel(Markdown(result), title="Final Response", style="green"))
                
                # Store the result in memory
                self.remember({
                    "prompt": prompt,
                    "response": result,
                    "timestamp": time.time()
                })
                
                elapsed_time = time.time() - start_time
                console.log(f"Completed in {elapsed_time:.2f} seconds")
                display_token_usage(total_input_tokens, total_output_tokens)
                
                return {
                    "success": True,
                    "prompt": prompt,
                    "response": result,
                    "token_usage": {
                        "input_tokens": total_input_tokens,
                        "output_tokens": total_output_tokens,
                        "total_tokens": total_input_tokens + total_output_tokens
                    },
                    "elapsed_time": elapsed_time
                }
            
            elif message.type == "tool_use":
                # Extract the tool use
                tool_use = json.loads(message.input)
                
                # Handle the tool use
                tool_result = self.handle_tool_use(tool_use)
                
                # Add the result to the messages
                if "error" in tool_result:
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message.id,
                                "content": {
                                    "error": tool_result["error"]
                                }
                            }
                        ]
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message.id,
                                "content": {
                                    "result": tool_result["result"]
                                }
                            }
                        ]
                    })
            else:
                # Unexpected message type
                console.log(f"[yellow]Unexpected message type: {message.type}[/yellow]")
        
        # If we reach here, we've hit the maximum number of loops
        elapsed_time = time.time() - start_time
        console.log(f"Reached maximum number of loops ({max_loops}) in {elapsed_time:.2f} seconds")
        display_token_usage(total_input_tokens, total_output_tokens)
        
        return {
            "success": False,
            "prompt": prompt,
            "error": f"Maximum number of loops ({max_loops}) reached without completion",
            "token_usage": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens
            },
            "elapsed_time": elapsed_time
        }


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="File Editor Agent with Anthropic Claude")
    parser.add_argument("--prompt", required=True, help="The task to perform")
    parser.add_argument("--thinking", type=int, default=DEFAULT_THINKING_TOKENS, help="Maximum thinking tokens")
    parser.add_argument("--max-loops", type=int, default=10, help="Maximum number of agent loops")
    parser.add_argument("--efficiency", action="store_true", help="Use token-efficient tools")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize the agent
    agent = FileEditorAgent()
    
    # Run the agent
    result = agent.run(args.prompt, args.thinking, args.max_loops, args.efficiency)
    
    # The result is already displayed in the agent's run method
    if not result["success"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
