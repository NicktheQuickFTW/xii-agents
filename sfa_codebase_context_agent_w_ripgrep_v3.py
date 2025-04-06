#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "anthropic>=0.47.1",
#   "rich>=13.7.0",
#   "pydantic>=2.0.0",
# ]
# ///

"""
Usage:
    uv run sfa_codebase_context_agent_w_ripgrep_v3.py \
        --prompt "Let's build a new metaprompt sfa agent using anthropic claude 3.7" \
        --directory "." \
        --globs "*.py" \
        --extensions py md \
        --limit 10 \
        --file-line-limit 1000 \
        --output-file relevant_files.json \
        --compute 15
        
    # Find files related to DuckDB implementations
    uv run sfa_codebase_context_agent_w_ripgrep_v3.py \
        --prompt "Find all files related to DuckDB agent implementations" \
        --file-line-limit 1000 \
        --extensions py
        
    # Find all files related to Anthropic-powered agents
    uv run sfa_codebase_context_agent_w_ripgrep_v3.py \
        --prompt "Identify all agents that use the new Claude 3.7 model"

    # Use ripgrep to search codebase for specific query
    uv run sfa_codebase_context_agent_w_ripgrep_v3.py \
        --prompt "Find all files that use the Anthropic API" \
        --use-ripgrep
    
"""

import os
import sys
import json
import argparse
import subprocess
import time
import fnmatch
import concurrent.futures
from typing import List, Dict, Any, Optional
from rich.console import Console
from anthropic import Anthropic
from rich.table import Table
from rich.panel import Panel
from core.base_agent import BaseAgent

# Initialize rich console
console = Console()

# Constants
THINKING_BUDGET_TOKENS_PER_FILE = 2000
BATCH_SIZE = 10
MAX_RETRIES = 3
RETRY_WAIT = 1


class CodebaseContextAgent(BaseAgent):
    """Codebase context agent using ripgrep for efficient code search."""
    
    def __init__(self, agent_id: str = "codebase_context_agent", memory_type: str = "mem0"):
        """Initialize the CodebaseContextAgent.
        
        Args:
            agent_id: Unique identifier for the agent
            memory_type: Type of memory to use
        """
        super().__init__(agent_id, memory_type)
        self.workspace_root = os.getenv("WORKSPACE_ROOT", ".")
        
        # Set up Anthropic client
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
        
        # Agent state
        self.relevant_files = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.file_line_limit = 500
        self.max_files = 10
        self.use_ripgrep = False
        
    def find_file(self, file_path: str) -> Optional[str]:
        """Find a file by name or partial path.
        
        Args:
            file_path: Name or partial path of the file
            
        Returns:
            Optional[str]: Full path if found, None otherwise
        """
        try:
            result = subprocess.run(
                f'find {self.workspace_root} -name "{file_path}" -type f',
                shell=True,
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                paths = result.stdout.strip().split('\n')
                return paths[0] if paths else None
            return None
        except Exception:
            return None
    
    def read_file(self, file_path: str, start_line: int = None, end_line: int = None) -> str:
        """Read contents of a file.
        
        Args:
            file_path: Path to the file
            start_line: Optional start line
            end_line: Optional end line
            
        Returns:
            str: File contents
        """
        full_path = os.path.join(self.workspace_root, file_path)
        
        try:
            with open(full_path, 'r') as f:
                if start_line is None and end_line is None:
                    return f.read()
                    
                lines = f.readlines()
                if start_line is not None:
                    start_idx = max(0, start_line - 1)
                else:
                    start_idx = 0
                    
                if end_line is not None:
                    end_idx = min(len(lines), end_line)
                else:
                    end_idx = len(lines)
                    
                return ''.join(lines[start_idx:end_idx])
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def git_list_files(self, reasoning: str, directory: str = None, 
                     globs: List[str] = None, extensions: List[str] = None) -> List[str]:
        """Returns a list of files in the repository, respecting gitignore.
    
        Args:
            reasoning: Explanation of why we're listing files
            directory: Directory to search in (defaults to workspace root)
            globs: List of glob patterns to filter files (optional)
            extensions: List of file extensions to filter files (optional)
    
        Returns:
            List of file paths as strings
        """
        try:
            directory = directory or self.workspace_root
            globs = globs or []
            extensions = extensions or []
            
            console.log(f"[blue]Git List Files Tool[/blue] - Reasoning: {reasoning}")
            console.log(
                f"[dim]Directory: {directory}, Globs: {globs}, Extensions: {extensions}[/dim]"
            )
    
            # Change to the specified directory
            original_dir = os.getcwd()
            os.chdir(directory)
    
            # Get all files tracked by git
            result = subprocess.run(
                "git ls-files",
                shell=True,
                text=True,
                capture_output=True,
            )
    
            files = result.stdout.strip().split("\n")
    
            # Filter by globs if provided
            if globs:
                filtered_files = []
                for pattern in globs:
                    for file in files:
                        if fnmatch.fnmatch(file, pattern):
                            filtered_files.append(file)
                files = filtered_files
    
            # Filter by extensions if provided
            if extensions:
                files = [
                    file
                    for file in files
                    if any(file.endswith(f".{ext}") for ext in extensions)
                ]
    
            # Change back to the original directory
            os.chdir(original_dir)
    
            # Keep paths relative
            files = files
    
            console.log(f"[dim]Found {len(files)} files[/dim]")
            return files
        except Exception as e:
            console.log(f"[red]Error listing files: {str(e)}[/red]")
            return []
    
    def check_file_paths_line_length(self, reasoning: str, file_paths: List[str], 
                                   file_line_limit: int = None) -> Dict[str, int]:
        """Checks the line length of each file and returns a dictionary of file paths and their line counts.
    
        Args:
            reasoning: Explanation of why we're checking line lengths
            file_paths: List of file paths to check
            file_line_limit: Maximum number of lines per file
    
        Returns:
            Dictionary mapping file paths to their total line counts
        """
        try:
            file_line_limit = file_line_limit or self.file_line_limit
            
            console.log(
                f"[blue]Check File Paths Line Length Tool[/blue] - Reasoning: {reasoning}"
            )
            console.log(
                f"[dim]Checking {len(file_paths)} files with line limit {file_line_limit}[/dim]"
            )
    
            result = {}
            for file_path in file_paths:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        line_count = len(lines)
                        if line_count <= file_line_limit:
                            result[file_path] = line_count
                        else:
                            console.log(
                                f"[yellow]Skipping {file_path}: {line_count} lines exceed limit of {file_line_limit}[/yellow]"
                            )
                except Exception as e:
                    console.log(f"[red]Error reading file {file_path}: {str(e)}[/red]")
    
            console.log(f"[dim]Found {len(result)} files within line limit[/dim]")
            return result
        except Exception as e:
            console.log(f"[red]Error checking file paths: {str(e)}[/red]")
            return {}
    
    def determine_if_file_is_relevant(self, prompt: str, file_path: str) -> Dict[str, Any]:
        """Determines if a single file is relevant to the prompt.
    
        Args:
            prompt: The user prompt
            file_path: Path to the file to check
    
        Returns:
            Dictionary with reasoning and is_relevant flag
        """
        result = {
            "reasoning": "Error: Could not process file",
            "file_path": file_path,
            "is_relevant": False,
        }
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
    
            console.log(f"[dim]Evaluating relevance of {file_path}[/dim]")
    
            # Call Claude to determine relevance
            response = self.anthropic.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Given the following prompt, determine if the file content below is relevant to the task.

PROMPT: {prompt}

FILE PATH: {file_path}

FILE CONTENT:
```
{file_content[:5000]}
```

Is this file relevant to the prompt? Think step-by-step and provide your reasoning. 
Respond in this exact format:
REASONING: <your detailed reasoning>
IS_RELEVANT: <true or false>
""",
                    }
                ],
            )
    
            # Update token usage
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens
    
            output = response.content[0].text.strip()
    
            # Extract reasoning and is_relevant flag
            reasoning_line = next(
                (line for line in output.split("\n") if line.startswith("REASONING:")), ""
            )
            is_relevant_line = next(
                (line for line in output.split("\n") if line.startswith("IS_RELEVANT:")), ""
            )
    
            reasoning = reasoning_line.replace("REASONING:", "").strip()
            is_relevant = (
                "true" in is_relevant_line.replace("IS_RELEVANT:", "").strip().lower()
            )
    
            result = {
                "reasoning": reasoning,
                "file_path": file_path,
                "is_relevant": is_relevant,
            }
    
            return result
        except Exception as e:
            console.log(f"[red]Error determining file relevance for {file_path}: {str(e)}[/red]")
            return result
    
    def determine_if_files_are_relevant(self, reasoning: str, file_paths: List[str], prompt: str) -> Dict[str, Any]:
        """Determines if multiple files are relevant to the prompt.
    
        Args:
            reasoning: Explanation of why we're checking relevance
            file_paths: List of file paths to check
            prompt: The user prompt
    
        Returns:
            Dictionary with reasoning and list of relevant files
        """
        try:
            console.log(
                f"[blue]Determine If Files Are Relevant Tool[/blue] - Reasoning: {reasoning}"
            )
            console.log(f"[dim]Checking {len(file_paths)} files[/dim]")
    
            # Process files in parallel
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_file = {
                    executor.submit(self.determine_if_file_is_relevant, prompt, file_path): file_path
                    for file_path in file_paths
                }
                for future in concurrent.futures.as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        results.append(result)
                        if result["is_relevant"]:
                            console.log(
                                f"[green]Relevant: {file_path}[/green] - {result['reasoning'][:100]}..."
                            )
                        else:
                            console.log(
                                f"[dim]Not relevant: {file_path}[/dim]"
                            )
                    except Exception as e:
                        console.log(
                            f"[red]Error processing {file_path}: {str(e)}[/red]"
                        )
    
            relevant_files = [
                {"file_path": result["file_path"], "reasoning": result["reasoning"]}
                for result in results
                if result["is_relevant"]
            ]
    
            # Update the relevant files list
            self.relevant_files.extend(relevant_files)
    
            console.log(f"[dim]Found {len(relevant_files)} relevant files[/dim]")
            return {
                "reasoning": f"Found {len(relevant_files)} relevant files",
                "relevant_files": relevant_files,
            }
        except Exception as e:
            console.log(f"[red]Error determining file relevance: {str(e)}[/red]")
            return {
                "reasoning": f"Error: {str(e)}",
                "relevant_files": [],
            }
    
    def add_relevant_files(self, reasoning: str, file_paths: List[str]) -> str:
        """Add files directly to the list of relevant files without checking content.
    
        Args:
            reasoning: Explanation for adding these files
            file_paths: List of file paths to add
    
        Returns:
            String with confirmation message
        """
        try:
            console.log(f"[blue]Add Relevant Files Tool[/blue] - Reasoning: {reasoning}")
            console.log(f"[dim]Adding {len(file_paths)} files directly[/dim]")
    
            for file_path in file_paths:
                self.relevant_files.append(
                    {"file_path": file_path, "reasoning": reasoning}
                )
    
            return f"Successfully added {len(file_paths)} files directly to the relevant files list."
        except Exception as e:
            console.log(f"[red]Error adding relevant files: {str(e)}[/red]")
            return f"Error adding relevant files: {str(e)}"
    
    def search_code(self, query: str, file_pattern: Optional[str] = None, 
                  max_results: int = 10) -> List[Dict[str, Any]]:
        """Search codebase for matching query.
        
        Args:
            query: Search query
            file_pattern: Optional file pattern to limit search
            max_results: Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of matching code snippets
        """
        return self.search_with_ripgrep(query, file_pattern, True, max_results)
    
    def search_codebase_with_ripgrep(self, reasoning: str, query: str, base_path: str = None, 
                                  max_files: int = None, extensions: List[str] = None, 
                                  globs: List[str] = None) -> Dict[str, Any]:
        """Searches the codebase using ripgrep.
    
        Args:
            reasoning: Explanation of why we're searching the codebase
            query: The search query
            base_path: The base path to search in
            max_files: Maximum number of files to return
            extensions: List of file extensions to filter by
            globs: List of glob patterns to filter by
    
        Returns:
            Dictionary with search results
        """
        try:
            base_path = base_path or self.workspace_root
            max_files = max_files or self.max_files
            extensions = extensions or []
            globs = globs or []
            
            console.log(f"[blue]Search Codebase with Ripgrep Tool[/blue] - Reasoning: {reasoning}")
            console.log(f"[dim]Query: {query}[/dim]")
    
            # Construct ripgrep command
            cmd = ["rg", "--no-heading", "--line-number", "-H", "--sort", "path"]
    
            # Add extension filters if provided
            if extensions:
                for ext in extensions:
                    cmd.extend(["-g", f"*.{ext}"])
    
            # Add glob filters if provided
            if globs:
                for glob in globs:
                    cmd.extend(["-g", glob])
    
            # Add query and base path
            cmd.append(query)
            cmd.append(base_path)
    
            # Run ripgrep
            result = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
            )
    
            # Parse the output
            lines = result.stdout.strip().split("\n")
            file_match_count = {}
            
            for line in lines:
                if not line:
                    continue
                    
                parts = line.split(":", 2)
                if len(parts) >= 2:
                    file_path = parts[0]
                    if file_path not in file_match_count:
                        file_match_count[file_path] = 0
                    file_match_count[file_path] += 1
    
            # Sort files by match count
            sorted_files = sorted(
                file_match_count.items(), key=lambda x: x[1], reverse=True
            )
            
            # Limit the number of files
            top_files = sorted_files[:max_files]
            
            # Add the files to relevant files
            for file_path, count in top_files:
                self.relevant_files.append(
                    {
                        "file_path": file_path,
                        "reasoning": f"Found {count} matches for query '{query}'",
                    }
                )
    
            console.log(f"[dim]Found {len(top_files)} files with matches[/dim]")
            return {
                "reasoning": f"Found {len(top_files)} files with matches for query '{query}'",
                "files": [
                    {"file_path": file_path, "matches": count}
                    for file_path, count in top_files
                ],
            }
        except Exception as e:
            console.log(f"[red]Error searching codebase: {str(e)}[/red]")
            return {
                "reasoning": f"Error: {str(e)}",
                "files": [],
            }
    
    def run(self, prompt: str, directory: str = None, globs: List[str] = None, 
          extensions: List[str] = None, max_files: int = 10, 
          file_line_limit: int = 500, output_file: str = None, 
          use_ripgrep: bool = False) -> Dict[str, Any]:
        """Main execution method for the codebase context agent.
        
        Args:
            prompt: User's query about the codebase
            directory: Directory to search
            globs: List of glob patterns for filtering
            extensions: List of file extensions to filter by
            max_files: Maximum number of files to return
            file_line_limit: Maximum number of lines per file
            output_file: Path to output file for results
            use_ripgrep: Whether to use ripgrep for search
            
        Returns:
            Dict[str, Any]: Results of the agent's execution with relevant files
        """
        start_time = time.time()
        self.relevant_files = []
        self.file_line_limit = file_line_limit
        self.max_files = max_files
        self.use_ripgrep = use_ripgrep
        
        console.log(f"[bold green]Running Codebase Context Agent[/bold green]")
        console.log(f"Prompt: {prompt}")
        
        try:
            # If ripgrep is enabled, use it for searching
            if use_ripgrep:
                self.search_codebase_with_ripgrep(
                    "Using ripgrep to search codebase as requested", 
                    prompt, 
                    directory,
                    max_files,
                    extensions,
                    globs
                )
            else:
                # List files in the repository
                file_list = self.git_list_files(
                    "Finding all available files in the repository", 
                    directory, 
                    globs, 
                    extensions
                )
                
                # Check file line lengths
                file_line_dict = self.check_file_paths_line_length(
                    "Filtering files by line length limit", 
                    file_list, 
                    file_line_limit
                )
                
                # Process files in batches
                file_paths = list(file_line_dict.keys())
                total_files = len(file_paths)
                
                for i in range(0, min(total_files, max_files), BATCH_SIZE):
                    batch = file_paths[i:min(i + BATCH_SIZE, total_files, max_files)]
                    
                    console.log(f"[yellow]Processing batch {i//BATCH_SIZE + 1}[/yellow]")
                    self.determine_if_files_are_relevant(
                        "Checking file relevance based on content", 
                        batch,
                        prompt
                    )
                    
                    # Stop if we've reached the maximum number of files
                    if len(self.relevant_files) >= max_files:
                        console.log(f"[yellow]Reached maximum number of files ({max_files})[/yellow]")
                        break
            
            # Display results
            if self.relevant_files:
                table = Table(title=f"Relevant Files for: {prompt}")
                table.add_column("File Path", style="cyan")
                table.add_column("Reasoning", style="green")
                
                for file in self.relevant_files:
                    table.add_row(
                        file["file_path"],
                        file["reasoning"][:100] + "..." if len(file["reasoning"]) > 100 else file["reasoning"],
                    )
                
                console.print(table)
                
                # Save results to file if requested
                if output_file:
                    with open(output_file, "w") as f:
                        json.dump(self.relevant_files, f, indent=2)
                    console.log(f"[green]Results saved to {output_file}[/green]")
                
                # Store results in memory
                self.remember({
                    "prompt": prompt,
                    "relevant_files": self.relevant_files,
                    "timestamp": time.time()
                })
                
                # Display token usage
                if not use_ripgrep:
                    table = Table(title="Token Usage Statistics")
                    table.add_column("Metric", style="cyan")
                    table.add_column("Count", style="magenta", justify="right")
                    
                    table.add_row("Input Tokens", f"{self.total_input_tokens:,}")
                    table.add_row("Output Tokens", f"{self.total_output_tokens:,}")
                    table.add_row("Total Tokens", f"{self.total_input_tokens + self.total_output_tokens:,}")
                    
                    console.print(table)
            else:
                console.print("[yellow]No relevant files found.[/yellow]")
            
            elapsed_time = time.time() - start_time
            console.log(f"[dim]Completed in {elapsed_time:.2f} seconds[/dim]")
            
            return {
                "success": True,
                "prompt": prompt,
                "relevant_files": self.relevant_files,
                "token_usage": {
                    "input_tokens": self.total_input_tokens,
                    "output_tokens": self.total_output_tokens,
                    "total_tokens": self.total_input_tokens + self.total_output_tokens
                },
                "elapsed_time": elapsed_time
            }
            
        except Exception as e:
            console.log(f"[red]Error running codebase context agent: {str(e)}[/red]")
            return {
                "success": False,
                "prompt": prompt,
                "error": str(e)
            }


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Codebase Context Agent")
    parser.add_argument("--prompt", required=True, help="The prompt to find relevant files for")
    parser.add_argument("--directory", default=".", help="Directory to search in")
    parser.add_argument("--globs", help="Comma-separated list of glob patterns")
    parser.add_argument("--extensions", help="Comma-separated list of file extensions")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of files to return")
    parser.add_argument("--file-line-limit", type=int, default=500, help="Maximum number of lines per file")
    parser.add_argument("--output-file", help="File to save results to")
    parser.add_argument("--use-ripgrep", action="store_true", help="Use ripgrep for searching")
    
    args = parser.parse_args()
    
    # Process glob patterns
    globs = args.globs.split(",") if args.globs else []
    
    # Process extensions
    extensions = args.extensions.split(",") if args.extensions else []
    
    # Initialize and run the agent
    agent = CodebaseContextAgent()
    agent.run(
        args.prompt,
        args.directory,
        globs,
        extensions,
        args.limit,
        args.file_line_limit,
        args.output_file,
        args.use_ripgrep
    )


if __name__ == "__main__":
    main()
