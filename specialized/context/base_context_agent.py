from typing import Any, Dict, List, Optional
from core.base_agent import BaseAgent
import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv

class BaseContextAgent(BaseAgent):
    """Base class for context gathering agents that retrieve code context."""
    
    def __init__(self, agent_id: str, memory_type: str = "mem0"):
        """Initialize the context agent.
        
        Args:
            agent_id: Unique identifier for the agent
            memory_type: Type of memory to use
        """
        super().__init__(agent_id, memory_type)
        self.workspace_root = os.getenv("WORKSPACE_ROOT", ".")
        
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
        raise NotImplementedError("Subclasses must implement this method")
    
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
    
    def search_with_ripgrep(self, pattern: str, file_pattern: Optional[str] = None,
                          ignore_case: bool = True, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search codebase using ripgrep.
        
        Args:
            pattern: Search pattern
            file_pattern: Optional file pattern
            ignore_case: Whether to ignore case
            max_results: Maximum number of results
            
        Returns:
            List[Dict[str, Any]]: List of matching code snippets
        """
        try:
            cmd = ['rg', '--json']
            
            if ignore_case:
                cmd.append('-i')
                
            if max_results:
                cmd.extend(['-m', str(max_results)])
                
            cmd.append(pattern)
            
            if file_pattern:
                cmd.extend(['-g', file_pattern])
                
            cmd.append(self.workspace_root)
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            matches = []
            for line in result.stdout.splitlines():
                if not line:
                    continue
                    
                try:
                    data = eval(line)  # Parse JSON-like output
                    if 'type' in data and data['type'] == 'match':
                        match_data = data['data']
                        file_path = match_data['path']['text']
                        line_number = match_data['line_number']
                        line_text = match_data['lines']['text'].strip()
                        
                        # Read context around the match
                        context = self.read_file(
                            file_path,
                            max(1, line_number - 5),
                            line_number + 5
                        )
                        
                        matches.append({
                            'file_path': file_path,
                            'line_number': line_number,
                            'text': line_text,
                            'context': context
                        })
                except:
                    pass
                    
            return matches[:max_results]
        except Exception as e:
            return [{"error": f"Search failed: {str(e)}"}] 