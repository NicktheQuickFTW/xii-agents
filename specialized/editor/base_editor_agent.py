from typing import Any, Dict, List, Optional
from core.base_agent import BaseAgent
import os
from pathlib import Path
from dotenv import load_dotenv

class BaseEditorAgent(BaseAgent):
    """Base class for file and code editing agents."""
    
    def __init__(self, agent_id: str, memory_type: str = "mem0"):
        """Initialize the editor agent.
        
        Args:
            agent_id: Unique identifier for the agent
            memory_type: Type of memory to use
        """
        super().__init__(agent_id, memory_type)
        self.workspace_root = os.getenv("WORKSPACE_ROOT", ".")
        
    def read_file(self, file_path: str) -> str:
        """Read contents of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: File contents
        """
        full_path = os.path.join(self.workspace_root, file_path)
        with open(full_path, 'r') as f:
            return f.read()
    
    def write_file(self, file_path: str, content: str) -> bool:
        """Write content to a file.
        
        Args:
            file_path: Path to the file
            content: Content to write
            
        Returns:
            bool: True if successful
        """
        full_path = os.path.join(self.workspace_root, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w') as f:
            f.write(content)
        return True
    
    def edit_file(self, file_path: str, edits: List[Dict[str, Any]]) -> bool:
        """Apply edits to a file.
        
        Args:
            file_path: Path to the file
            edits: List of edit operations
            
        Returns:
            bool: True if successful
        """
        content = self.read_file(file_path)
        lines = content.split('\n')
        
        for edit in edits:
            if edit['type'] == 'replace':
                start_line = edit['start_line'] - 1
                end_line = edit['end_line']
                lines[start_line:end_line] = edit['content'].split('\n')
            elif edit['type'] == 'insert':
                line = edit['line'] - 1
                lines[line:line] = edit['content'].split('\n')
            elif edit['type'] == 'delete':
                start_line = edit['start_line'] - 1
                end_line = edit['end_line']
                lines[start_line:end_line] = []
                
        return self.write_file(file_path, '\n'.join(lines))
    
    def search_files(self, pattern: str, include: Optional[str] = None, exclude: Optional[str] = None) -> List[str]:
        """Search for files matching a pattern.
        
        Args:
            pattern: Search pattern
            include: Optional include pattern
            exclude: Optional exclude pattern
            
        Returns:
            List[str]: List of matching file paths
        """
        matches = []
        for root, _, files in os.walk(self.workspace_root):
            for file in files:
                if include and not file.endswith(include):
                    continue
                if exclude and file.endswith(exclude):
                    continue
                    
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        if pattern in content:
                            matches.append(os.path.relpath(file_path, self.workspace_root))
                except:
                    continue
                    
        return matches
    
    def execute_command(self, command: str) -> Dict[str, Any]:
        """Execute a shell command.
        
        Args:
            command: Command to execute
            
        Returns:
            Dict[str, Any]: Command results
        """
        import subprocess
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace_root,
                capture_output=True,
                text=True
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            } 