# CLAUDE.md - Single File Agents Repository

## Commands
- **Run agents**: `uv run <agent_filename.py> [options]`

## Environment
- Set API keys before running agents:
  ```bash
  export GEMINI_API_KEY='your-api-key-here'
  export OPENAI_API_KEY='your-api-key-here'
  export ANTHROPIC_API_KEY='your-api-key-here'
  export FIRECRAWL_API_KEY='your-api-key-here'
  ```

## Code Style
- Single file agents with embedded dependencies (using `uv`)
- Dependencies specified at top of file in `/// script` comments
- Include example usage in docstrings
- Detailed error handling with user-friendly messages
- Consistent format for command-line arguments

## Structure
- Each agent focuses on a single capability (DuckDB, SQLite, JQ, etc.)
- Command-line arguments use argparse with consistent patterns
- File naming: `sfa_<capability>_<provider>_v<version>.py`

## Usage
> We use astral `uv` as our python package manager.
>
> This enables us to run SINGLE FILE AGENTS with embedded dependencies.

To run an agent, use the following command:

```bash
uv run sfa_<capability>_<provider>_v<version>.py <arguments>
```