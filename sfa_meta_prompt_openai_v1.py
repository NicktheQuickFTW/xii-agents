#!/usr/bin/env python3

# /// script
# dependencies = [
#   "openai>=1.62.0",
# ]
# ///

"""
/// Example Usage

# Generate a meta prompt using command-line arguments.
# Optional arguments are marked with a ?.

uv run sfa_meta_prompt_openai_v1.py \
    --purpose "generate mermaid diagrams" \
    --instructions "generate a mermaid valid chart, use diagram type specified or default flow, use examples to understand the structure of the output" \
    --sections "examples, user-prompt" \
    --examples "create examples of 3 basic mermaid charts with <user-chart-request> and <chart-response> blocks" \
    --variables "user-prompt"

# Without optional arguments, the script will enter interactive mode.
uv run sfa_meta_prompt_openai_v1.py \
    --purpose "generate mermaid diagrams" \
    --instructions "generate a mermaid valid chart, use diagram type specified or default flow, use examples to understand the structure of the output"

# Alternatively, just run the script without any flags to enter interactive mode.
uv run sfa_meta_prompt_openai_v1.py

///
"""

import os
import sys
import argparse
import json
from typing import Dict, List, Any, Optional
import openai
from openai import OpenAI
from core.base_agent import BaseAgent

META_PROMPT = """<purpose>
    You are an expert prompt engineer, capable of creating detailed and effective prompts for language models.
    
    Your task is to generate a comprehensive prompt based on the user's input structure.
    
    Follow the instructions closely to generate a new prompt template.
</purpose>

<instructions>
    <instruction>Analyze the user-input carefully, paying attention to the purpose, required sections, and variables.</instruction>
    <instruction>Create a detailed prompt that includes all specified sections and incorporates the provided variables.</instruction>
    <instruction>Use clear and concise language in the generated prompt.</instruction>
    <instruction>Ensure that the generated prompt maintains a logical flow and structure.</instruction>
    <instruction>Include placeholders for variables values in the format [[variable-name]].</instruction>
    <instruction>If a section is plural, create a nested section with three items in the singular form.</instruction>
    <instruction>The key xml blocks are purpose, instructions, sections, examples, user-prompt.
    <instruction>Purpose defines the high level goal of the prompt.</instruction>
    <instruction>Instructions are the detailed instructions for the prompt.</instruction>
    <instruction>Sections are arbitrary blocks to include in the prompt.</instruction>
    <instruction>Examples are showcases of what the output should be for the prompt. Use this to steer the structure of the output based on the user-input. This will typically be a list of examples with the expected output.</instruction>
    <instruction>Variables are placeholders for values to be substituted in the prompt.</instruction>
    <instruction>Not every section is required, but purpose and instructions are typically essential. Create the xml blocks based on the user-input.</instruction>
    <instruction>Use the examples to understand the structure of the output.</instruction>
    <instruction>Your output should be in XML format, mirroring the structure of the examples output.</instruction>
    <instruction>Exclude CDATA sections in your output.</instruction>
    <instruction>Response exclusively with the desired output, no other text.</instruction>
    <instruction>If the user-input is structured like the input-format, use it as is. If it's not, infer the purpose, sections, and variables from the user-input.</instruction>
    <instruction>The goal is to fill in the blanks and best infer the purpose, instructions, sections, and variables from the user-input. If instructions are given, use them to guide the other xml blocks.</instruction>
    <instruction>Emphasize exact XML structure and nesting. Clearly define which blocks must contain which elements to ensure a well-formed output.</instruction>
    <instruction>Ensure that each section builds logically upon the previous ones, creating a coherent narrative from purpose to instructions, sections, and examples.</instruction>
    <instruction>Use direct, simple language and avoid unnecessary complexity to make the final prompt easy to understand.</instruction>
    <instruction>After creating the full prompt, perform a final validation to confirm that all placeholders, instructions, and examples are included, properly formatted, and consistent.</instruction>
    <instruction>If examples are not requested, don't create them.</instruction>
    <instruction>If sections are not requested, don't create them.</instruction>
    <instruction>If variables are not requested, just create a section for the user-input.</instruction>
</instructions>

<input-format>
    Purpose: [main purpose of the prompt], Instructions: [list of details of how to generate the output comma sep], Sections: [list of additional sections to include, e.g., examples, user-prompt], Examples: [list of examples of the output for the prompt], Variables: [list of variables to be used in the prompt]
</input-format>

<examples>
    <example>
        <input>
            Purpose: generate mermaid diagrams. Instructions: generate a mermaid valid chart, use diagram type specified or default flow, use examples to understand the structure of the output. Sections: examples, user-prompt. Variables: user-prompt
        </input>
        <o>
<![CDATA[
You are a world-class expert at creating mermaid charts.

You follow the instructions perfectly to generate mermaid charts.

<instructions>
    <instruction>Generate valid a mermaid chart based on the user-prompt.</instruction>
    <instruction>Use the diagram type specified in the user-prompt if non-specified use a flowchart.</instruction>
    <instruction>Use the examples to understand the structure of the output.</instruction>
</instructions>

<examples>
    <example>
        <user-chart-request>
            Create a flowchart that shows A flowing to E. At C, branch out to H and I.
        </user-chart-request>
        <chart-response>
            graph LR;
                A
                B
                C
                D
                E
                H
                I
                A --> B
                A --> C
                A --> D
                C --> H
                C --> I
                D --> E
        </chart-response>
    </example>
    <example>
        <user-chart-request>
            Build a pie chart that shows the distribution of Apples: 40, Bananas: 35, Oranges: 25.
        </user-chart-request>
        <chart-response>
            pie title Distribution of Fruits
                "Apples" : 40
                "Bananas" : 35
                "Oranges" : 25
        </chart-response>
    </example>
    <example>
        <user-chart-request>
            State diagram for a traffic light. Still, Moving, Crash.
        </user-chart-request>
        <chart-response>
            stateDiagram-v2
                [*] --> Still
                Still --> [*]
                Still --> Moving
                Moving --> Still
                Moving --> Crash
                Crash --> [*]
        </chart-response>
    </example>
    <example>
        <user-chart-request>
            Create a timeline of major social media platforms from 2002 to 2006.
        </user-chart-request>
        <chart-response>
            timeline
                title History of Social Media Platforms
                2002 : LinkedIn
                2004 : Facebook
                        : Google
                2005 : Youtube
                2006 : Twitter
        </chart-response>
    </example>
    </examples>

<user-prompt>
    [[user-prompt]]
</user-prompt>

Your mermaid chart:
</o>
    </example>
    <example>
        <input>
            Purpose: review git diff to improve code quality. Instructions: Review git diff, give suggestions for improvements to the code organized in a list sorted by priority. Sections: git-diff. Variables: git-diff
        </input>
        <o>
<![CDATA[
<purpose>
    You are an expert at reviewing git diffs to improve code quality.
    You follow the instructions perfectly to review git diffs.
</purpose>

<instructions>
    <instruction>Review the git diff and provide a detailed analysis of the changes made.</instruction>
    <instruction>Give suggestions for improvements to the code organized in a list sorted by priority.</instruction>
    <instruction>Think through the changes in a wholistic manner and offer suggestions for improvements.</instruction>
</instructions>

<git-diff>
    [[git-diff]]
</git-diff>

Your review of the git diff:
]]>
        </o>
    </example>
    <example>
        <input>
            Purpose: convert user mathematical expressions into LaTeX. Instructions: Take the user-input, which is a mathematical expression in plain text, and output a properly formatted LaTeX equation. Sections: user-input. Variables: user-input
        </input>
        <o>
<![CDATA[
<purpose>
    You are a highly skilled mathematician who can transform plain text math expressions into LaTeX formatted equations.
</purpose>

<instructions>
    <instruction>Take the user-input, which is a mathematical expression in plain text, and output a properly formatted LaTeX equation.</instruction>
    <instruction>Ensure the LaTeX output follows standard mathematical notation and formatting principles.</instruction>
</instructions>

<user-input>
    [[user-input]]
</user-input>

Your LaTeX equation:
]]>
        </o>
    </example>
</examples>

<user-input>
    {{user_input}}
</user-input>
"""


class MetaPromptAgent(BaseAgent):
    """Meta prompt agent using OpenAI for prompt generation."""
    
    def __init__(self, agent_id: str = "meta_prompt_openai", memory_type: str = "mem0"):
        """Initialize the Meta Prompt OpenAI agent.
        
        Args:
            agent_id: Unique identifier for the agent
            memory_type: Type of memory to use
        """
        super().__init__(agent_id, memory_type)
        self.agent_type = "prompt"
        
        # Set up OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4-turbo")
        
    def generate_prompt(self, instruction: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate a prompt based on instruction and context.
        
        Args:
            instruction: The instruction for prompt generation
            context: Optional context information
            
        Returns:
            str: Generated prompt
        """
        context = context or {}
        
        # Prepare the input for the meta prompt
        user_input = f"Purpose: {context.get('purpose', '')}"
        
        if 'instructions' in context:
            user_input += f". Instructions: {context['instructions']}"
            
        if 'sections' in context:
            user_input += f". Sections: {context['sections']}"
            
        if 'examples' in context:
            user_input += f". Examples: {context['examples']}"
            
        if 'variables' in context:
            user_input += f". Variables: {context['variables']}"
        
        # Replace the placeholder in the meta prompt
        prompt = META_PROMPT.replace("{{user_input}}", user_input)
        
        # Call OpenAI to generate the prompt
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert prompt engineer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        generated_prompt = response.choices[0].message.content
        
        # Store the result in memory
        self.remember({
            "instruction": instruction,
            "context": context,
            "generated_prompt": generated_prompt,
            "model": self.model
        })
        
        return generated_prompt
    
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a specialized task.
        
        Args:
            task: Task specification with parameters
            
        Returns:
            Dict[str, Any]: Task results
        """
        # Extract task parameters
        instruction = task.get("instruction", "")
        context = task.get("context", {})
        
        # Generate the prompt
        generated_prompt = self.generate_prompt(instruction, context)
        
        return {
            "success": True,
            "result": generated_prompt,
            "task": task
        }
    
    def compose_agents(self, agents: List[BaseAgent], task: Dict[str, Any]) -> Dict[str, Any]:
        """Compose multiple agents to complete a complex task.
        
        Args:
            agents: List of agents to compose
            task: The complex task to complete
            
        Returns:
            Dict[str, Any]: Task results
        """
        # This implementation assumes agents are used in sequence
        result = task
        
        for agent in agents:
            result = agent.process_task(result)
            
        return result
    
    def evaluate_result(self, result: Dict[str, Any], criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a result against specified criteria.
        
        Args:
            result: The result to evaluate
            criteria: Evaluation criteria
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        # Implement simple evaluation based on criteria
        evaluation = {
            "score": 0,
            "feedback": []
        }
        
        if "length" in criteria and isinstance(result.get("result"), str):
            actual_length = len(result["result"])
            min_length = criteria["length"].get("min", 0)
            max_length = criteria["length"].get("max", float("inf"))
            
            if actual_length < min_length:
                evaluation["feedback"].append(f"Result is too short: {actual_length} < {min_length}")
            elif actual_length > max_length:
                evaluation["feedback"].append(f"Result is too long: {actual_length} > {max_length}")
            else:
                evaluation["score"] += 1
        
        if "contains" in criteria and isinstance(result.get("result"), str):
            for required in criteria["contains"]:
                if required in result["result"]:
                    evaluation["score"] += 1
                else:
                    evaluation["feedback"].append(f"Result does not contain: {required}")
        
        # Calculate final score as percentage
        total_criteria = len(criteria)
        if total_criteria > 0:
            evaluation["percentage"] = (evaluation["score"] / total_criteria) * 100
        else:
            evaluation["percentage"] = 100
            
        return evaluation
    
    def run(self, purpose: str, instructions: str = None, sections: str = None, 
          examples: str = None, variables: str = None) -> Dict[str, Any]:
        """Main execution method for the meta prompt agent.
        
        Args:
            purpose: The main purpose of the prompt
            instructions: List of details of how to generate the output
            sections: List of additional sections to include
            examples: List of examples of the output for the prompt
            variables: List of variables to be used in the prompt
            
        Returns:
            Dict[str, Any]: Results of the agent's execution
        """
        # Prepare context
        context = {"purpose": purpose}
        
        if instructions:
            context["instructions"] = instructions
            
        if sections:
            context["sections"] = sections
            
        if examples:
            context["examples"] = examples
            
        if variables:
            context["variables"] = variables
        
        # Generate the prompt
        generated_prompt = self.generate_prompt("Generate a prompt template", context)
        
        return {
            "success": True,
            "prompt": generated_prompt,
            "context": context
        }


def interactive_input():
    """Collect input from the user in interactive mode.
    
    Returns:
        Dict[str, str]: Dictionary with the collected user input
    """
    print("Interactive Meta Prompt Generator")
    print("=================================")
    print("Please provide the following information (press Enter to skip optional fields):")
    
    purpose = input("Purpose (required): ")
    while not purpose:
        print("Purpose is required.")
        purpose = input("Purpose (required): ")
        
    instructions = input("Instructions (comma-separated): ")
    sections = input("Sections (comma-separated): ")
    examples = input("Examples: ")
    variables = input("Variables (comma-separated): ")
    
    result = {"purpose": purpose}
    
    if instructions:
        result["instructions"] = instructions
        
    if sections:
        result["sections"] = sections
        
    if examples:
        result["examples"] = examples
        
    if variables:
        result["variables"] = variables
        
    return result


def main():
    # Check if any command-line arguments besides the script name were provided
    if len(sys.argv) > 1:
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description="Meta Prompt Generator")
        parser.add_argument("--purpose", help="Main purpose of the prompt", required=False)
        parser.add_argument("--instructions", help="List of instructions, comma-separated", required=False)
        parser.add_argument("--sections", help="List of additional sections to include, comma-separated", required=False)
        parser.add_argument("--examples", help="List of examples of the output for the prompt", required=False)
        parser.add_argument("--variables", help="List of variables to be used in the prompt, comma-separated", required=False)
        
        args = parser.parse_args()
        
        # If purpose is not provided, use interactive mode
        if not args.purpose:
            inputs = interactive_input()
        else:
            inputs = {
                "purpose": args.purpose, 
                "instructions": args.instructions,
                "sections": args.sections, 
                "examples": args.examples, 
                "variables": args.variables
            }
            # Remove None values
            inputs = {k: v for k, v in inputs.items() if v is not None}
    else:
        # No command-line args, use interactive mode
        inputs = interactive_input()
    
    # Initialize and run the agent
    agent = MetaPromptAgent()
    result = agent.run(**inputs)
    
    if result["success"]:
        print("\nGenerated Prompt:")
        print("=================")
        print(result["prompt"])
    else:
        print("\nError generating prompt:")
        print(result.get("error", "Unknown error"))


if __name__ == "__main__":
    main()
