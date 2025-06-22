"""
Simple Pseudocode Generator for CADQuery
Takes plan JSON files from planner agent and generates pseudocode
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PseudocodeStep(BaseModel):
    """Single step in pseudocode"""
    step_number: int
    operation: str
    target: str
    method: str
    parameters: Dict[str, Any]
    comment: str
    dependencies: List[int] = []


class SimplePseudocode(BaseModel):
    """Complete pseudocode structure"""
    plan_id: str
    title: str
    description: str
    imports: List[str]
    constants: Dict[str, Any]
    steps: List[PseudocodeStep]


class SimplePseudocodeGenerator:
    """Generates pseudocode from planner agent output"""

    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        # Initialize Gemini
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=self.api_key,
            temperature=0.1,
            max_tokens=4096
        )

        self.output_parser = StrOutputParser()
        self.prompt = self._create_prompt()
        self.chain = self.prompt | self.llm | self.output_parser

    def _create_prompt(self) -> ChatPromptTemplate:
        """Create prompt for pseudocode generation"""

        system_prompt = """You are a CADQuery pseudocode generator. Convert CAD plans into structured pseudocode.

Your task is to convert each step from the plan into a structured format that can be easily converted to CADQuery code.

For each step in the plan, create a pseudocode entry with:
- step_number: The step number from the plan
- operation: The type of operation (sketch, extrude, cut, modify, etc.)
- target: The variable to store the result (usually "result")
- method: The CADQuery method to call
- parameters: The parameters for the method as a dictionary
- comment: A description of what this step does
- dependencies: List of step numbers this depends on

CADQUERY METHOD MAPPING:
- Workplane.circle -> circle
- Workplane.rect -> rect
- extrude -> extrude
- cut -> cut
- hole -> hole
- fillet -> fillet
- chamfer -> chamfer
- translate -> translate
- rotate -> rotate

OUTPUT JSON FORMAT:
{{
    "plan_id": "from the input plan",
    "title": "from the input plan",
    "description": "from the input plan",
    "imports": ["cadquery as cq"],
    "constants": {{
        "extracted dimensions and parameters from the plan"
    }},
    "steps": [
        {{
            "step_number": 1,
            "operation": "operation type",
            "target": "result",
            "method": "cadquery method",
            "parameters": {{"param": value}},
            "comment": "what this step does",
            "dependencies": []
        }}
    ]
}}

Important: 
- Extract ALL constants from dimensions and step parameters
- Maintain exact step order from the plan
- Use proper CADQuery method names
- Include all parameters from each step"""

        human_prompt = """Convert this CAD plan to structured pseudocode:

{plan_json}

Generate a JSON response with the structured pseudocode."""

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])

    def generate_from_file(self, plan_file: str) -> Optional[SimplePseudocode]:
        """Generate pseudocode from a plan file"""
        try:
            # Load the plan
            with open(plan_file, 'r') as f:
                plan_data = json.load(f)

            logger.info(f"Loaded plan: {plan_data.get('title', 'Unknown')}")

            # Generate pseudocode
            response = self.chain.invoke({"plan_json": json.dumps(plan_data, indent=2)})

            # Parse response
            pseudocode_data = self._parse_response(response)
            if not pseudocode_data:
                logger.error("Failed to parse LLM response")
                return None

            # Create pseudocode object
            pseudocode = SimplePseudocode(**pseudocode_data)
            logger.info(f"Generated pseudocode with {len(pseudocode.steps)} steps")

            return pseudocode

        except Exception as e:
            logger.error(f"Error generating pseudocode: {str(e)}")
            return None

    def _parse_response(self, response: str) -> Optional[Dict]:
        """Parse JSON from LLM response"""
        try:
            # Try direct parsing
            return json.loads(response)
        except:
            pass

        # Try extracting from code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            try:
                return json.loads(response[start:end])
            except:
                pass

        # Try finding JSON boundaries
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except:
            pass

        return None

    def display_pseudocode(self, pseudocode: SimplePseudocode):
        """Display pseudocode in readable format"""
        print(f"\n{'='*70}")
        print(f"PSEUDOCODE: {pseudocode.title}")
        print(f"{'='*70}")
        print(f"Plan ID: {pseudocode.plan_id}")
        print(f"Description: {pseudocode.description}")

        print(f"\nIMPORTS:")
        for imp in pseudocode.imports:
            print(f"  import {imp}")

        print(f"\nCONSTANTS:")
        for name, value in pseudocode.constants.items():
            print(f"  {name} = {value}")

        print(f"\nSTEPS:")
        for step in pseudocode.steps:
            print(f"\n  Step {step.step_number}: {step.comment}")
            if step.dependencies:
                print(f"    Dependencies: {step.dependencies}")
            print(f"    {step.target} = {step.target}.{step.method}({step.parameters})")

    def save_pseudocode(self, pseudocode: SimplePseudocode, output_dir: str = "pseudocode_output"):
        """Save pseudocode to files"""
        os.makedirs(output_dir, exist_ok=True)

        # Save as JSON
        json_file = os.path.join(output_dir, f"{pseudocode.plan_id}_pseudocode.json")
        with open(json_file, 'w') as f:
            json.dump(pseudocode.dict(), f, indent=2)

        # Save as readable text
        txt_file = os.path.join(output_dir, f"{pseudocode.plan_id}_pseudocode.txt")
        with open(txt_file, 'w') as f:
            f.write(f"# {pseudocode.title}\n")
            f.write(f"# {pseudocode.description}\n\n")

            f.write("# Imports\n")
            for imp in pseudocode.imports:
                f.write(f"import {imp}\n")

            f.write("\n# Constants\n")
            for name, value in pseudocode.constants.items():
                f.write(f"{name} = {value}\n")

            f.write("\n# Main Construction\n")
            f.write("result = cq.Workplane('XY')\n\n")

            for step in pseudocode.steps:
                f.write(f"# Step {step.step_number}: {step.comment}\n")
                if step.parameters:
                    params = ", ".join([f"{k}={v}" for k, v in step.parameters.items()])
                    f.write(f"{step.target} = {step.target}.{step.method}({params})\n\n")
                else:
                    f.write(f"{step.target} = {step.target}.{step.method}()\n\n")

        print(f"\nSaved files:")
        print(f"  JSON: {json_file}")
        print(f"  Text: {txt_file}")


def main():
    """Process plan files and generate pseudocode"""
    import sys
    import glob

    generator = SimplePseudocodeGenerator()

    # Check if a specific file was provided as argument
    if len(sys.argv) > 1:
        plan_file = sys.argv[1]
        if os.path.exists(plan_file):
            print(f"Processing specified file: {plan_file}")
            pseudocode = generator.generate_from_file(plan_file)
            if pseudocode:
                generator.display_pseudocode(pseudocode)
                generator.save_pseudocode(pseudocode)
                print("✓ Success!")
            else:
                print("✗ Failed to generate pseudocode")
        else:
            print(f"File not found: {plan_file}")
        return

    # Look for plan files with various patterns
    all_json_files = glob.glob("*.json")

    # Filter out files that are likely outputs (not inputs)
    plan_files = []
    for f in all_json_files:
        # Skip files that are outputs from other agents
        if any(skip in f.lower() for skip in ["pseudocode", "code", "output"]):
            continue
        # Include files that might be plans
        if any(include in f.lower() for include in ["plan", "gear", "model", "design", "test"]):
            plan_files.append(f)

    # If no files found with specific patterns, show all JSON files
    if not plan_files:
        print("No plan files found with typical naming patterns.")
        print("\nAll JSON files in current directory:")
        for i, file in enumerate(all_json_files):
            print(f"  {i+1}. {file}")

        if all_json_files:
            print("\nWould you like to process one of these files?")
            choice = input("Enter file number (or 'n' to exit): ")
            if choice.lower() != 'n':
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(all_json_files):
                        plan_files = [all_json_files[idx]]
                except:
                    print("Invalid choice")
                    return
        else:
            print("\nNo JSON files found in current directory!")
            print("Make sure you're in the directory with your plan files.")
            return

    if plan_files:
        print(f"\nFound {len(plan_files)} plan file(s):")
        for i, file in enumerate(plan_files):
            print(f"  {i+1}. {file}")

        # Process each file
        for plan_file in plan_files:
            print(f"\n{'='*50}")
            print(f"Processing: {plan_file}")
            print("="*50)

            pseudocode = generator.generate_from_file(plan_file)

            if pseudocode:
                generator.display_pseudocode(pseudocode)
                generator.save_pseudocode(pseudocode)
                print("✓ Success!")
            else:
                print("✗ Failed to generate pseudocode")


if __name__ == "__main__":
    main()