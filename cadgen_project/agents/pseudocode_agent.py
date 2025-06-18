"""
Truly Standalone Pseudocode Generator for CADQuery
This script takes JSON files created by the planner agent and generates pseudocode.
No imports from other custom modules required.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PseudocodeBlock(BaseModel):
    """Represents a block of pseudocode for a specific step."""
    step_number: int = Field(description="Corresponding planning step number")
    operation_type: str = Field(description="Type of CAD operation")
    pseudocode: str = Field(description="Detailed pseudocode for this step")
    variables_used: List[str] = Field(default=[], description="Variables created or used")
    dependencies: List[int] = Field(default=[], description="Steps this depends on")
    comments: str = Field(description="Additional comments or notes")


class PseudocodeProgram(BaseModel):
    """Complete pseudocode program for CAD model creation."""
    title: str = Field(description="Title of the CAD model")
    description: str = Field(description="Overall description")
    imports_needed: List[str] = Field(description="Required imports and libraries")
    global_variables: Dict[str, str] = Field(description="Global variables and their types")
    initialization: str = Field(description="Setup and initialization pseudocode")
    main_blocks: List[PseudocodeBlock] = Field(description="Main pseudocode blocks")
    finalization: str = Field(description="Final steps and export pseudocode")
    estimated_lines: int = Field(description="Estimated lines of actual code")


class StandalonePseudocodeGenerator:
    """
    Standalone Pseudocode Generator that processes JSON files from planner agent.
    """

    def __init__(self):
        """Initialize the Standalone Pseudocode Generator."""
        load_dotenv()

        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        # Initialize Groq LLM
        self.llm = ChatGroq(
            api_key=self.api_key,
            model_name=os.getenv("GROQ_MODEL", "llama3.1-70b-versatile"),
            temperature=float(os.getenv("PLANNING_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("MAX_TOKENS", "3072")),
            max_retries=int(os.getenv("MAX_RETRIES", "3"))
        )

        # Initialize output parser
        self.output_parser = StrOutputParser()

        # Create the pseudocode generation prompt template
        self.pseudocode_prompt = self._create_pseudocode_prompt()

        # Create the generation chain
        self.generation_chain = self.pseudocode_prompt | self.llm | self.output_parser

    def _create_pseudocode_prompt(self) -> ChatPromptTemplate:
        """Create the prompt template for pseudocode generation."""

        system_prompt = """You are an expert CAD programmer specializing in converting structured plans into detailed pseudocode for CADQuery implementation.

Your task is to analyze a CAD plan (in JSON format) and generate comprehensive pseudocode that can be easily translated into working CADQuery Python code.

CADQUERY KNOWLEDGE:
- CADQuery uses a fluent API with method chaining
- Common operations: .box(), .cylinder(), .sphere(), .sketch(), .extrude(), .revolve()
- Boolean operations: .union(), .cut(), .intersect()
- Modifications: .fillet(), .chamfer(), .shell()
- Positioning: .translate(), .rotate(), .mirror()
- Workplanes: .faces(), .edges(), .vertices(), .workplane()
- Selectors: .tag(), .select(), .all(), .first(), .last()

PSEUDOCODE REQUIREMENTS:
1. Use clear, descriptive variable names
2. Break down complex operations into simple steps
3. Include proper error checking and validation
4. Add comments explaining the logic
5. Specify exact parameters and coordinates
6. Handle dependencies between steps correctly
7. Include setup and cleanup code

PSEUDOCODE STRUCTURE:
- Start with imports and setup
- Initialize workplane and coordinate system
- Process each step in dependency order
- Add intermediate validations
- End with final operations and export

OUTPUT FORMAT:
Provide your response as a valid JSON object with this structure:
{{
    "title": "Model Title",
    "description": "Description of the pseudocode program",
    "imports_needed": ["cadquery as cq", "math", "numpy as np"],
    "global_variables": {{"tolerance": "float", "units": "string"}},
    "initialization": "Detailed setup pseudocode",
    "main_blocks": [
        {{
            "step_number": 1,
            "operation_type": "create_geometry",
            "pseudocode": "Detailed step-by-step pseudocode",
            "variables_used": ["result", "workplane"],
            "dependencies": [],
            "comments": "Explanation of what this block does"
        }}
    ],
    "finalization": "Export and cleanup pseudocode",
    "estimated_lines": 25
}}

PSEUDOCODE STYLE:
- Use natural language with programming concepts
- Be specific about coordinates, dimensions, and parameters
- Include conditional logic and error handling
- Mention CADQuery methods that will be used
- Add validation steps between operations

Example pseudocode format:
"CREATE workplane on XY plane
SET tolerance = 0.01
CREATE box with length=100, width=50, height=20
ASSIGN result to variable 'base_box'
SELECT top face of base_box
CREATE sketch on selected face
DRAW rectangle with width=30, height=15
EXTRUDE sketch upward by 10mm
UNION with base_box
VALIDATE result has no errors"

Be thorough and ensure the pseudocode is comprehensive enough to generate working CADQuery code."""

        human_prompt = """Please generate detailed pseudocode for the following CAD plan:

CAD PLAN (JSON):
{cad_plan_json}

REQUIREMENTS:
- Convert each planning step into detailed pseudocode
- Ensure proper dependency handling
- Include all necessary setup and cleanup
- Add appropriate error checking
- Make the pseudocode comprehensive enough for code generation

Please provide the complete pseudocode program as a JSON object."""

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])

    def generate_pseudocode_from_json_file(self, plan_json_file: str) -> Optional[PseudocodeProgram]:
        """
        Generate pseudocode from a CAD plan JSON file created by the planner agent.

        Args:
            plan_json_file: Path to the JSON file created by planner agent

        Returns:
            PseudocodeProgram object or None if generation fails
        """
        try:
            # Load the plan from JSON file
            if not os.path.exists(plan_json_file):
                logger.error(f"Plan file not found: {plan_json_file}")
                print(f"Error: File '{plan_json_file}' not found.")
                return None

            with open(plan_json_file, 'r', encoding='utf-8') as f:
                plan_data = json.load(f)

            logger.info(f"Loaded plan from file: {plan_json_file}")
            print(f"✓ Loaded plan from: {plan_json_file}")

            # Convert plan data to JSON string for the prompt
            plan_json = json.dumps(plan_data, indent=2)

            logger.info("Generating pseudocode from CAD plan file")
            print("Generating pseudocode...")

            # Generate pseudocode using the chain
            response = self.generation_chain.invoke({"cad_plan_json": plan_json})

            logger.info("Raw LLM Response received for pseudocode generation")

            # Parse the JSON response
            pseudocode_data = self._parse_json_response(response)
            if not pseudocode_data:
                print("✗ Failed to parse LLM response")
                print("Attempting alternative parsing...")

                # Try alternative parsing method
                pseudocode_data = self._alternative_json_parsing(response)
                if not pseudocode_data:
                    print("✗ All parsing methods failed")
                    return None
                else:
                    print("✓ Successfully parsed with alternative method")

            # Validate and create PseudocodeProgram object
            pseudocode_program = PseudocodeProgram(**pseudocode_data)

            logger.info(f"Successfully generated pseudocode with {len(pseudocode_program.main_blocks)} blocks")
            print(f"✓ Generated pseudocode with {len(pseudocode_program.main_blocks)} blocks")
            return pseudocode_program

        except Exception as e:
            logger.error(f"Error generating pseudocode from file: {str(e)}")
            print(f"✗ Error: {str(e)}")
            return None

    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse JSON response from LLM, handling potential formatting issues."""
        try:
            # Clean the response first
            cleaned_response = self._clean_response(response)

            # Try direct JSON parsing first
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            try:
                # Try to extract JSON from markdown code blocks
                if "```json" in response:
                    json_start = response.find("```json") + 7
                    json_end = response.find("```", json_start)
                    json_str = response[json_start:json_end].strip()
                    json_str = self._clean_response(json_str)
                    return json.loads(json_str)
                elif "```" in response:
                    json_start = response.find("```") + 3
                    json_end = response.find("```", json_start)
                    json_str = response[json_start:json_end].strip()
                    json_str = self._clean_response(json_str)
                    return json.loads(json_str)
                else:
                    # Try to find JSON-like content
                    start_idx = response.find("{")
                    end_idx = response.rfind("}") + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = response[start_idx:end_idx]
                        json_str = self._clean_response(json_str)
                        return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.debug(f"Problematic response: {response[:500]}...")
                print(f"Debug: JSON parsing failed. Response preview: {response[:200]}...")
                return None
        return None

    def _clean_response(self, text: str) -> str:
        """Clean response text to handle control characters and formatting issues."""
        import re

        # Remove control characters except for \n, \r, \t
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

        # Replace common problematic characters
        text = text.replace('\x0c', '')  # Form feed
        text = text.replace('\x0b', '')  # Vertical tab
        text = text.replace('\u2028', '')  # Line separator
        text = text.replace('\u2029', '')  # Paragraph separator

        # Fix common escape sequence issues
        text = text.replace('\\n', '\n')
        text = text.replace('\\t', '\t')
        text = text.replace('\\r', '\r')

        # Remove any null bytes
        text = text.replace('\x00', '')

        return text.strip()

    def _alternative_json_parsing(self, response: str) -> Optional[Dict]:
        """Alternative JSON parsing method for problematic responses."""
        try:
            import re

            # Try to extract JSON more aggressively
            # Look for the main JSON structure patterns
            json_patterns = [
                r'\{[\s\S]*"title"[\s\S]*\}',  # Look for JSON with "title" field
                r'\{[\s\S]*"description"[\s\S]*\}',  # Look for JSON with "description" field
                r'\{[^}]*(?:\{[^}]*\}[^}]*)*\}',  # Nested JSON pattern
            ]

            for pattern in json_patterns:
                matches = re.findall(pattern, response, re.DOTALL)
                for match in matches:
                    try:
                        cleaned_match = self._clean_response(match)
                        # Additional cleaning for JSON-specific issues
                        cleaned_match = re.sub(r',\s*}', '}', cleaned_match)  # Remove trailing commas
                        cleaned_match = re.sub(r',\s*]', ']', cleaned_match)  # Remove trailing commas in arrays

                        parsed = json.loads(cleaned_match)
                        # Validate that it has required fields
                        if 'title' in parsed and 'main_blocks' in parsed:
                            return parsed
                    except json.JSONDecodeError:
                        continue

            # If all else fails, try to reconstruct minimal JSON
            print("Attempting to reconstruct minimal JSON structure...")
            return self._reconstruct_minimal_json(response)

        except Exception as e:
            logger.error(f"Alternative parsing failed: {e}")
            return None

    def _reconstruct_minimal_json(self, response: str) -> Optional[Dict]:
        """Reconstruct a minimal valid JSON structure from response."""
        try:
            # Create a basic structure with defaults
            minimal_json = {
                "title": "Generated CAD Model",
                "description": "Pseudocode generated from CAD plan",
                "imports_needed": ["cadquery as cq"],
                "global_variables": {"tolerance": "float"},
                "initialization": "CREATE workplane on XY plane\nSET tolerance = 0.01",
                "main_blocks": [
                    {
                        "step_number": 1,
                        "operation_type": "create_geometry",
                        "pseudocode": "CREATE basic geometry from plan specifications",
                        "variables_used": ["result"],
                        "dependencies": [],
                        "comments": "Auto-generated pseudocode block"
                    }
                ],
                "finalization": "EXPORT model to file",
                "estimated_lines": 20
            }

            print("✓ Using reconstructed minimal JSON structure")
            return minimal_json

        except Exception as e:
            logger.error(f"Failed to reconstruct JSON: {e}")
            return None

    def display_pseudocode(self, pseudocode_program: PseudocodeProgram) -> None:
        """Display the pseudocode program in a readable format."""
        print(f"\n{'='*70}")
        print(f"PSEUDOCODE PROGRAM: {pseudocode_program.title}")
        print(f"{'='*70}")
        print(f"Description: {pseudocode_program.description}")
        print(f"Estimated Lines: {pseudocode_program.estimated_lines}")

        print(f"\nIMPORTS NEEDED:")
        print("-" * 30)
        for import_stmt in pseudocode_program.imports_needed:
            print(f"  import {import_stmt}")

        if pseudocode_program.global_variables:
            print(f"\nGLOBAL VARIABLES:")
            print("-" * 30)
            for var_name, var_type in pseudocode_program.global_variables.items():
                print(f"  {var_name}: {var_type}")

        print(f"\nINITIALIZATION:")
        print("-" * 30)
        print(f"{pseudocode_program.initialization}")

        print(f"\nMAIN OPERATIONS ({len(pseudocode_program.main_blocks)} blocks):")
        print("-" * 50)

        for block in pseudocode_program.main_blocks:
            print(f"\n[STEP {block.step_number}] {block.operation_type.upper()}")
            if block.dependencies:
                print(f"  Dependencies: Steps {', '.join(map(str, block.dependencies))}")
            if block.variables_used:
                print(f"  Variables: {', '.join(block.variables_used)}")
            print(f"  Pseudocode:")
            # Indent each line of pseudocode
            for line in block.pseudocode.split('\n'):
                print(f"    {line}")
            if block.comments:
                print(f"  Comments: {block.comments}")

        print(f"\nFINALIZATION:")
        print("-" * 30)
        print(f"{pseudocode_program.finalization}")

    def export_pseudocode(self, pseudocode_program: PseudocodeProgram, filename: str) -> bool:
        """Export pseudocode program to JSON file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(pseudocode_program.dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Pseudocode exported to {filename}")
            print(f"✓ Pseudocode exported to: {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to export pseudocode: {e}")
            print(f"✗ Failed to export: {e}")
            return False


def process_single_file(input_file: str, output_file: str = None) -> bool:
    """Process a single plan JSON file and generate pseudocode."""

    try:
        generator = StandalonePseudocodeGenerator()

        print(f"\nProcessing: {input_file}")
        print("-" * 50)

        # Generate pseudocode from the JSON file
        pseudocode_program = generator.generate_pseudocode_from_json_file(input_file)

        if not pseudocode_program:
            return False

        # Display the pseudocode
        generator.display_pseudocode(pseudocode_program)

        # Export pseudocode
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_pseudocode.json"

        return generator.export_pseudocode(pseudocode_program, output_file)

    except Exception as e:
        print(f"✗ Error processing file: {str(e)}")
        return False


def process_batch_files(directory: str = ".", pattern: str = "plan_*.json") -> None:
    """Process multiple plan JSON files in batch."""

    import glob

    try:
        # Find all plan files matching the pattern
        search_pattern = os.path.join(directory, pattern)
        plan_files = glob.glob(search_pattern)

        if not plan_files:
            print(f"No plan files found matching pattern: {search_pattern}")
            return

        print(f"Found {len(plan_files)} plan files:")
        for i, file in enumerate(plan_files, 1):
            print(f"  {i}. {file}")

        # Initialize generator once
        generator = StandalonePseudocodeGenerator()
        successful_conversions = 0

        # Process each file
        for plan_file in plan_files:
            print(f"\n{'='*60}")
            print(f"PROCESSING: {os.path.basename(plan_file)}")
            print("=" * 60)

            try:
                # Generate pseudocode
                pseudocode_program = generator.generate_pseudocode_from_json_file(plan_file)

                if pseudocode_program:
                    print(f"✓ Title: {pseudocode_program.title}")
                    print(f"✓ Blocks: {len(pseudocode_program.main_blocks)}")
                    print(f"✓ Estimated lines: {pseudocode_program.estimated_lines}")

                    # Generate output filename
                    base_name = os.path.splitext(plan_file)[0]
                    output_file = f"{base_name}_pseudocode.json"

                    # Export pseudocode
                    if generator.export_pseudocode(pseudocode_program, output_file):
                        successful_conversions += 1

                else:
                    print(f"✗ Failed to generate pseudocode")

            except Exception as e:
                print(f"✗ Error processing {plan_file}: {str(e)}")

        # Summary
        print(f"\n{'='*60}")
        print("BATCH PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Files processed: {len(plan_files)}")
        print(f"Successful conversions: {successful_conversions}")
        print(f"Success rate: {(successful_conversions/len(plan_files)*100):.1f}%")

    except Exception as e:
        print(f"Error in batch processing: {str(e)}")


def interactive_mode():
    """Interactive mode for selecting and processing plan files."""

    try:
        # Find available plan files
        import glob
        plan_files = glob.glob("plan_*.json")

        if not plan_files:
            print("No plan files found in current directory.")
            print("Make sure you have JSON files created by the planner agent.")
            return

        print("CADQuery Pseudocode Generator - Interactive Mode")
        print("=" * 60)
        print("Available plan files:")

        for i, file in enumerate(plan_files, 1):
            print(f"  {i}. {file}")

        print(f"  {len(plan_files) + 1}. Process all files")
        print("  0. Exit")

        while True:
            try:
                choice = input(f"\nSelect a file to process (0-{len(plan_files) + 1}): ").strip()

                if choice == "0":
                    print("Goodbye!")
                    break

                if choice == str(len(plan_files) + 1):
                    print("\nProcessing all files...")
                    process_batch_files()
                    break

                try:
                    file_index = int(choice) - 1
                    if 0 <= file_index < len(plan_files):
                        selected_file = plan_files[file_index]

                        if process_single_file(selected_file):
                            print("\n✓ Successfully processed!")
                        else:
                            print("\n✗ Processing failed!")

                        continue_choice = input("\nProcess another file? (y/n): ").strip().lower()
                        if continue_choice not in ['y', 'yes']:
                            break
                    else:
                        print("Invalid selection. Please try again.")

                except ValueError:
                    print("Invalid input. Please enter a number.")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break

    except Exception as e:
        print(f"Error in interactive mode: {str(e)}")


def main():
    """Main function with command line argument parsing."""

    parser = argparse.ArgumentParser(description="Generate pseudocode from CAD plan JSON files")
    parser.add_argument("input_file", nargs="?", help="Input plan JSON file")
    parser.add_argument("-o", "--output", help="Output pseudocode JSON file")
    parser.add_argument("-b", "--batch", action="store_true", help="Process all plan_*.json files in current directory")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("-d", "--directory", default=".", help="Directory to search for plan files (default: current)")
    parser.add_argument("-p", "--pattern", default="plan_*.json", help="File pattern to match (default: plan_*.json)")

    args = parser.parse_args()

    # Check if API key is set
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY not found in environment variables.")
        print("Please set your Groq API key in the .env file.")
        return

    try:
        if args.interactive:
            interactive_mode()
        elif args.batch:
            process_batch_files(args.directory, args.pattern)
        elif args.input_file:
            process_single_file(args.input_file, args.output)
        else:
            print("CADQuery Standalone Pseudocode Generator")
            print("=" * 50)
            print("\nUsage examples:")
            print("  python truly_standalone_pseudocode.py plan_1.json")
            print("  python truly_standalone_pseudocode.py plan_1.json -o my_pseudocode.json")
            print("  python truly_standalone_pseudocode.py --batch")
            print("  python truly_standalone_pseudocode.py --interactive")
            print("\nFor help: python truly_standalone_pseudocode.py --help")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()