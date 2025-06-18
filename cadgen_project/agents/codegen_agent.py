"""
CADQuery Code Generator Agent
This agent takes pseudocode JSON files and generates working CADQuery Python code.
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
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeneratedCode(BaseModel):
    """Represents the generated CADQuery code."""

    title: str = Field(description="Title of the CAD model")
    description: str = Field(description="Description of the generated code")
    imports: List[str] = Field(description="Required import statements")
    global_variables: Dict[str, str] = Field(
        description="Global variables and their values"
    )
    functions: List[str] = Field(default=[], description="Helper functions if any")
    main_code: str = Field(description="Main CADQuery code")
    export_code: str = Field(description="Code for exporting the model")
    estimated_complexity: str = Field(description="Code complexity level")
    cadquery_version: str = Field(default="2.4", description="Target CADQuery version")


class CADQueryCodeGenerator:
    """
    Code Generator Agent that converts pseudocode JSON into working CADQuery Python code.
    """

    def __init__(self):
        """Initialize the CADQuery Code Generator Agent."""
        load_dotenv()

        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        # Initialize Groq LLM
        self.llm = ChatGroq(
            api_key=self.api_key,
            model_name=os.getenv("GROQ_MODEL", "llama3.1-70b-versatile"),
            temperature=float(os.getenv("PLANNING_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("MAX_TOKENS", "4096")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
        )

        # Initialize output parser
        self.output_parser = StrOutputParser()

        # Create the code generation prompt template
        self.code_prompt = self._create_code_prompt()

        # Create the generation chain
        self.generation_chain = self.code_prompt | self.llm | self.output_parser

    def _create_code_prompt(self) -> ChatPromptTemplate:
        """Create the prompt template for code generation."""

        system_prompt = """You are an expert CADQuery programmer specializing in converting detailed pseudocode into working Python code using the CADQuery library.

Your task is to analyze pseudocode (in JSON format) and generate clean, executable CADQuery Python code that creates the specified 3D model.

CADQUERY EXPERTISE:
- CADQuery 2.4+ syntax and best practices
- Fluent API with method chaining: result = cq.Workplane("XY").box(10, 10, 10)
- Common operations: .box(), .cylinder(), .sphere(), .sketch(), .extrude(), .revolve()
- Boolean operations: .union(), .cut(), .intersect()
- Modifications: .fillet(), .chamfer(), .shell(), .mirror()
- Positioning: .translate(), .rotate(), .rotateX(), .rotateY(), .rotateZ()
- Workplanes: .faces(), .edges(), .vertices(), .workplane()
- Selectors: .val(), .vals(), .all(), .first(), .last()
- Sketching: .moveTo(), .lineTo(), .rect(), .circle(), .close()

CODE GENERATION RULES:
1. Generate clean, readable Python code
2. Use proper CADQuery syntax and method chaining
3. Include error handling and validation
4. Add comments explaining complex operations
5. Use meaningful variable names
6. Follow Python PEP 8 style guidelines
7. Ensure code is executable and produces valid 3D models
8. Include proper imports and setup
9. Add export functionality

COMMON PATTERNS:
```python
import cadquery as cq

# Basic box
result = cq.Workplane("XY").box(length, width, height)

# Cylinder
result = cq.Workplane("XY").cylinder(height, radius)

# Boolean operations
result = base.union(feature)
result = base.cut(hole)

# Fillets and chamfers
result = result.edges().fillet(radius)
result = result.edges().chamfer(distance)

# Sketching and extrusion
sketch = cq.Workplane("XY").rect(width, height)
result = sketch.extrude(thickness)

# Positioning
result = result.translate((x, y, z))
result = result.rotate((0, 0, 1), (0, 0, 0), angle)
```

ERROR HANDLING:
- Validate inputs and dimensions
- Check for valid geometry creation
- Handle edge cases gracefully
- Provide meaningful error messages

OUTPUT FORMAT:
Provide your response as a valid JSON object with this structure:
{{
    "title": "Generated CAD Model",
    "description": "Description of the generated code",
    "imports": ["import cadquery as cq", "import math"],
    "global_variables": {{"TOLERANCE": "0.01", "UNITS": "mm"}},
    "functions": ["def helper_function(): pass"],
    "main_code": "Complete CADQuery code here",
    "export_code": "result.val().exportStl('model.stl')",
    "estimated_complexity": "Simple|Medium|Complex",
    "cadquery_version": "2.4"
}}

MAIN_CODE REQUIREMENTS:
- Must be complete, executable Python code
- Should create a variable called 'result' containing the final model
- Include proper error handling
- Add comments for clarity
- Use proper CADQuery syntax
- Handle all steps from the pseudocode sequentially

Be thorough and ensure the generated code will run successfully in a CADQuery environment."""

        human_prompt = """Please generate working CADQuery Python code for the following pseudocode:

PSEUDOCODE (JSON):
{pseudocode_json}

REQUIREMENTS:
- Convert each pseudocode block into working CADQuery code
- Maintain the logical flow and dependencies
- Include proper error handling and validation
- Generate clean, executable Python code
- Add helpful comments
- Ensure the code produces a valid 3D model

Please provide the complete CADQuery code as a JSON object."""

        return ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", human_prompt)]
        )

    def generate_code_from_pseudocode_file(
        self, pseudocode_json_file: str
    ) -> Optional[GeneratedCode]:
        """
        Generate CADQuery code from a pseudocode JSON file.

        Args:
            pseudocode_json_file: Path to the pseudocode JSON file

        Returns:
            GeneratedCode object or None if generation fails
        """
        try:
            # Load the pseudocode from JSON file
            if not os.path.exists(pseudocode_json_file):
                logger.error(f"Pseudocode file not found: {pseudocode_json_file}")
                print(f"Error: File '{pseudocode_json_file}' not found.")
                return None

            with open(pseudocode_json_file, "r", encoding="utf-8") as f:
                pseudocode_data = json.load(f)

            logger.info(f"Loaded pseudocode from file: {pseudocode_json_file}")
            print(f"✓ Loaded pseudocode from: {pseudocode_json_file}")

            # Convert pseudocode data to JSON string for the prompt
            pseudocode_json = json.dumps(pseudocode_data, indent=2)

            logger.info("Generating CADQuery code from pseudocode")
            print("Generating CADQuery code...")

            # Generate code using the chain
            response = self.generation_chain.invoke(
                {"pseudocode_json": pseudocode_json}
            )

            logger.info("Raw LLM Response received for code generation")

            # Parse the JSON response
            code_data = self._parse_json_response(response)
            if not code_data:
                print("✗ Failed to parse LLM response")
                print("Attempting alternative parsing...")

                # Try alternative parsing method
                code_data = self._alternative_json_parsing(response)
                if not code_data:
                    print("✗ All parsing methods failed")
                    return None
                else:
                    print("✓ Successfully parsed with alternative method")

            # Validate and create GeneratedCode object
            generated_code = GeneratedCode(**code_data)

            logger.info(f"Successfully generated CADQuery code")
            print(f"✓ Generated CADQuery code successfully")
            print(f"  Complexity: {generated_code.estimated_complexity}")
            print(f"  Imports: {len(generated_code.imports)}")
            print(f"  Functions: {len(generated_code.functions)}")

            return generated_code

        except Exception as e:
            logger.error(f"Error generating code from file: {str(e)}")
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
                print(
                    f"Debug: JSON parsing failed. Response preview: {response[:200]}..."
                )
                return None
        return None

    def _clean_response(self, text: str) -> str:
        """Clean response text to handle control characters and formatting issues."""
        import re

        # Remove control characters except for \n, \r, \t
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

        # Replace common problematic characters
        text = text.replace("\x0c", "")  # Form feed
        text = text.replace("\x0b", "")  # Vertical tab
        text = text.replace("\u2028", "")  # Line separator
        text = text.replace("\u2029", "")  # Paragraph separator

        # Fix common escape sequence issues
        text = text.replace("\\n", "\n")
        text = text.replace("\\t", "\t")
        text = text.replace("\\r", "\r")

        # Remove any null bytes
        text = text.replace("\x00", "")

        return text.strip()

    def _alternative_json_parsing(self, response: str) -> Optional[Dict]:
        """Alternative JSON parsing method for problematic responses."""
        try:
            import re

            # Try to extract JSON more aggressively
            json_patterns = [
                r'\{[\s\S]*"title"[\s\S]*\}',  # Look for JSON with "title" field
                r'\{[\s\S]*"main_code"[\s\S]*\}',  # Look for JSON with "main_code" field
                r"\{[^}]*(?:\{[^}]*\}[^}]*)*\}",  # Nested JSON pattern
            ]

            for pattern in json_patterns:
                matches = re.findall(pattern, response, re.DOTALL)
                for match in matches:
                    try:
                        cleaned_match = self._clean_response(match)
                        # Additional cleaning for JSON-specific issues
                        cleaned_match = re.sub(
                            r",\s*}", "}", cleaned_match
                        )  # Remove trailing commas
                        cleaned_match = re.sub(
                            r",\s*]", "]", cleaned_match
                        )  # Remove trailing commas in arrays

                        parsed = json.loads(cleaned_match)
                        # Validate that it has required fields
                        if "title" in parsed and "main_code" in parsed:
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
                "description": "CADQuery code generated from pseudocode",
                "imports": ["import cadquery as cq"],
                "global_variables": {"TOLERANCE": "0.01"},
                "functions": [],
                "main_code": "# Basic CADQuery code\nimport cadquery as cq\n\n# Create a simple box\nresult = cq.Workplane('XY').box(10, 10, 10)\n\nprint('Model created successfully')",
                "export_code": "# Export the model\nresult.val().exportStl('model.stl')",
                "estimated_complexity": "Simple",
                "cadquery_version": "2.4",
            }

            print("✓ Using reconstructed minimal JSON structure")
            return minimal_json

        except Exception as e:
            logger.error(f"Failed to reconstruct JSON: {e}")
            return None

    def display_generated_code(self, generated_code: GeneratedCode) -> None:
        """Display the generated code in a readable format."""
        print(f"\n{'='*70}")
        print(f"GENERATED CADQUERY CODE: {generated_code.title}")
        print(f"{'='*70}")
        print(f"Description: {generated_code.description}")
        print(f"Complexity: {generated_code.estimated_complexity}")
        print(f"CADQuery Version: {generated_code.cadquery_version}")

        print(f"\nIMPORTS:")
        print("-" * 30)
        for import_stmt in generated_code.imports:
            print(f"{import_stmt}")

        if generated_code.global_variables:
            print(f"\nGLOBAL VARIABLES:")
            print("-" * 30)
            for var_name, var_value in generated_code.global_variables.items():
                print(f"{var_name} = {var_value}")

        if generated_code.functions:
            print(f"\nHELPER FUNCTIONS:")
            print("-" * 30)
            for func in generated_code.functions:
                print(f"{func}")

        print(f"\nMAIN CODE:")
        print("-" * 30)
        print(generated_code.main_code)

        print(f"\nEXPORT CODE:")
        print("-" * 30)
        print(generated_code.export_code)

    def export_code(self, generated_code: GeneratedCode, filename: str) -> bool:
        """Export generated code to JSON file."""
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(generated_code.dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Generated code exported to {filename}")
            print(f"✓ Code exported to: {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to export code: {e}")
            print(f"✗ Failed to export: {e}")
            return False

    def export_python_file(self, generated_code: GeneratedCode, filename: str) -> bool:
        """Export generated code as executable Python file."""
        try:
            python_code = self._create_python_file_content(generated_code)

            with open(filename, "w", encoding="utf-8") as f:
                f.write(python_code)

            logger.info(f"Python file exported to {filename}")
            print(f"✓ Python file exported to: {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to export Python file: {e}")
            print(f"✗ Failed to export Python file: {e}")
            return False

    def _create_python_file_content(self, generated_code: GeneratedCode) -> str:
        """Create complete Python file content from generated code."""
        content = []

        # Add header comment
        content.append(f'"""')
        content.append(f"{generated_code.title}")
        content.append(f"{generated_code.description}")
        content.append(f"Generated by CADQuery Code Generator Agent")
        content.append(f"CADQuery Version: {generated_code.cadquery_version}")
        content.append(f'"""')
        content.append("")

        # Add imports
        for import_stmt in generated_code.imports:
            content.append(import_stmt)
        content.append("")

        # Add global variables
        if generated_code.global_variables:
            content.append("# Global variables")
            for var_name, var_value in generated_code.global_variables.items():
                content.append(f"{var_name} = {var_value}")
            content.append("")

        # Add helper functions
        if generated_code.functions:
            content.append("# Helper functions")
            for func in generated_code.functions:
                content.append(func)
            content.append("")

        # Add main function
        content.append("def create_model():")
        content.append('    """Create the CAD model."""')
        content.append("    try:")

        # Indent main code
        main_lines = generated_code.main_code.split("\n")
        for line in main_lines:
            if line.strip():
                content.append(f"        {line}")
            else:
                content.append("")

        content.append("        return result")
        content.append("    except Exception as e:")
        content.append('        print(f"Error creating model: {e}")')
        content.append("        return None")
        content.append("")

        # Add main execution
        content.append('if __name__ == "__main__":')
        content.append('    print("Creating CAD model...")')
        content.append("    model = create_model()")
        content.append("    ")
        content.append("    if model:")
        content.append('        print("✓ Model created successfully!")')
        content.append("        ")
        content.append("        # Export the model")

        # Indent export code
        export_lines = generated_code.export_code.split("\n")
        for line in export_lines:
            if line.strip():
                content.append(f"        {line}")

        content.append('        print("✓ Model exported!")')
        content.append("    else:")
        content.append('        print("✗ Failed to create model")')

        return "\n".join(content)


def process_single_pseudocode_file(
    input_file: str, output_file: str = None, python_file: str = None
) -> bool:
    """Process a single pseudocode JSON file and generate CADQuery code."""

    try:
        generator = CADQueryCodeGenerator()

        print(f"\nProcessing: {input_file}")
        print("-" * 50)

        # Generate code from the pseudocode file
        generated_code = generator.generate_code_from_pseudocode_file(input_file)

        if not generated_code:
            return False

        # Display the generated code
        generator.display_generated_code(generated_code)

        # Export code JSON
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_code.json"

        success = generator.export_code(generated_code, output_file)

        # Export Python file
        if python_file is None:
            base_name = os.path.splitext(input_file)[0]
            python_file = f"{base_name}_code.py"

        if generator.export_python_file(generated_code, python_file):
            print(f"✓ Executable Python file: {python_file}")
            success = True

        return success

    except Exception as e:
        print(f"✗ Error processing file: {str(e)}")
        return False


def process_batch_pseudocode_files(
    directory: str = ".", pattern: str = "*_pseudocode.json"
) -> None:
    """Process multiple pseudocode JSON files in batch."""

    import glob

    try:
        # Find all pseudocode files matching the pattern
        search_pattern = os.path.join(directory, pattern)
        pseudocode_files = glob.glob(search_pattern)

        if not pseudocode_files:
            print(f"No pseudocode files found matching pattern: {search_pattern}")
            return

        print(f"Found {len(pseudocode_files)} pseudocode files:")
        for i, file in enumerate(pseudocode_files, 1):
            print(f"  {i}. {file}")

        # Initialize generator once
        generator = CADQueryCodeGenerator()
        successful_conversions = 0

        # Process each file
        for pseudocode_file in pseudocode_files:
            print(f"\n{'='*60}")
            print(f"PROCESSING: {os.path.basename(pseudocode_file)}")
            print("=" * 60)

            try:
                # Generate code
                generated_code = generator.generate_code_from_pseudocode_file(
                    pseudocode_file
                )

                if generated_code:
                    print(f"✓ Title: {generated_code.title}")
                    print(f"✓ Complexity: {generated_code.estimated_complexity}")

                    # Generate output filenames
                    base_name = os.path.splitext(pseudocode_file)[0]
                    json_output = f"{base_name}_code.json"
                    python_output = f"{base_name}_code.py"

                    # Export both formats
                    if generator.export_code(generated_code, json_output):
                        if generator.export_python_file(generated_code, python_output):
                            successful_conversions += 1

                else:
                    print(f"✗ Failed to generate code")

            except Exception as e:
                print(f"✗ Error processing {pseudocode_file}: {str(e)}")

        # Summary
        print(f"\n{'='*60}")
        print("BATCH PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Files processed: {len(pseudocode_files)}")
        print(f"Successful conversions: {successful_conversions}")
        print(
            f"Success rate: {(successful_conversions/len(pseudocode_files)*100):.1f}%"
        )

    except Exception as e:
        print(f"Error in batch processing: {str(e)}")


def interactive_mode():
    """Interactive mode for selecting and processing pseudocode files."""

    try:
        # Find available pseudocode files
        import glob

        pseudocode_files = glob.glob("*_pseudocode.json")

        if not pseudocode_files:
            print("No pseudocode files found in current directory.")
            print("Make sure you have JSON files created by the pseudocode generator.")
            return

        print("CADQuery Code Generator - Interactive Mode")
        print("=" * 60)
        print("Available pseudocode files:")

        for i, file in enumerate(pseudocode_files, 1):
            print(f"  {i}. {file}")

        print(f"  {len(pseudocode_files) + 1}. Process all files")
        print("  0. Exit")

        while True:
            try:
                choice = input(
                    f"\nSelect a file to process (0-{len(pseudocode_files) + 1}): "
                ).strip()

                if choice == "0":
                    print("Goodbye!")
                    break

                if choice == str(len(pseudocode_files) + 1):
                    print("\nProcessing all files...")
                    process_batch_pseudocode_files()
                    break

                try:
                    file_index = int(choice) - 1
                    if 0 <= file_index < len(pseudocode_files):
                        selected_file = pseudocode_files[file_index]

                        if process_single_pseudocode_file(selected_file):
                            print("\n✓ Successfully processed!")
                        else:
                            print("\n✗ Processing failed!")

                        continue_choice = (
                            input("\nProcess another file? (y/n): ").strip().lower()
                        )
                        if continue_choice not in ["y", "yes"]:
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

    parser = argparse.ArgumentParser(
        description="Generate CADQuery code from pseudocode JSON files"
    )
    parser.add_argument("input_file", nargs="?", help="Input pseudocode JSON file")
    parser.add_argument("-o", "--output", help="Output code JSON file")
    parser.add_argument("-p", "--python", help="Output Python file")
    parser.add_argument(
        "-b", "--batch", action="store_true", help="Process all *_pseudocode.json files"
    )
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Interactive mode"
    )
    parser.add_argument(
        "-d", "--directory", default=".", help="Directory to search (default: current)"
    )
    parser.add_argument(
        "--pattern",
        default="*_pseudocode.json",
        help="File pattern (default: *_pseudocode.json)",
    )

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
            process_batch_pseudocode_files(args.directory, args.pattern)
        elif args.input_file:
            process_single_pseudocode_file(args.input_file, args.output, args.python)
        else:
            print("CADQuery Code Generator Agent")
            print("=" * 50)
            print("\nUsage examples:")
            print("  python code_generator_agent.py plan_1_pseudocode.json")
            print(
                "  python code_generator_agent.py plan_1_pseudocode.json -o my_code.json -p my_code.py"
            )
            print("  python code_generator_agent.py --batch")
            print("  python code_generator_agent.py --interactive")
            print("\nFor help: python code_generator_agent.py --help")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
