"""
CADQuery Code Generator Agent
Converts pseudocode from the pseudocode generator into executable CADQuery Python code
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import re

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


class GeneratedCode(BaseModel):
    """Represents generated CADQuery code"""
    plan_id: str
    title: str
    description: str
    code: str
    imports: List[str]
    helper_functions: Optional[str] = None
    main_code: str
    export_code: str
    full_script: str


class CADQueryCodeGenerator:
    """Generates executable CADQuery code from pseudocode"""

    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        # Initialize Gemini
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=self.api_key,
            temperature=0.1,  # Low temperature for consistent code
            max_tokens=8192
        )

        self.output_parser = StrOutputParser()
        self.prompt = self._create_prompt()
        self.chain = self.prompt | self.llm | self.output_parser

    def _create_prompt(self) -> ChatPromptTemplate:
        """Create prompt for code generation"""

        system_prompt = """You are an expert CADQuery code generator. Convert pseudocode into working CADQuery Python code.

IMPORTANT CADQUERY RULES:

1. WORKPLANE CREATION:
   - Always start with: result = cq.Workplane("XY")
   - For new sketches on faces: result.faces(">Z").workplane()

2. VARIABLE MANAGEMENT:
   - Main object is always "result"
   - Don't create undefined variables
   - Chain operations when possible: result = result.circle(10).extrude(5)

3. SELECTORS:
   - Faces: ">Z" (top), "<Z" (bottom), "|Z" (vertical), ">X", "<X", "|X", etc.
   - Edges: "|Z" (vertical edges), ">Y" (edges in +Y direction), etc.
   - Use .faces(), .edges(), .vertices() before operations that need selection

4. COMMON OPERATIONS:
   - Sketch: .circle(radius), .rect(width, height), .polygon(npoints, radius)
   - 3D: .extrude(distance), .revolve(angle), .loft(), .sweep()
   - Boolean: .cut(object), .union(object), .intersect(object)
   - Modify: .fillet(radius), .chamfer(distance), .shell(thickness)
   - Holes: .hole(diameter, depth), .cboreHole(), .cskHole()

5. SPECIAL CASES:
   - For holes through all: .faces(">Z").workplane().circle(radius).cutThruAll()
   - For patterns: use .polarArray() or .rectArray()
   - For complex profiles: create separate objects and use boolean operations

6. GEAR PROFILES:
   Since CADQuery doesn't have built-in gear generation, create a helper function:
   ```python
   def create_gear_teeth(num_teeth, module, pressure_angle, thickness):
       # Simplified gear tooth profile
       # In real implementation, use involute curve equations
       tooth_angle = 360 / num_teeth
       teeth = None
       for i in range(num_teeth):
           # Create simplified tooth shape
           tooth = (cq.Workplane("XY")
                   .moveTo(radius, 0)
                   .lineTo(outer_radius, tooth_width/2)
                   .lineTo(outer_radius, -tooth_width/2)
                   .close()
                   .extrude(thickness)
                   .rotate((0,0,0), (0,0,1), i * tooth_angle))
           teeth = teeth.union(tooth) if teeth else tooth
       return teeth
   ```

OUTPUT FORMAT:
Generate a complete Python script with:
1. Imports
2. Helper functions (if needed)
3. Main construction code
4. Export statement

The code must be syntactically correct and executable."""

        human_prompt = """Convert this pseudocode into working CADQuery code:

PSEUDOCODE JSON:
{pseudocode_json}

Generate a complete, executable Python script that:
1. Implements all steps from the pseudocode
2. Handles complex operations (like gear profiles) properly
3. Uses correct CADQuery syntax
4. Includes proper selectors and parameters
5. Can be run directly to generate the CAD model

Return ONLY the Python code, no explanations."""

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])

    def generate_from_pseudocode_file(self, pseudocode_file: str) -> Optional[GeneratedCode]:
        """Generate code from a pseudocode JSON file"""
        try:
            # Load pseudocode
            with open(pseudocode_file, 'r') as f:
                pseudocode_data = json.load(f)

            logger.info(f"Loaded pseudocode: {pseudocode_data.get('title', 'Unknown')}")

            # Generate code
            code_response = self.chain.invoke({
                "pseudocode_json": json.dumps(pseudocode_data, indent=2)
            })

            # Parse and structure the code
            generated_code = self._structure_generated_code(code_response, pseudocode_data)

            if generated_code:
                logger.info("Successfully generated CADQuery code")
                return generated_code
            else:
                logger.error("Failed to structure generated code")
                return None

        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            return None

    def _structure_generated_code(self, code_response: str, pseudocode_data: Dict) -> Optional[GeneratedCode]:
        """Structure the generated code into components"""
        try:
            # Clean the response
            code = code_response.strip()

            # Remove markdown code blocks if present
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0]
            elif "```" in code:
                code = code.split("```")[1].split("```")[0]

            # Extract components
            lines = code.split('\n')

            # Find imports
            imports = []
            import_end = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('import') or line.strip().startswith('from'):
                    imports.append(line.strip())
                    import_end = i + 1
                elif line.strip() and not line.strip().startswith('#'):
                    break

            # Find helper functions
            helper_functions = []
            main_start = import_end
            for i in range(import_end, len(lines)):
                if lines[i].strip().startswith('def '):
                    # Find the end of the function
                    j = i + 1
                    while j < len(lines) and (lines[j].startswith(' ') or not lines[j].strip()):
                        j += 1
                    helper_functions.extend(lines[i:j])
                    main_start = j

            # Find main code and export
            main_code_lines = []
            export_lines = []
            for i in range(main_start, len(lines)):
                line = lines[i]
                if 'export' in line.lower() or 'save' in line.lower():
                    export_lines.extend(lines[i:])
                    break
                else:
                    main_code_lines.append(line)

            # Ensure we have required imports
            if not any('cadquery' in imp for imp in imports):
                imports.insert(0, 'import cadquery as cq')

            # Build the complete script
            full_script_parts = []

            # Header
            full_script_parts.append(f'"""\n{pseudocode_data.get("title", "CAD Model")}\n{pseudocode_data.get("description", "Generated by CADQuery Code Generator")}\n"""\n')

            # Imports
            full_script_parts.extend(imports)
            full_script_parts.append('')  # Empty line

            # Helper functions
            if helper_functions:
                full_script_parts.extend(helper_functions)
                full_script_parts.append('')  # Empty line

            # Main code
            full_script_parts.append('# Main construction')
            full_script_parts.extend(main_code_lines)
            full_script_parts.append('')  # Empty line

            # Export
            if not export_lines:
                export_lines = [
                    '# Export the model',
                    'result.exportStep("output.step")',
                    'print("Model exported to output.step")'
                ]
            full_script_parts.extend(export_lines)

            # Create GeneratedCode object
            return GeneratedCode(
                plan_id=pseudocode_data.get('plan_id', 'unknown'),
                title=pseudocode_data.get('title', 'CAD Model'),
                description=pseudocode_data.get('description', ''),
                code=code,
                imports=imports,
                helper_functions='\n'.join(helper_functions) if helper_functions else None,
                main_code='\n'.join(main_code_lines),
                export_code='\n'.join(export_lines),
                full_script='\n'.join(full_script_parts)
            )

        except Exception as e:
            logger.error(f"Error structuring code: {str(e)}")
            return None

    def validate_code(self, code: str) -> Tuple[bool, List[str]]:
        """Basic validation of generated code"""
        issues = []

        # Check for syntax errors
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")

        # Check for common issues
        lines = code.split('\n')

        # Check for undefined variables
        defined_vars = {'result', 'cq'}
        for line in lines:
            if '=' in line:
                var_name = line.split('=')[0].strip()
                if var_name:
                    defined_vars.add(var_name)

            # Check for usage of undefined variables
            for match in re.findall(r'\b(\w+)\s*=\s*\1\.', line):
                if match not in defined_vars:
                    issues.append(f"Undefined variable: {match}")

        # Check for required imports
        if not any('cadquery' in line for line in lines):
            issues.append("Missing cadquery import")

        # Check for at least one operation
        if not any(op in code for op in ['.circle', '.rect', '.box', '.cylinder', '.extrude']):
            issues.append("No CADQuery operations found")

        return len(issues) == 0, issues

    def save_code(self, generated_code: GeneratedCode, output_dir: str = "generated_code"):
        """Save generated code to file"""
        os.makedirs(output_dir, exist_ok=True)

        # Save the Python file
        py_file = os.path.join(output_dir, f"{generated_code.plan_id}_cadquery.py")
        with open(py_file, 'w') as f:
            f.write(generated_code.full_script)

        # Validate the code
        is_valid, issues = self.validate_code(generated_code.full_script)

        # Save validation report
        report_file = os.path.join(output_dir, f"{generated_code.plan_id}_validation.txt")
        with open(report_file, 'w') as f:
            f.write(f"Code Validation Report\n")
            f.write(f"="*50 + "\n")
            f.write(f"Title: {generated_code.title}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Valid: {is_valid}\n")
            if issues:
                f.write(f"\nIssues found:\n")
                for issue in issues:
                    f.write(f"  - {issue}\n")
            else:
                f.write(f"\nNo issues found. Code appears to be valid.\n")

        print(f"\nSaved files:")
        print(f"  Python code: {py_file}")
        print(f"  Validation: {report_file}")

        if is_valid:
            print("✓ Code validation passed!")
        else:
            print("⚠ Code has validation issues. Check the validation report.")

        return py_file, is_valid

    def display_code(self, generated_code: GeneratedCode):
        """Display generated code with syntax highlighting (if available)"""
        print(f"\n{'='*70}")
        print(f"GENERATED CADQUERY CODE: {generated_code.title}")
        print(f"{'='*70}")

        # Try to use syntax highlighting if available
        try:
            from pygments import highlight
            from pygments.lexers import PythonLexer
            from pygments.formatters import TerminalFormatter

            highlighted = highlight(generated_code.full_script, PythonLexer(), TerminalFormatter())
            print(highlighted)
        except ImportError:
            # Fallback to plain text
            print(generated_code.full_script)


def main():
    """Process pseudocode files and generate CADQuery code"""
    import sys
    import glob

    generator = CADQueryCodeGenerator()

    # Check if specific file provided
    if len(sys.argv) > 1:
        pseudocode_file = sys.argv[1]
        if os.path.exists(pseudocode_file):
            print(f"Processing: {pseudocode_file}")

            generated_code = generator.generate_from_pseudocode_file(pseudocode_file)
            if generated_code:
                generator.display_code(generated_code)
                generator.save_code(generated_code)
                print("\n✓ Code generation successful!")
            else:
                print("\n✗ Code generation failed!")
        else:
            print(f"File not found: {pseudocode_file}")
        return

    # Look for pseudocode files
    pseudocode_files = []

    # Check pseudocode_output directory first
    if os.path.exists("pseudocode_output"):
        pseudocode_files.extend(glob.glob("pseudocode_output/*_pseudocode.json"))

    # Also check current directory
    pseudocode_files.extend(glob.glob("*_pseudocode.json"))

    # Remove duplicates
    pseudocode_files = list(set(pseudocode_files))

    if not pseudocode_files:
        print("No pseudocode files found!")
        print("Make sure you've run the pseudocode generator first.")
        return

    print(f"Found {len(pseudocode_files)} pseudocode file(s):")
    for i, file in enumerate(pseudocode_files):
        print(f"  {i+1}. {file}")

    # Process all or let user choose
    if len(pseudocode_files) == 1:
        choice = 0
    else:
        print(f"\nEnter number to process (1-{len(pseudocode_files)}) or 'a' for all: ")
        user_input = input().strip()

        if user_input.lower() == 'a':
            # Process all files
            for file in pseudocode_files:
                print(f"\n{'='*50}")
                print(f"Processing: {file}")
                print("="*50)

                generated_code = generator.generate_from_pseudocode_file(file)
                if generated_code:
                    generator.display_code(generated_code)
                    generator.save_code(generated_code)
                    print("✓ Success!")
                else:
                    print("✗ Failed!")
            return
        else:
            try:
                choice = int(user_input) - 1
            except:
                print("Invalid choice")
                return

    if 0 <= choice < len(pseudocode_files):
        file = pseudocode_files[choice]
        print(f"\nProcessing: {file}")

        generated_code = generator.generate_from_pseudocode_file(file)
        if generated_code:
            generator.display_code(generated_code)
            generator.save_code(generated_code)
            print("\n✓ Code generation successful!")
            print("\nYou can now run the generated code with:")
            print(f"  python {os.path.join('generated_code', generated_code.plan_id + '_cadquery.py')}")
        else:
            print("\n✗ Code generation failed!")


if __name__ == "__main__":
    main()