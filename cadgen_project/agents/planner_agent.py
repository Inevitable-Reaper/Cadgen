"""
Advanced Planner Agent for CADQuery Code Generation
Supports multiple manufacturing processes: CNC, 3D printing, laser cutting, etc.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ManufacturingProcess(str, Enum):
    """Supported manufacturing processes"""
    CNC_MILLING = "cnc_milling"
    THREE_D_PRINTING = "3d_printing"
    LASER_CUTTING = "laser_cutting"
    SHEET_METAL = "sheet_metal"
    INJECTION_MOLDING = "injection_molding"
    GENERAL_CAD = "general_cad"

    @classmethod
    def _missing_(cls, value):
        """Handle values that don't match exactly"""
        if isinstance(value, str):
            # Normalize the value
            normalized = value.lower().strip()
            # Remove common variations
            normalized = normalized.replace(' ', '_').replace('-', '_')
            # Map common variations
            mappings = {
                '3d_printing': cls.THREE_D_PRINTING,
                '3d': cls.THREE_D_PRINTING,
                'printing': cls.THREE_D_PRINTING,
                'additive': cls.THREE_D_PRINTING,
                'fdm': cls.THREE_D_PRINTING,
                'cnc': cls.CNC_MILLING,
                'milling': cls.CNC_MILLING,
                'machining': cls.CNC_MILLING,
                'laser': cls.LASER_CUTTING,
                'cutting': cls.LASER_CUTTING,
                'laser_cut': cls.LASER_CUTTING,
                'sheet': cls.SHEET_METAL,
                'metal': cls.SHEET_METAL,
                'bending': cls.SHEET_METAL,
                'injection': cls.INJECTION_MOLDING,
                'molding': cls.INJECTION_MOLDING,
                'casting': cls.INJECTION_MOLDING,
            }
            # Try exact match first
            for member in cls:
                if member.value == normalized:
                    return member
            # Try mapping
            for key, member in mappings.items():
                if key in normalized:
                    return member
            # Default to general CAD for unknown processes
            return cls.GENERAL_CAD
        return cls.GENERAL_CAD


class GeometryOperation(BaseModel):
    """Represents a geometry operation in CADQuery"""
    operation_type: str = Field(default="geometry", description="Type of CADQuery operation")
    method_name: str = Field(description="CADQuery method to call")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the operation")
    constraints: Optional[Dict[str, Any]] = Field(default=None, description="Manufacturing constraints")
    workplane: Optional[str] = Field(default="XY", description="Workplane reference")
    axis_reference: Optional[str] = Field(default="N/A", description="Axis orientation reference")


class PlanningStep(BaseModel):
    """Enhanced planning step with manufacturing considerations"""
    step_number: int = Field(description="Sequential step number")
    phase: str = Field(description="Phase: sketch/3d_creation/modification/assembly/finishing")
    operation: GeometryOperation = Field(description="CADQuery operation details")
    dependencies: List[int] = Field(default_factory=list, description="Steps this depends on")
    description: str = Field(description="Human-readable description")
    manufacturing_notes: Optional[str] = Field(default=None, description="Process-specific notes")
    geometry_notes: Optional[str] = Field(default=None, description="Shape and positioning details")

    @field_validator('dependencies')
    @classmethod
    def validate_dependencies(cls, v, info):
        if info.data.get('step_number') and any(dep >= info.data['step_number'] for dep in v):
            raise ValueError("Dependencies must reference earlier steps")
        return v

    @model_validator(mode='before')
    @classmethod
    def handle_operation_format(cls, data):
        """Handle different operation formats from LLM response"""
        if isinstance(data, dict) and 'operation' in data:
            operation = data['operation']

            # Handle case where operation has 'method' instead of 'method_name'
            if isinstance(operation, dict):
                if 'method' in operation and 'method_name' not in operation:
                    operation['method_name'] = operation.pop('method')

                # Ensure required fields exist
                if 'operation_type' not in operation:
                    operation['operation_type'] = 'geometry'
                if 'parameters' not in operation:
                    operation['parameters'] = {}

                data['operation'] = operation

        return data


class MaterialSpecification(BaseModel):
    """Material specification for the part"""
    material_type: str = Field(description="Type of material")
    thickness: Optional[float] = Field(default=None, description="Material thickness in mm")
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Material properties")


class CADPlan(BaseModel):
    """Comprehensive CAD plan with manufacturing details"""
    plan_id: str = Field(description="Unique plan identifier")
    title: str = Field(description="Title of the CAD model")
    description: str = Field(description="Detailed description")
    manufacturing_process: ManufacturingProcess = Field(default=ManufacturingProcess.GENERAL_CAD, description="Primary manufacturing process")
    material: Optional[MaterialSpecification] = Field(default=None, description="Material specification")
    dimensions: Dict[str, float] = Field(description="Overall dimensions")
    tolerance: float = Field(default=0.1, description="General tolerance in mm")
    steps: List[PlanningStep] = Field(description="Ordered planning steps")
    post_processing: List[str] = Field(default_factory=list, description="Post-processing steps")
    estimated_complexity: str = Field(default="medium", description="Simple/Medium/Complex/Expert")
    complexity: Optional[str] = Field(default=None, description="Alternative complexity field")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    coordinate_system: Optional[Dict[str, Any]] = Field(default=None, description="Coordinate system info")

    @field_validator('manufacturing_process', mode='before')
    @classmethod
    def normalize_manufacturing_process(cls, v):
        """Normalize manufacturing process to enum"""
        if isinstance(v, str):
            # The enum's _missing_ method will handle the conversion
            return ManufacturingProcess(v)
        return v

    @field_validator('dimensions', mode='before')
    @classmethod
    def ensure_dimensions(cls, v):
        """Ensure dimensions dict has required keys"""
        if isinstance(v, dict):
            # Provide defaults for missing dimensions
            defaults = {"length": 0, "width": 0, "height": 0}
            return {**defaults, **v}
        return v

    @model_validator(mode='before')
    @classmethod
    def handle_complexity_fields(cls, data):
        """Handle different complexity field names"""
        if isinstance(data, dict):
            # Use 'complexity' if 'estimated_complexity' is missing
            if 'complexity' in data and 'estimated_complexity' not in data:
                data['estimated_complexity'] = data['complexity']
            elif 'estimated_complexity' not in data and 'complexity' not in data:
                data['estimated_complexity'] = 'medium'
        return data


class PlannerAgent:
    """
    Advanced Planner Agent for multi-process CAD planning
    """

    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        """Initialize the Planner Agent with Gemini 2.5 Flash"""
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        # Initialize Gemini 2.5 Flash LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self.api_key,
            temperature=0.2,  # Lower for consistent planning
            max_tokens=8192,  # Increased for complex plans
            timeout=90,
            max_retries=3
        )

        self.output_parser = StrOutputParser()
        self.planning_prompt = self._create_enhanced_prompt()
        self.planning_chain = self.planning_prompt | self.llm | self.output_parser

    def _create_enhanced_prompt(self) -> ChatPromptTemplate:
        """Create comprehensive planning prompt"""

        system_prompt = """You are an expert CAD designer specialized in creating detailed plans for CADQuery 3D modeling.

CADQUERY CORE OPERATIONS:
# Workplane Creation & Positioning
- Workplane(): Create workplane (default: XY plane at origin)
- Workplane("XY"/"XZ"/"YZ"): Specific plane
- Workplane(offset=z): Offset from current plane
- workplaneFromTagged("tag"): Use tagged workplane
- center(x, y): Move workplane center
- tag("name"): Tag current workplane for reference

# 2D Sketching Operations
- sketch(): Start 2D sketch on current workplane
- rect(width, height, centered=True): Rectangle
- circle(radius): Circle
- ellipse(major_axis, minor_axis): Ellipse
- polygon(sides, diameter): Regular polygon
- polyline(points): Connected line segments
- spline(points, tangents=None): Smooth curve
- arc(radius, start_angle, end_angle): Circular arc
- close(): Close current sketch

# 3D Creation Operations
- extrude(distance): Extrude sketch perpendicular to workplane
- revolve(angleDegrees, axisStart, axisEnd): Revolve around axis
- loft(ruled=False): Loft between multiple sketches
- sweep(path): Sweep profile along path
- box(length, width, height, centered=True): Create box
- cylinder(radius, height, centered=True): Create cylinder
- sphere(radius, centered=True): Create sphere
- cone(radius1, radius2, height, centered=True): Create cone
- torus(major_radius, minor_radius): Create torus

# Selection & Positioning
- faces(selector): Select faces (e.g., ">Z", "<X", "|Y")
- edges(selector): Select edges (e.g., "%CIRCLE", "#Z")
- vertices(): Select vertices
- translate(x, y, z): Move geometry
- rotate(axisStart, axisEnd, angleDegrees): Rotate around axis
- mirror(mirrorPlane): Mirror geometry

# Modification Operations
- fillet(radius): Round selected edges
- chamfer(length): Chamfer selected edges
- shell(thickness): Hollow out solid
- offset2D(distance): Offset 2D curves
- split(splittingObject): Split solid

# Boolean Operations
- union(otherObject): Combine solids
- cut(otherObject): Subtract solid
- intersect(otherObject): Keep intersection only

# Feature Operations
- hole(diameter, depth=None): Create cylindrical hole
- cboreHole(diameter, cbore_diameter, cbore_depth, depth=None): Counterbore hole
- cskHole(diameter, cskAngle, depth=None): Countersink hole

OUTPUT REQUIREMENTS:
Generate a complete JSON plan with this EXACT structure and field names:
{{
    "plan_id": "unique_identifier", 
    "title": "descriptive_title",
    "description": "detailed_description",
    "dimensions": {{
        "length": number,
        "width": number, 
        "height": number
    }},
    "coordinate_system": {{
        "primary_axis": "X/Y/Z",
        "orientation": "description"
    }},
    "steps": [
        {{
            "step_number": number,
            "phase": "workplane_setup/sketch/extrude/feature/boolean/finishing",
            "operation": {{
                "operation_type": "geometry",
                "method_name": "cadquery_method_name",
                "parameters": {{
                    "param_name": value
                }},
                "workplane": "XY/XZ/YZ or offset description",
                "axis_reference": "axis orientation if applicable"
            }},
            "dependencies": [step_numbers],
            "description": "clear step description",
            "geometry_notes": "shape and positioning details"
        }}
    ],
    "estimated_complexity": "simple/medium/complex/expert"
}}

CRITICAL REQUIREMENTS:
- Use EXACT field names: "method_name" NOT "method"
- Use EXACT field names: "estimated_complexity" NOT "complexity"
- Always include "operation_type": "geometry" in each operation
- Always specify workplane for each operation
- Include axis orientations for rotations and mirrors
- Use proper CADQuery method names and parameter syntax
- Ensure geometric feasibility of each step
- Plan logical construction sequence with proper dependencies"""

        human_prompt = """Create a comprehensive CAD modeling plan for the following 3D object:

        DESIGN SPECIFICATION: {user_input}

        ANALYSIS REQUIREMENTS:
        1. Analyze the geometric composition of the object
        2. Identify primary shapes, features, and their spatial relationships
        3. Determine optimal workplane orientations and coordinate system
        4. Plan construction sequence with proper axis references
        5. Break down complex features into simple CADQuery operations
        6. Ensure each step specifies workplane and axis orientations

        PLANNING REQUIREMENTS:
        - Minimum 4-6 steps for typical objects
        - Include workplane_setup, sketch, extrude, and feature phases
        - Specify coordinate system and axis orientations clearly
        - Use proper CADQuery method names and parameters
        - Include geometric dependencies between steps
        - Focus purely on 3D modeling geometry, not manufacturing

        Generate a complete JSON plan following the exact schema provided, with special attention to using the correct field names."""

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])

    def create_plan(self, user_input: str, process_hint: Optional[ManufacturingProcess] = None) -> Optional[CADPlan]:
        """
        Create a comprehensive CAD plan from user input

        Args:
            user_input: Natural language design specification
            process_hint: Optional manufacturing process hint

        Returns:
            CADPlan object or None if planning fails
        """
        try:
            # Add process hint to input if provided
            if process_hint:
                enhanced_input = f"{user_input} [Manufacturing process: {process_hint.value}]"
            else:
                enhanced_input = user_input

            logger.info(f"Creating plan for: {user_input}")

            # Generate plan
            response = self.planning_chain.invoke({"user_input": enhanced_input})

            # Parse response
            plan_data = self._robust_json_parse(response)
            if not plan_data:
                logger.error("Failed to parse plan response")
                return None

            # Generate unique plan ID if not present
            if "plan_id" not in plan_data:
                plan_data["plan_id"] = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Auto-detect manufacturing process from description if not specified
            if "manufacturing_process" not in plan_data or not plan_data["manufacturing_process"]:
                plan_data["manufacturing_process"] = self._detect_process(user_input)

            # Ensure all required fields exist with defaults
            self._ensure_required_fields(plan_data)

            # Validate and create plan
            plan = CADPlan(**plan_data)
            logger.info(f"Successfully created plan '{plan.title}' with {len(plan.steps)} steps")

            return plan

        except Exception as e:
            logger.error(f"Error creating plan: {str(e)}")
            logger.error(f"Plan data that failed validation: {plan_data if 'plan_data' in locals() else 'No plan data'}")
            return None

    def _ensure_required_fields(self, plan_data: Dict) -> None:
        """Ensure all required fields exist with sensible defaults"""
        # Ensure basic fields
        if "title" not in plan_data:
            plan_data["title"] = "CAD Model"
        if "description" not in plan_data:
            plan_data["description"] = "3D CAD model"
        if "dimensions" not in plan_data:
            plan_data["dimensions"] = {"length": 100, "width": 100, "height": 100}
        if "steps" not in plan_data:
            plan_data["steps"] = []
        if "estimated_complexity" not in plan_data:
            plan_data["estimated_complexity"] = "medium"

        # Fix steps if they exist
        for step in plan_data.get("steps", []):
            if "operation" in step:
                operation = step["operation"]
                # Ensure operation_type exists
                if "operation_type" not in operation:
                    operation["operation_type"] = "geometry"
                # Handle method vs method_name
                if "method" in operation and "method_name" not in operation:
                    operation["method_name"] = operation.pop("method")
                # Ensure parameters exist
                if "parameters" not in operation:
                    operation["parameters"] = {}
                # Ensure workplane exists
                if "workplane" not in operation:
                    operation["workplane"] = "XY"
                # Ensure axis_reference exists
                if "axis_reference" not in operation:
                    operation["axis_reference"] = "N/A"

    def _detect_process(self, text: str) -> str:
        """Auto-detect manufacturing process from text"""
        text_lower = text.lower()
        if any(word in text_lower for word in ['print', '3d', 'fdm', 'sla']):
            return "3d_printing"
        elif any(word in text_lower for word in ['laser', 'cut', 'engrave']):
            return "laser_cutting"
        elif any(word in text_lower for word in ['cnc', 'mill', 'machine']):
            return "cnc_milling"
        elif any(word in text_lower for word in ['sheet', 'bend', 'fold']):
            return "sheet_metal"
        elif any(word in text_lower for word in ['mold', 'inject', 'cast']):
            return "injection_molding"
        else:
            return "general_cad"

    def _robust_json_parse(self, response: str) -> Optional[Dict]:
        """Robust JSON parsing with multiple fallback strategies"""
        # Strategy 1: Direct parsing
        try:
            return json.loads(response)
        except:
            pass

        # Strategy 2: Extract from code blocks
        import re
        patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'\{[\s\S]*\}'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except:
                    continue

        # Strategy 3: Clean and retry
        try:
            cleaned = response.strip()
            if cleaned.startswith("```") and cleaned.endswith("```"):
                cleaned = cleaned[3:-3].strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
            return json.loads(cleaned)
        except:
            logger.error(f"All JSON parsing strategies failed for response: {response[:200]}...")
            return None

    def display_plan(self, plan: CADPlan) -> None:
        """Display plan in a formatted way"""
        print(f"\n{'='*80}")
        print(f"CAD PLAN: {plan.title}")
        print(f"{'='*80}")
        print(f"ID: {plan.plan_id}")
        print(f"Description: {plan.description}")
        print(f"Process: {plan.manufacturing_process.value}")
        print(f"Complexity: {plan.estimated_complexity}")
        print(f"Tolerance: ±{plan.tolerance}mm")

        if plan.material:
            print(f"\nMATERIAL:")
            print(f"  Type: {plan.material.material_type}")
            if plan.material.thickness:
                print(f"  Thickness: {plan.material.thickness}mm")

        print(f"\nDIMENSIONS:")
        for dim, value in plan.dimensions.items():
            print(f"  {dim}: {value}mm")

        print(f"\nMANUFACTURING STEPS:")
        print("-" * 60)

        current_phase = None
        for step in plan.steps:
            if step.phase != current_phase:
                current_phase = step.phase
                print(f"\n[{current_phase.upper()}]")

            print(f"\nStep {step.step_number}: {step.description}")
            print(f"  Operation: {step.operation.method_name}({step.operation.parameters})")
            if step.dependencies:
                print(f"  Depends on: {step.dependencies}")
            if step.manufacturing_notes:
                print(f"  Note: {step.manufacturing_notes}")

        if plan.post_processing:
            print(f"\nPOST-PROCESSING:")
            for pp in plan.post_processing:
                print(f"  - {pp}")

    def save_plan(self, plan: CADPlan, filepath: str) -> bool:
        """Save plan to JSON file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(plan.model_dump(), f, indent=2, ensure_ascii=False)
            logger.info(f"Plan saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save plan: {e}")
            return False


def main():
    """Test the planner agent with various examples"""
    # Initialize planner
    planner = PlannerAgent()

    # Test cases with natural language descriptions
    test_cases = [
        "Design a spur gear with 24 teeth and an approximate pitch diameter of 60 millimeters. The gear should follow standard geometric conventions, with the module automatically determined based on the specified pitch diameter and number of teeth. Use a standard pressure angle of 20 degrees to define the involute tooth profile. This should be a straight-cut spur gear, with no helical angles or bevels. At the center, include a circular bore hole with a diameter of 5 millimeters to allow for shaft insertion or mounting. The gear should have a face width of 8 millimeters. Apply a chamfer of approximately 0.5 millimeters to the outer edges to soften sharp transitions and improve model clarity. Include a clearance of about 0.2 millimeters where necessary to account for general fitting tolerances. The tooth geometry must accurately represent an involute profile or a close approximation sufficient for modeling purposes. The final output should be a clean, watertight solid 3D model suitable for downstream use in CAD workflows or 3D model exports, such as STL or CadQuery script output. Avoid using features or geometries that would complicate rendering or violate standard gear modeling practices."
    ]

    print("CADQuery Advanced Planner Agent - Test Suite")
    print("=" * 80)

    for i, description in enumerate(test_cases, 1):
        print(f"\nTest {i}: {description[:60]}...")
        print("-" * 60)

        try:
            # Create plan
            plan = planner.create_plan(description)

            if plan:
                # Display summary
                print(f"✓ Created plan: {plan.title}")
                print(f"  Process: {plan.manufacturing_process.value}")
                print(f"  Steps: {len(plan.steps)}")
                print(f"  Complexity: {plan.estimated_complexity}")

                # Save plan
                filename = f"plans/test_{i}_{plan.plan_id}.json"
                if planner.save_plan(plan, filename):
                    print(f"  Saved to: {filename}")

                # Optionally display full plan for first test
                if i == 1:
                    planner.display_plan(plan)
            else:
                print(f"✗ Failed to create plan")

        except Exception as e:
            print(f"✗ Error: {str(e)}")
            logger.error(f"Test {i} failed: {str(e)}", exc_info=True)

    print("\n" + "="*80)
    print("Test suite completed!")


if __name__ == "__main__":
    # Create plans directory if it doesn't exist
    os.makedirs("plans", exist_ok=True)
    main()