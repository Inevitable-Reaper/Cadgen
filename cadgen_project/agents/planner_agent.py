"""
Basic Planner Agent for CADQuery Code Generation
This agent breaks down user design specifications into structured steps for CAD model creation.
"""

import os
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import json
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlanningStep(BaseModel):
    """Represents a single step in the CAD model creation plan."""
    step_number: int = Field(description="Sequential step number")
    action: str = Field(description="The action to be performed")
    geometry_type: str = Field(description="Type of geometry (box, cylinder, sphere, etc.)")
    parameters: Dict = Field(description="Parameters for the geometry")
    dependencies: List[int] = Field(default=[], description="Steps this depends on")
    description: str = Field(description="Human-readable description of the step")


class CADPlan(BaseModel):
    """Complete plan for CAD model creation."""
    title: str = Field(description="Title of the CAD model")
    description: str = Field(description="Overall description of the model")
    steps: List[PlanningStep] = Field(description="Ordered list of planning steps")
    estimated_complexity: str = Field(description="Simple/Medium/Complex")
    materials_needed: List[str] = Field(default=[], description="Materials or components")


class PlannerAgent:
    """
    Planner Agent that converts user specifications into structured CAD plans.
    """

    def __init__(self):
        """Initialize the Planner Agent with Groq LLM."""
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        # Initialize Groq LLM
        self.llm = ChatGroq(
            api_key=self.api_key,
            model_name=os.getenv("GROQ_MODEL", "llama3.1-70b-versatile"),
            temperature=float(os.getenv("PLANNING_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("MAX_TOKENS", "2048")),
            max_retries=int(os.getenv("MAX_RETRIES", "3"))
        )

        # Initialize output parser
        self.output_parser = StrOutputParser()

        # Create the planning prompt template
        self.planning_prompt = self._create_planning_prompt()

        # Create the planning chain
        self.planning_chain = self.planning_prompt | self.llm | self.output_parser

    def _create_planning_prompt(self) -> ChatPromptTemplate:
        """Create the prompt template for planning."""

        system_prompt = """You are an expert CAD designer and planner specialized in creating structured plans for CADQuery code generation.

Your task is to analyze user design specifications and create a detailed, step-by-step plan for generating CAD models using CADQuery.

IMPORTANT GUIDELINES:
1. Break down complex designs into simple, sequential steps
2. Each step should create or modify ONE geometric element
3. Identify dependencies between steps clearly
4. Use standard CADQuery operations: box, cylinder, sphere, extrude, cut, fillet, chamfer, etc.
5. Specify precise parameters for each geometry
6. Consider manufacturing constraints and best practices

AVAILABLE CADQUERY OPERATIONS:
- Basic shapes: box(length, width, height), cylinder(radius, height), sphere(radius)
- Sketch operations: sketch(), rect(width, height), circle(radius), polygon()
- 3D operations: extrude(distance), revolve(), loft(), sweep()
- Boolean operations: union(), cut(), intersect()
- Modifications: fillet(radius), chamfer(distance)
- Positioning: translate(), rotate(), mirror()

OUTPUT FORMAT:
Provide your response as a valid JSON object matching this structure:
{{
    "title": "Name of the CAD model",
    "description": "Overall description of what will be created",
    "estimated_complexity": "Simple|Medium|Complex",
    "materials_needed": ["material1", "material2"],
    "steps": [
        {{
            "step_number": 1,
            "action": "create_base",
            "geometry_type": "box",
            "parameters": {{"length": 50, "width": 30, "height": 10}},
            "dependencies": [],
            "description": "Create the base platform"
        }}
    ]
}}

Be precise with measurements and ensure all steps are executable in CADQuery."""

        human_prompt = """Please create a detailed CAD plan for the following design specification:

USER SPECIFICATION: {user_input}

REQUIREMENTS:
- Break this down into clear, sequential steps
- Each step should be actionable in CADQuery
- Include specific measurements and parameters
- Identify any dependencies between steps
- Consider the manufacturing process

Please provide the complete plan as a JSON object."""

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])

    def create_plan(self, user_input: str) -> Optional[CADPlan]:
        """
        Create a structured CAD plan from user input.

        Args:
            user_input: User's design specification

        Returns:
            CADPlan object or None if planning fails
        """
        try:
            logger.info(f"Creating plan for: {user_input}")

            # Generate the plan using the chain
            response = self.planning_chain.invoke({"user_input": user_input})

            logger.info("Raw LLM Response received")
            logger.debug(f"Response: {response}")

            # Parse the JSON response
            plan_data = self._parse_json_response(response)
            if not plan_data:
                return None

            # Validate and create CADPlan object
            plan = CADPlan(**plan_data)

            logger.info(f"Successfully created plan with {len(plan.steps)} steps")
            return plan

        except Exception as e:
            logger.error(f"Error creating plan: {str(e)}")
            return None

    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse JSON response from LLM, handling potential formatting issues."""
        try:
            # Try direct JSON parsing first
            return json.loads(response)
        except json.JSONDecodeError:
            try:
                # Try to extract JSON from markdown code blocks
                if "```json" in response:
                    json_start = response.find("```json") + 7
                    json_end = response.find("```", json_start)
                    json_str = response[json_start:json_end].strip()
                    return json.loads(json_str)
                elif "```" in response:
                    json_start = response.find("```") + 3
                    json_end = response.find("```", json_start)
                    json_str = response[json_start:json_end].strip()
                    return json.loads(json_str)
                else:
                    # Try to find JSON-like content
                    start_idx = response.find("{")
                    end_idx = response.rfind("}") + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = response[start_idx:end_idx]
                        return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.debug(f"Problematic response: {response}")
                return None
        return None

    def display_plan(self, plan: CADPlan) -> None:
        """Display the plan in a readable format."""
        print(f"\n{'='*60}")
        print(f"CAD MODEL PLAN: {plan.title}")
        print(f"{'='*60}")
        print(f"Description: {plan.description}")
        print(f"Complexity: {plan.estimated_complexity}")
        if plan.materials_needed:
            print(f"Materials: {', '.join(plan.materials_needed)}")
        print(f"\nSTEPS ({len(plan.steps)} total):")
        print("-" * 40)

        for step in plan.steps:
            print(f"\nStep {step.step_number}: {step.action}")
            print(f"  Geometry: {step.geometry_type}")
            print(f"  Parameters: {step.parameters}")
            if step.dependencies:
                print(f"  Depends on: Steps {', '.join(map(str, step.dependencies))}")
            print(f"  Description: {step.description}")

    def export_plan(self, plan: CADPlan, filename: str) -> bool:
        """Export plan to JSON file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(plan.dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Plan exported to {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to export plan: {e}")
            return False


def main():
    """Example usage of the Planner Agent."""
    try:
        # Initialize the planner agent
        planner = PlannerAgent()

        # Example design specifications
        test_cases = [
            "Create a simple rectangular box with dimensions 100mm x 50mm x 20mm",
            "Design a cylindrical cup with a handle, 80mm diameter and 100mm height",
            "Make a simple bracket with mounting holes for attaching to a wall"
        ]

        print("CADQuery Planner Agent - Test Run")
        print("=" * 50)

        for i, specification in enumerate(test_cases, 1):
            print(f"\nTest Case {i}: {specification}")
            print("-" * 50)

            # Create the plan
            plan = planner.create_plan(specification)

            if plan:
                planner.display_plan(plan)
                # Optionally export the plan
                planner.export_plan(plan, f"plan_{i}.json")
            else:
                print("Failed to create plan for this specification.")

            print("\n" + "="*60)

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"Failed to run planner agent: {str(e)}")
        print("Please check your .env configuration and ensure GROQ_API_KEY is set.")
    finally:
        print("\nPlanner Agent execution completed.")
        logger.info("Main function execution finished")


if __name__ == "__main__":
    main()