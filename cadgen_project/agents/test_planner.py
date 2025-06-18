"""
Test script for the CADQuery Planner Agent
Run this to test your planner agent with different design specifications.
"""

from planner_agent import (PlannerAgent, CADPlan)
import os
from dotenv import load_dotenv


def test_planner_agent():
    """Test the planner agent with various design specifications."""

    # Load environment variables
    load_dotenv()

    # Check if API key is set
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY not found in environment variables.")
        print("Please set your Groq API key in the .env file.")
        return

    try:
        # Initialize planner
        print("Initializing Planner Agent...")
        planner = PlannerAgent()
        print("✓ Planner Agent initialized successfully!")

        # Test cases with increasing complexity
        test_cases = [
            {
                "name": "Simple Box",
                "spec": "Create a simple rectangular box with dimensions 100mm x 50mm x 20mm",
            },
            {
                "name": "Cylindrical Container",
                "spec": "Design a cylindrical container with lid, 60mm diameter and 80mm height",
            },
            {
                "name": "L-Bracket",
                "spec": "Create an L-shaped mounting bracket with holes for M6 bolts",
            },
            {
                "name": "Phone Stand",
                "spec": "Design a simple phone stand that can hold a phone at 60-degree angle",
            },
        ]

        print(f"\nRunning {len(test_cases)} test cases...")
        print("=" * 60)

        successful_plans = 0

        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTEST {i}: {test_case['name']}")
            print(f"Specification: {test_case['spec']}")
            print("-" * 40)

            try:
                # Create plan
                plan = planner.create_plan(test_case["spec"])

                if plan:
                    print("✓ Plan created successfully!")
                    planner.display_plan(plan)

                    # Export plan
                    filename = f"test_plan_{i}_{test_case['name'].lower().replace(' ', '_')}.json"
                    if planner.export_plan(plan, filename):
                        print(f"✓ Plan exported to {filename}")

                    successful_plans += 1
                else:
                    print("✗ Failed to create plan")

            except Exception as e:
                print(f"✗ Error during planning: {str(e)}")

            print("\n" + "=" * 60)

        # Summary
        print(f"\nSUMMARY:")
        print(f"Successful plans: {successful_plans}/{len(test_cases)}")
        print(f"Success rate: {(successful_plans/len(test_cases)*100):.1f}%")

        if successful_plans == len(test_cases):
            print("🎉 All tests passed! Your planner agent is working correctly.")
        elif successful_plans > 0:
            print("⚠️  Some tests passed. Check the failed cases for issues.")
        else:
            print("❌ No tests passed. Check your configuration and API key.")

    except Exception as e:
        print(f"❌ Failed to initialize planner agent: {str(e)}")
        print(
            "Check your .env configuration and ensure all dependencies are installed."
        )


def interactive_test():
    """Interactive test mode - enter your own specifications."""

    load_dotenv()

    try:
        planner = PlannerAgent()
        print("CADQuery Planner Agent - Interactive Mode")
        print("=" * 50)
        print("Enter design specifications and get structured CAD plans!")
        print("Type 'quit' to exit.\n")

        while True:
            try:
                user_input = input("Enter your design specification: ").strip()

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break

                if not user_input:
                    print("Please enter a valid specification.")
                    continue

                print("\nCreating plan...")
                plan = planner.create_plan(user_input)

                if plan:
                    planner.display_plan(plan)

                    # Ask if user wants to save
                    save = input("\nSave this plan to file? (y/n): ").strip().lower()
                    if save in ["y", "yes"]:
                        filename = input("Enter filename (without extension): ").strip()
                        if filename:
                            filename = f"{filename}.json"
                            if planner.export_plan(plan, filename):
                                print(f"✓ Plan saved to {filename}")
                else:
                    print(
                        "Failed to create plan. Please try a different specification."
                    )

                print("\n" + "-" * 50)

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")

    except Exception as e:
        print(f"Failed to start interactive mode: {str(e)}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_test()
    else:
        test_planner_agent()
