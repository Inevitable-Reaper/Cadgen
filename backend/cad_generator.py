"""
Ultra Simple CAD Generator
Usage: python generate_cad.py "description of your part"
"""

import sys
import os
import time
from ai_cad_system import CADQueryCrewAI, CADDesignRequest
from visualization_engine import STLPlotlyVisualizer


def generate_and_show_cad(description: str):
    """Generate CAD model from description and show with Plotly"""
    
    print(f"🚀 Generating: {description}")
    print("=" * 50)
    
    # Check API key
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not found in environment")
        print("💡 Add your Google AI API key to the .env file")
        return
    
    try:
        # Initialize systems
        print("🤖 Initializing AI agents system...")
        cad_system = CADQueryCrewAI()  # No fallback - always use AI agents
        visualizer = STLPlotlyVisualizer()
        
        # Generate code
        print("✍️  Generating CADQuery code...")
        request = CADDesignRequest(description=description)
        result = cad_system.generate_cad_model(request)
        
        # Clean and execute code
        print("⚙️  Executing code and creating STL...")
        cleaned_code = result.cadquery_code.replace("```python", "").replace("```", "").strip()
        
        # Create visualization
        print("🎯 Creating Plotly visualization...")
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Attempt to visualize the code
                fig, stl_path, error = visualizer.visualize_from_code(cleaned_code, description)
                if error:
                    raise ValueError(error)
                break  # Exit loop if successful
            except Exception as e:
                print(f"⚠️  Attempt {attempt + 1}/{max_attempts} failed: {e}")
                if attempt < max_attempts - 1:
                    print("🔄 Retrying...")
                else:
                    print("❌ All attempts failed. Please check the generated code.")
                    return
        
        if error:
            print(f"❌ Error: {error}")
            print("\n📝 Generated code (with errors):")
            print(cleaned_code)
            return
        
        # Success!
        print("✅ Success! Opening interactive visualization...")
        
        if stl_path:
            file_size = os.path.getsize(stl_path) / 1024
            print(f"📁 STL saved: {stl_path} ({file_size:.1f} KB)")
        
        # Show the plot
        try:
            fig.show()
        except Exception as show_error:
            print(f"⚠️  Could not open browser visualization: {show_error}")
            print("💾 Saving HTML file instead...")
            html_file = f"cad_model_{int(time.time())}.html"
            fig.write_html(html_file)
            print("💡 You can open this file in your browser to view the 3D model")
        
        print(f"\n📝 Generated CADQuery Code:")
        print("-" * 40)
        print(cleaned_code)
        print("-" * 40)
        
    except Exception as e:
        print(f"❌ Failed: {e}")


def main():
    """Main function"""
    
    if len(sys.argv) < 2:
        print("🔧 Ultra Simple CAD Generator")
        print("=" * 40)
        print("Usage:")
        print("  python generate_cad.py \"description of your part\"")
        print()
        print("Examples:")
        print("  python generate_cad.py \"hex bolt M8 x 25mm\"")
        print("  python generate_cad.py \"gear with 20 teeth, 50mm diameter\"")
        print("  python generate_cad.py \"L-bracket 60x40x5mm with holes\"")
        print("  python generate_cad.py \"coffee mug with handle\"")
        return
    
    # Get description from command line
    description = " ".join(sys.argv[1:])
    generate_and_show_cad(description)


if __name__ == "__main__":
    main()
