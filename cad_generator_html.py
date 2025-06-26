"""
Alternative CAD Generator that saves HTML files instead of showing directly
Avoids nbformat issues by creating downloadable HTML viewers
"""

import sys
import os
import time
from pathlib import Path
from ai_cad_system import CADQueryCrewAI, CADDesignRequest
from visualization_engine import STLPlotlyVisualizer


def generate_and_save_cad(description: str):
    """Generate CAD model from description and save as HTML viewer"""
    
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
        fig, stl_path, error = visualizer.visualize_from_code(cleaned_code, description)
        
        if error:
            print(f"❌ Error: {error}")
            print("\n📝 Generated code (with errors):")
            print(cleaned_code)
            return
        
        # Success! Save HTML file instead of showing
        print("✅ Success! Creating HTML viewer...")
        
        # Create safe filename
        safe_name = "".join(c for c in description if c.isalnum() or c in (' ', '-', '_'))
        safe_name = safe_name.replace(' ', '_').lower()[:30]
        
        html_filename = f"cad_viewer_{safe_name}_{int(time.time())}.html"
        fig.write_html(html_filename, include_plotlyjs='cdn')
        
        print(f"🌐 HTML viewer saved: {html_filename}")
        print("💡 Open this file in your browser to view the interactive 3D model")
        
        if stl_path:
            file_size = os.path.getsize(stl_path) / 1024
            print(f"📁 STL saved: {stl_path} ({file_size:.1f} KB)")
        
        print(f"\n📝 Generated CADQuery Code:")
        print("-" * 40)
        print(cleaned_code)
        print("-" * 40)
        
        # Also save the code
        code_filename = f"cad_code_{safe_name}_{int(time.time())}.py"
        with open(code_filename, 'w') as f:
            f.write(f'"""\nGenerated CADQuery code for: {description}\n"""\n\n')
            f.write("import cadquery as cq\n\n")
            f.write(cleaned_code)
        
        print(f"💾 Code saved: {code_filename}")
        
        return html_filename, stl_path, code_filename
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function"""
    
    if len(sys.argv) < 2:
        print("🔧 CAD Generator - HTML Viewer Mode")
        print("=" * 40)
        print("Usage:")
        print("  python generate_cad_html.py \"description of your part\"")
        print()
        print("This version saves HTML files instead of opening browser directly")
        print("(Useful when you get nbformat errors)")
        print()
        print("Examples:")
        print("  python generate_cad_html.py \"hex bolt M8 x 25mm\"")
        print("  python generate_cad_html.py \"gear with 20 teeth, 50mm diameter\"")
        print("  python generate_cad_html.py \"L-bracket 60x40x5mm with holes\"")
        print("  python generate_cad_html.py \"coffee mug with handle\"")
        return
    
    # Get description from command line
    description = " ".join(sys.argv[1:])
    
    result = generate_and_save_cad(description)
    
    if result:
        html_file, stl_file, code_file = result
        print(f"\n🎯 Generation Complete!")
        print(f"📂 Files created:")
        print(f"   🌐 HTML Viewer: {html_file}")
        if stl_file:
            print(f"   📁 STL File: {stl_file}")
        if code_file:
            print(f"   💾 Code File: {code_file}")
        print(f"\n💡 Double-click the HTML file to view your 3D model!")


if __name__ == "__main__":
    main()
