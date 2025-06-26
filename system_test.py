"""
Test the clean STL + Plotly pipeline
"""

import time
from ai_cad_system import CADQueryCrewAI, CADDesignRequest
from visualization_engine import STLPlotlyVisualizer


def test_simple_pipeline():
    """Test the simple pipeline end to end"""
    
    print("🧪 Testing Simple STL + Plotly Pipeline")
    print("=" * 50)
    
    # Simple test case
    test_description = "Create a simple hex bolt M6 x 20mm"
    
    try:
        print("🤖 Initializing AI agents system...")
        cad_system = CADQueryCrewAI()  # No fallback - always use AI agents
        visualizer = STLPlotlyVisualizer()
        
        print("✍️  Generating CADQuery code...")
        request = CADDesignRequest(description=test_description)
        result = cad_system.generate_cad_model(request)
        
        print(f"✅ Code generated (Quality: {result.quality_score:.2f})")
        
        # Clean code
        cleaned_code = result.cadquery_code.replace("```python", "").replace("```", "").strip()
        
        print("🎯 Creating visualization...")
        fig, stl_path, error = visualizer.visualize_from_code(cleaned_code, test_description)
        
        if error:
            print(f"❌ Visualization error: {error}")
            return False
        
        print("✅ Visualization created successfully!")
        
        if stl_path:
            print(f"📁 STL file: {stl_path}")
        
        if fig:
            print("🎯 Showing interactive plot...")
            try:
                fig.show()
            except Exception as show_error:
                print(f"⚠️  Could not open browser visualization: {show_error}")
                print("💾 Saving HTML file instead...")
                html_file = f"test_visualization_{int(time.time())}.html"
                fig.write_html(html_file)
                print(f"✅ HTML saved: {html_file}")
                print("💡 You can open this file in your browser to view the 3D model")
        
        print("\n📝 Generated Code:")
        print("-" * 30)
        print(cleaned_code)
        print("-" * 30)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_simple_pipeline()
