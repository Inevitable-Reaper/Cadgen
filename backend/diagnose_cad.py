"""
CAD System Diagnostic Script
Test initialization step by step
"""

import os
import sys
from pathlib import Path

def test_environment():
    """Test environment setup"""
    print("🔍 Testing Environment Setup...")
    
    # Test 1: Check if .env file exists
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file found")
        with open(env_file, 'r') as f:
            content = f.read()
            if "GEMINI_API_KEY" in content:
                print("✅ GEMINI_API_KEY found in .env file")
            else:
                print("❌ GEMINI_API_KEY not found in .env file")
    else:
        print("❌ .env file not found")
        return False
    
    # Test 2: Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ dotenv loaded successfully")
    except ImportError:
        print("❌ python-dotenv not installed")
        return False
    except Exception as e:
        print(f"❌ Failed to load .env: {e}")
        return False
    
    # Test 3: Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        print(f"✅ API key loaded (length: {len(api_key)})")
        if len(api_key) < 30:
            print("⚠️ API key seems too short")
        else:
            print("✅ API key length looks good")
    else:
        print("❌ GEMINI_API_KEY not found in environment")
        return False
    
    return True

def test_imports():
    """Test critical imports"""
    print("\n🔍 Testing Critical Imports...")
    
    imports_to_test = [
        ("cadquery", "cq"),
        ("plotly.graph_objects", "go"),
        ("numpy", "np"),
        ("crewai", "Agent, Task, Crew"),
        ("google.generativeai", "genai"),
        ("pydantic", "BaseModel"),
    ]
    
    for module_name, import_as in imports_to_test:
        try:
            if module_name == "cadquery":
                import cadquery as cq
                print(f"✅ {module_name} imported successfully")
            elif module_name == "plotly.graph_objects":
                import plotly.graph_objects as go
                print(f"✅ {module_name} imported successfully")
            elif module_name == "numpy":
                import numpy as np
                print(f"✅ {module_name} imported successfully")
            elif module_name == "crewai":
                from crewai import Agent, Task, Crew
                print(f"✅ {module_name} imported successfully")
            elif module_name == "google.generativeai":
                import google.generativeai as genai
                print(f"✅ {module_name} imported successfully")
            elif module_name == "pydantic":
                from pydantic import BaseModel
                print(f"✅ {module_name} imported successfully")
                
        except ImportError as e:
            print(f"❌ {module_name} import failed: {e}")
            return False
        except Exception as e:
            print(f"❌ {module_name} error: {e}")
            return False
    
    return True

def test_cad_system_init():
    """Test CAD system initialization"""
    print("\n🔍 Testing CAD System Initialization...")
    
    try:
        # Test AI system
        print("Testing ai_cad_system import...")
        from ai_cad_system import CADQueryCrewAI, CADDesignRequest
        print("✅ ai_cad_system imported successfully")
        
        print("Testing CAD system initialization...")
        cad_system = CADQueryCrewAI()
        print("✅ CADQueryCrewAI initialized successfully")
        
        # Test visualization engine
        print("Testing visualization_engine import...")
        from visualization_engine import STLPlotlyVisualizer
        print("✅ visualization_engine imported successfully")
        
        print("Testing visualizer initialization...")
        visualizer = STLPlotlyVisualizer()
        print("✅ STLPlotlyVisualizer initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ CAD system initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic tests"""
    print("🏥 CAD System Diagnostic")
    print("=" * 50)
    
    # Test 1: Environment
    if not test_environment():
        print("\n❌ Environment test failed!")
        return
    
    # Test 2: Imports  
    if not test_imports():
        print("\n❌ Import test failed!")
        print("💡 Try: pip install -r requirements.txt")
        return
    
    # Test 3: CAD System
    if not test_cad_system_init():
        print("\n❌ CAD system initialization failed!")
        return
    
    print("\n🎉 All tests passed!")
    print("✅ CAD system should work correctly")

if __name__ == "__main__":
    main()
