# AI-Powered CAD Model Generator

**College Project - Computer Science Engineering**

An intelligent CAD model generation system that uses AI agents to create 3D models from natural language descriptions. Built with CADQuery, CrewAI, and Plotly for interactive visualization.

## Project Overview

This project demonstrates the integration of:
- **Artificial Intelligence**: Multi-agent system using CrewAI framework
- **Computer-Aided Design**: CADQuery for parametric 3D modeling
- **Natural Language Processing**: Convert descriptions to CAD code
- **3D Visualization**: Interactive Plotly-based 3D viewers
- **File Processing**: STL/STEP export for 3D printing and CAD software

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Set API Key
```bash
# Create .env file
echo "GEMINI_API_KEY=your_google_ai_api_key" > .env
```

### Generate CAD Models
```bash
# Beautiful Streamlit Web Interface (Recommended)
streamlit run streamlit_app.py

# Simple command-line usage
python cad_generator.py "hex bolt M8 x 25mm"
python cad_generator.py "gear with 20 teeth, 40mm diameter"

# HTML viewer mode (if visualization issues)
python cad_generator_html.py "L-bracket 50x30x5mm"
```

## Project Structure

```
cadquery_crewai_pro/
├── streamlit_app.py             # Beautiful Streamlit Web Interface
├── ai_cad_system.py             # Core AI agent system
├── cad_generator.py             # Simple CLI interface
├── cad_generator_html.py        # HTML viewer mode
├── visualization_engine.py      # 3D visualization engine
├── system_test.py               # System testing
├── requirements.txt             # Dependencies
├── .env.example                 # Environment template
└── outputs/                     # Generated files (STL/STEP)
```

## 🛠️ Technical Features

### AI Multi-Agent System
- **CADQuery Expert Agent**: Generates professional CAD code
- **Parameter Extractor**: Extracts dimensions and features from descriptions
- **Template Matcher**: Maps requests to proven CAD patterns
- **Code Fixer Agent**: Automatically fixes code errors

### CAD Generation Pipeline
```
Natural Language → AI Processing → CADQuery Code → 3D Model → STL/STEP Export
```

### Supported CAD Models
-  Mechanical parts (bolts, gears, brackets)
-  Enclosures and housings
-  Tools and fixtures
-  Structural components
-  Custom parametric designs

## Usage Examples

### Basic Parts
```bash
python cad_generator.py "simple box 50x30x20mm"
python cad_generator.py "washer 25mm OD, 8mm ID, 3mm thick"
```

### Complex Parts
```bash
python cad_generator.py "spur gear 16 teeth, 40mm diameter, 6mm width"
python cad_generator.py "bearing pillow block for 22mm shaft"
```

### Programming Interface
```python
from ai_cad_system import CADQueryCrewAI, CADDesignRequest
from visualization_engine import STLPlotlyVisualizer

# Initialize system
cad_system = CADQueryCrewAI()
visualizer = STLPlotlyVisualizer()

# Generate CAD model
request = CADDesignRequest(description="hex bolt M8 x 25mm")
result = cad_system.generate_cad_model(request)

# Create 3D visualization
fig, stl_path, error = visualizer.visualize_from_code(
    result.cadquery_code, 
    "Generated Bolt"
)

# Show interactive 3D model
fig.show()
```

## Technical Implementation

### CADQuery Integration
- Parametric 3D modeling with Python
- Professional engineering patterns
- Boolean operations and features
- Industry-standard file export

### AI Agent Architecture
- CrewAI framework for multi-agent coordination
- Google Gemini for natural language understanding
- Real-world CAD examples knowledge base
- Automatic error detection and correction

### Visualization System
- Plotly for interactive 3D rendering
- STL file processing and mesh display
- Mouse controls (rotate, zoom, pan)
- HTML export for sharing

## Output Files

Generated files are saved in `outputs/` directory:
- **STL files**: Ready for 3D printing
- **STEP files**: Compatible with CAD software
- **Python code**: Generated CADQuery scripts
- **HTML viewers**: Interactive 3D web pages

##  Testing

```bash
# Test the complete pipeline
python system_test.py

# Test via web interface
streamlit run streamlit_app.py
```

##  Configuration

Environment variables in `.env`:
```
GEMINI_API_KEY=your_google_ai_api_key_here
```