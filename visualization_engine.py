"""
Clean CAD Visualizer with STL Export and Plotly Visualization
Simplified approach focusing on STL generation and Plotly 3D display
"""

import cadquery as cq
import numpy as np
import tempfile
import os
import time
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from typing import Tuple, Optional, List
import traceback


class STLPlotlyVisualizer:
    """Clean CAD visualizer focused on STL export and Plotly visualization"""
    
    def __init__(self):
        self.temp_files = []
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
    
    def execute_cadquery_code(self, code_string: str) -> Tuple[Optional[cq.Workplane], Optional[str]]:
        """Execute CADQuery code safely and return the result object"""
        try:
            # Clean environment for safe execution
            safe_globals = {
                'cq': cq,
                'cadquery': cq,
                'math': __import__('math'),
                'numpy': np,
                'np': np,
                '__builtins__': {
                    'print': print, 'len': len, 'range': range, 'enumerate': enumerate,
                    'zip': zip, 'list': list, 'dict': dict, 'str': str, 'int': int,
                    'float': float, 'bool': bool, 'abs': abs, 'min': min, 'max': max,
                    'round': round, 'sum': sum,
                }
            }
            
            # Clean the code - remove imports and empty lines
            lines = code_string.split('\n')
            cleaned_lines = []
            
            for line in lines:
                stripped = line.strip()
                if (stripped.startswith('import ') or 
                    stripped.startswith('from ') or
                    len(stripped) == 0 or
                    stripped.startswith('#')):
                    continue
                cleaned_lines.append(line)
            
            cleaned_code = '\n'.join(cleaned_lines)
            
            # Execute the code
            exec(cleaned_code, safe_globals)
            
            # Get the result
            if 'result' in safe_globals:
                result_obj = safe_globals['result']
                print(f"✅ CADQuery code executed successfully")
                return result_obj, None
            else:
                return None, "No 'result' variable found in code"
                
        except Exception as e:
            error_msg = f"Code execution error: {str(e)}\n{traceback.format_exc()}"
            print(f"❌ {error_msg}")
            return None, error_msg
    
    def export_to_stl(self, cq_object: cq.Workplane, filename: str = None) -> Tuple[Optional[str], Optional[str]]:
        """Export CADQuery object to STL file"""
        try:
            if filename is None:
                filename = f"cad_model_{int(time.time())}.stl"
            
            stl_path = self.output_dir / filename
            
            # Export to STL
            cq.exporters.export(cq_object, str(stl_path))
            
            if stl_path.exists():
                print(f"✅ STL exported: {stl_path}")
                return str(stl_path), None
            else:
                return None, "STL file was not created"
                
        except Exception as e:
            error_msg = f"STL export failed: {str(e)}"
            print(f"❌ {error_msg}")
            return None, error_msg
    
    def export_to_step(self, cq_object: cq.Workplane, filename: str = None) -> Tuple[Optional[str], Optional[str]]:
        """Export CADQuery object to STEP file"""
        try:
            if filename is None:
                filename = f"cad_model_{int(time.time())}.step"
            
            step_path = self.output_dir / filename
            
            # Export to STEP
            cq.exporters.export(cq_object, str(step_path))
            
            if step_path.exists():
                print(f"✅ STEP exported: {step_path}")
                return str(step_path), None
            else:
                return None, "STEP file was not created"
                
        except Exception as e:
            error_msg = f"STEP export failed: {str(e)}"
            print(f"❌ {error_msg}")
            return None, error_msg
    
    def read_stl_file(self, stl_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
        """Read STL file and extract vertices and faces"""
        try:
            with open(stl_path, 'rb') as f:
                # Check if binary STL
                header = f.read(80)
                if header.startswith(b'solid '):
                    # ASCII STL - rewind and read as text
                    f.seek(0)
                    return self._read_stl_ascii(f)
                else:
                    # Binary STL
                    return self._read_stl_binary(f)
                    
        except Exception as e:
            error_msg = f"STL reading failed: {str(e)}"
            print(f"❌ {error_msg}")
            return None, None, error_msg
    
    def _read_stl_binary(self, f) -> Tuple[np.ndarray, np.ndarray, None]:
        """Read binary STL format"""
        # Skip header (already read 80 bytes)
        num_triangles_bytes = f.read(4)
        if len(num_triangles_bytes) != 4:
            raise Exception("Invalid STL file")
        num_triangles = int.from_bytes(num_triangles_bytes, byteorder='little')
        
        vertices = []
        faces = []
        
        for i in range(num_triangles):
            # Skip normal (12 bytes)
            f.read(12)
            
            # Read triangle vertices
            triangle_vertices = []
            for j in range(3):
                x = np.frombuffer(f.read(4), dtype=np.float32)[0]
                y = np.frombuffer(f.read(4), dtype=np.float32)[0]
                z = np.frombuffer(f.read(4), dtype=np.float32)[0]
                triangle_vertices.append([x, y, z])
            
            # Add vertices and face
            start_idx = len(vertices)
            vertices.extend(triangle_vertices)
            faces.append([start_idx, start_idx + 1, start_idx + 2])
            
            # Skip attribute (2 bytes)
            f.read(2)
        
        return np.array(vertices), np.array(faces), None
    
    def _read_stl_ascii(self, f) -> Tuple[np.ndarray, np.ndarray, None]:
        """Read ASCII STL format"""
        vertices = []
        faces = []
        current_triangle = []
        
        f.seek(0)
        for line in f:
            line = line.decode('utf-8').strip()
            if line.startswith('vertex'):
                coords = list(map(float, line.split()[1:4]))
                current_triangle.append(coords)
                
                if len(current_triangle) == 3:
                    start_idx = len(vertices)
                    vertices.extend(current_triangle)
                    faces.append([start_idx, start_idx + 1, start_idx + 2])
                    current_triangle = []
        
        return np.array(vertices), np.array(faces), None
    
    def create_plotly_visualization(self, vertices: np.ndarray, faces: np.ndarray, title: str = "CAD Model") -> go.Figure:
        """Create interactive Plotly 3D visualization"""
        try:
            # Create mesh3d trace
            fig = go.Figure(data=[
                go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1], 
                    z=vertices[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color='lightsteelblue',
                    opacity=0.8,
                    lighting=dict(ambient=0.5, diffuse=0.8, specular=0.1),
                    lightposition=dict(x=100, y=100, z=100),
                    name="CAD Model"
                )
            ])
            
            # Calculate bounds for proper scaling
            x_range = [vertices[:, 0].min(), vertices[:, 0].max()]
            y_range = [vertices[:, 1].min(), vertices[:, 1].max()]
            z_range = [vertices[:, 2].min(), vertices[:, 2].max()]
            
            # Update layout for better visualization
            fig.update_layout(
                title=dict(
                    text=title,
                    x=0.5,
                    font=dict(size=16, color='black')
                ),
                scene=dict(
                    xaxis=dict(title='X (mm)', range=x_range, showgrid=True),
                    yaxis=dict(title='Y (mm)', range=y_range, showgrid=True),
                    zaxis=dict(title='Z (mm)', range=z_range, showgrid=True),
                    aspectmode='data',
                    camera=dict(
                        eye=dict(x=1.2, y=1.2, z=1.2),
                        projection=dict(type='perspective')
                    ),
                    bgcolor='white'
                ),
                width=800,
                height=600,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            print(f"✅ Plotly visualization created: {len(vertices)} vertices, {len(faces)} faces")
            return fig
            
        except Exception as e:
            print(f"❌ Plotly visualization failed: {str(e)}")
            raise
    
    def visualize_cadquery_object(self, cq_object: cq.Workplane, title: str = "CAD Model") -> Tuple[Optional[go.Figure], Optional[str], Optional[str]]:
        """Complete pipeline: CADQuery object -> STL -> Plotly visualization"""
        try:
            # Export to STL
            stl_path, export_error = self.export_to_stl(cq_object)
            if export_error:
                return None, None, export_error
            
            # Also export STEP for compatibility
            step_path, _ = self.export_to_step(cq_object)
            
            # Read STL and create visualization
            vertices, faces, read_error = self.read_stl_file(stl_path)
            if read_error:
                return None, stl_path, read_error
            
            # Create Plotly figure
            fig = self.create_plotly_visualization(vertices, faces, title)
            
            return fig, stl_path, None
            
        except Exception as e:
            error_msg = f"Visualization pipeline failed: {str(e)}"
            print(f"❌ {error_msg}")
            return None, None, error_msg
    
    def visualize_from_code(self, cadquery_code: str, title: str = "CAD Model") -> Tuple[Optional[go.Figure], Optional[str], Optional[str]]:
        """Complete pipeline: CADQuery code -> execution -> STL -> Plotly"""
        try:
            print("🔄 Executing CADQuery code...")
            
            # Execute code
            cq_object, exec_error = self.execute_cadquery_code(cadquery_code)
            if exec_error:
                return None, None, f"Code execution failed: {exec_error}"
            
            print("✅ Code executed successfully")
            print("🔄 Creating STL and visualization...")
            
            # Create visualization
            fig, stl_path, viz_error = self.visualize_cadquery_object(cq_object, title)
            if viz_error:
                return None, stl_path, f"Visualization failed: {viz_error}"
            
            print("✅ Complete pipeline successful!")
            return fig, stl_path, None
            
        except Exception as e:
            error_msg = f"Complete pipeline failed: {str(e)}"
            print(f"❌ {error_msg}")
            return None, None, error_msg
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass
        self.temp_files = []
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.cleanup_temp_files()


def test_stl_plotly_visualizer():
    """Test the STL + Plotly visualizer"""
    
    test_code = '''
# Simple test gear
import cadquery as cq
import math

# Gear parameters
num_teeth = 12
pitch_diameter = 30
thickness = 6
bore_diameter = 6

# Create gear body
result = (cq.Workplane("XY")
          .circle(pitch_diameter/2)
          .extrude(thickness))

# Add simplified teeth
tooth_angle = 360.0 / num_teeth
for i in range(num_teeth):
    angle = i * tooth_angle
    tooth = (cq.Workplane("XY")
             .center((pitch_diameter/2 + 1.5) * math.cos(math.radians(angle)),
                    (pitch_diameter/2 + 1.5) * math.sin(math.radians(angle)))
             .circle(1.2)
             .extrude(thickness))
    result = result.union(tooth)

# Center bore
result = (result
          .faces(">Z")
          .workplane()
          .circle(bore_diameter/2)
          .cutThruAll())

print("Test gear created")
'''
    
    print("🧪 Testing STL + Plotly Visualizer")
    print("=" * 50)
    
    try:
        visualizer = STLPlotlyVisualizer()
        
        fig, stl_path, error = visualizer.visualize_from_code(
            test_code, 
            title="Test Gear - STL + Plotly"
        )
        
        if error:
            print(f"❌ Test failed: {error}")
            return False
        
        if fig:
            # Save HTML file
            fig.write_html("test_gear_plotly.html")
            print("✅ Plotly visualization saved: test_gear_plotly.html")
        
        if stl_path:
            print(f"✅ STL file created: {stl_path}")
        
        print("✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False


if __name__ == "__main__":
    import time
    test_stl_plotly_visualizer()
