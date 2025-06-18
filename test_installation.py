import cadquery as cq
import streamlit as st
import langchain
import groq
import chromadb
import trimesh
import plotly

print("✅ CADQuery:", cq.__version__)
print("✅ Streamlit:", st.__version__)
print("✅ All core packages working!")

# Test CAD functionality
box = cq.Workplane("XY").box(10, 10, 5).faces(">Z").hole(3)
print("✅ CADQuery + OCP creating complex models!")