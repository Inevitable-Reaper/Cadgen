"""
AI-Powered CAD Generator - Modern UI/UX Design
Professional Streamlit Web App following UI/UX principles
"""

import streamlit as st
import os
import time
from pathlib import Path

# Import our CAD system
from ai_cad_system import CADQueryCrewAI, CADDesignRequest
from visualization_engine import STLPlotlyVisualizer

# Page configuration
st.set_page_config(
    page_title="AI CAD Generator",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="collapsed"  # Start with sidebar collapsed for cleaner look
)

# Modern CSS following UI/UX principles
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Reset and base styles */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* Typography hierarchy */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hero section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 4rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="white" opacity="0.1"/><circle cx="75" cy="75" r="1" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        opacity: 0.3;
    }
    
    .hero-content {
        position: relative;
        z-index: 2;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        line-height: 1.2;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        font-weight: 400;
        opacity: 0.9;
        margin-bottom: 0;
        line-height: 1.5;
    }
    
    /* Card system */
    .card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 10px 15px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1), 0 15px 35px rgba(0, 0, 0, 0.15);
    }
    
    .card-header {
        margin-bottom: 1.5rem;
    }
    
    .card-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1a202c;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .card-subtitle {
        font-size: 0.875rem;
        color: #718096;
        font-weight: 400;
    }
    
    /* Input section */
    .input-section {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        margin-bottom: 2rem;
    }
    
    /* Custom textarea */
    .stTextArea > div > div > textarea {
        border-radius: 12px !important;
        border: 2px solid #e2e8f0 !important;
        font-size: 1rem !important;
        font-family: 'Inter', sans-serif !important;
        padding: 1rem !important;
        line-height: 1.5 !important;
        resize: vertical !important;
        transition: all 0.2s ease !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        outline: none !important;
    }
    
    /* Custom button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.875rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
        transition: all 0.2s ease !important;
        text-transform: none !important;
        letter-spacing: 0 !important;
        height: auto !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    /* Example cards */
    .example-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .example-card {
        background: white;
        border: 2px solid #f7fafc;
        border-radius: 12px;
        padding: 1.5rem;
        cursor: pointer;
        transition: all 0.2s ease;
        text-align: left;
    }
    
    .example-card:hover {
        border-color: #667eea;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
    }
    
    .example-emoji {
        font-size: 2rem;
        margin-bottom: 0.75rem;
        display: block;
    }
    
    .example-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 0.5rem;
    }
    
    .example-desc {
        font-size: 0.875rem;
        color: #718096;
        line-height: 1.4;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.875rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .status-online {
        background: #f0fff4;
        color: #22543d;
        border: 1px solid #c6f6d5;
    }
    
    .status-warning {
        background: #fffaf0;
        color: #c05621;
        border: 1px solid #fed7aa;
    }
    
    /* Progress section */
    .progress-section {
        background: #f8fafc;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        border: 2px dashed #cbd5e0;
    }
    
    .progress-section.active {
        background: white;
        border: 2px solid #667eea;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.1);
    }
    
    /* Success/Error states */
    .alert {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .alert-success {
        background: #f0fff4;
        border-left-color: #38a169;
        color: #22543d;
    }
    
    .alert-error {
        background: #fed7d7;
        border-left-color: #e53e3e;
        color: #742a2a;
    }
    
    /* Metrics */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #718096;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Sidebar customization */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #edf2f7 100%);
        border-right: 1px solid #e2e8f0;
    }
    
    /* Expander customization */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: #2d3748 !important;
    }
    
    /* Remove Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }
        
        .hero-title {
            font-size: 2.5rem;
        }
        
        .hero-subtitle {
            font-size: 1.125rem;
        }
        
        .card {
            padding: 1.5rem;
        }
        
        .example-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'cad_system' not in st.session_state:
    st.session_state.cad_system = None
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = None
if 'generation_history' not in st.session_state:
    st.session_state.generation_history = []
if 'selected_example' not in st.session_state:
    st.session_state.selected_example = ""

def initialize_system():
    """Initialize the CAD generation system"""
    if st.session_state.cad_system is None:
        try:
            # Check API key
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                st.error("🔑 **API Key Required** • Please add your Google AI API key to the .env file")
                st.info("💡 Get your free API key from: https://ai.google.dev/")
                return False
            
            # Initialize systems
            with st.spinner("🤖 Initializing AI system..."):
                st.session_state.cad_system = CADQueryCrewAI()
                st.session_state.visualizer = STLPlotlyVisualizer()
            
            return True
            
        except Exception as e:
            st.error(f"❌ **Initialization Failed** • {str(e)}")
            return False
    return True

def main():
    """Main Streamlit application with modern UI/UX"""
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-content">
            <h1 class="hero-title">🛠️ AI CAD Generator</h1>
            <p class="hero-subtitle">Transform your ideas into professional 3D models using artificial intelligence</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # System initialization
    system_ready = initialize_system()
    
    if not system_ready:
        st.stop()
    
    # Status indicator
    st.markdown("""
    <div class="status-indicator status-online">
        <span>🟢</span>
        <span>AI System Online • CrewAI Multi-Agent System Ready</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content grid
    col1, col2 = st.columns([1.2, 1], gap="large")
    
    with col1:
        # Input section
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <h2 class="card-title">
                    📝 Describe Your CAD Model
                </h2>
                <p class="card-subtitle">Enter a detailed description with dimensions and features</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Text input
        description = st.text_area(
            label="",
            placeholder="Example: Create a ball and socket joint with 20mm ball diameter, 30mm socket depth, and 5mm wall thickness...",
            height=120,
            help="Be specific with dimensions, materials, and features for best results",
            key="description_input",
            label_visibility="collapsed"
        )
        
        # Update description if example was selected
        if st.session_state.selected_example and st.session_state.selected_example != description:
            description = st.session_state.selected_example
            st.rerun()
        
        # Settings in expander
        with st.expander("⚙️ **Advanced Settings**", expanded=False):
            col_set1, col_set2 = st.columns(2)
            
            with col_set1:
                complexity = st.selectbox(
                    "Complexity Level",
                    ["simple", "medium", "complex"],
                    index=1,
                    help="Choose the complexity of the generated model"
                )
                
            with col_set2:
                manufacturing = st.selectbox(
                    "Manufacturing Process",
                    ["general", "3d_printing", "cnc_milling", "casting"],
                    help="Select your intended manufacturing method"
                )
            
            auto_fix = st.checkbox(
                "🔧 **Auto-fix Errors**",
                value=True,
                help="Automatically fix code errors using AI"
            )
        
        # Generate button
        generate_button = st.button(
            "🚀 Generate 3D Model",
            type="primary",
            help="Generate your CAD model with AI"
        )
        
        # Quick Examples Section
        st.markdown("""
        <div style="margin-top: 2rem;">
            <h3 style="color: #2d3748; margin-bottom: 1rem; font-weight: 600;">💡 Quick Examples</h3>
            <p style="color: #718096; margin-bottom: 1rem; font-size: 0.875rem;">Click any example to use it as a starting point</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Example cards
        examples = [
            ("🔩", "Hex Bolt", "Create a hex head bolt M8 x 25mm with 6mm head thickness"),
            ("⚙️", "Spur Gear", "Design a spur gear with 16 teeth, 40mm diameter, 6mm width"),
            ("🔧", "L-Bracket", "Make an L-bracket 50x30x5mm with 6mm mounting holes"),
            ("🏀", "Ball Joint", "Create a ball and socket joint with 20mm ball diameter"),
            ("📦", "Enclosure", "Design an electronics enclosure 80x60x30mm with rounded edges"),
            ("⭕", "Washer", "Build a washer 25mm OD, 10mm ID, 3mm thick"),
        ]
        
        # Create example grid
        for i in range(0, len(examples), 2):
            cols = st.columns(2, gap="medium")
            for j, col in enumerate(cols):
                if i + j < len(examples):
                    emoji, title, desc = examples[i + j]
                    with col:
                        if st.button(
                            f"{emoji} **{title}**",
                            key=f"example_{i+j}",
                            use_container_width=True,
                            help=desc
                        ):
                            st.session_state.selected_example = desc
                            st.rerun()
    
    with col2:
        # Viewer section
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <h2 class="card-title">
                    🎯 3D Model Viewer
                </h2>
                <p class="card-subtitle">Your generated model will appear here</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Process generation
        if generate_button:
            if not description.strip():
                st.warning("⚠️ **Please enter a model description first!**")
            else:
                # Progress tracking
                progress_placeholder = st.empty()
                
                with progress_placeholder.container():
                    # Progress UI
                    st.markdown('<div class="progress-section active">', unsafe_allow_html=True)
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        start_time = time.time()
                        
                        # Step 1: Generate CAD code
                        status_text.info("🤖 **AI agents generating CADQuery code...**")
                        progress_bar.progress(25)
                        
                        request = CADDesignRequest(
                            description=description,
                            manufacturing_process=manufacturing,
                            complexity_target=complexity
                        )
                        
                        result = st.session_state.cad_system.generate_cad_model(request)
                        
                        # Step 2: Create visualization
                        status_text.info("🎯 **Creating 3D visualization...**")
                        progress_bar.progress(75)
                        
                        # Clean and execute code
                        cleaned_code = result.cadquery_code.replace("```python", "").replace("```", "").strip()
                        
                        # Attempt visualization with error fixing
                        max_attempts = 3 if auto_fix else 1
                        attempt = 1
                        current_code = cleaned_code
                        
                        while attempt <= max_attempts:
                            try:
                                fig, stl_path, viz_error = st.session_state.visualizer.visualize_from_code(
                                    current_code,
                                    f"Generated: {description[:30]}..."
                                )
                                
                                if viz_error is None:
                                    break
                                elif attempt < max_attempts and auto_fix:
                                    status_text.warning(f"🔧 **Auto-fixing code (attempt {attempt})...**")
                                    fix_request = CADDesignRequest(
                                        description=f"Fix this CADQuery code: {viz_error[:200]}\n\nCode:\n{current_code}",
                                        complexity_target="simple"
                                    )
                                    fixed_result = st.session_state.cad_system.generate_cad_model(fix_request)
                                    current_code = fixed_result.cadquery_code.replace("```python", "").replace("```", "").strip()
                                    attempt += 1
                                else:
                                    raise Exception(viz_error)
                                    
                            except Exception as e:
                                if attempt < max_attempts and auto_fix:
                                    attempt += 1
                                    continue
                                else:
                                    raise e
                        
                        progress_bar.progress(100)
                        generation_time = time.time() - start_time
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Clear progress
                        progress_placeholder.empty()
                        
                        if fig and stl_path:
                            # Success alert
                            st.markdown(f"""
                            <div class="alert alert-success">
                                <h3 style="margin: 0 0 0.5rem 0; color: #22543d;">✅ Model Generated Successfully!</h3>
                                <div class="metric-grid">
                                    <div class="metric-card">
                                        <div class="metric-value">{generation_time:.1f}s</div>
                                        <div class="metric-label">Generation Time</div>
                                    </div>
                                    <div class="metric-card">
                                        <div class="metric-value">{result.quality_score:.2f}</div>
                                        <div class="metric-label">Quality Score</div>
                                    </div>
                                    <div class="metric-card">
                                        <div class="metric-value">{attempt}/{max_attempts}</div>
                                        <div class="metric-label">Attempts</div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display 3D visualization
                            st.plotly_chart(fig, use_container_width=True, key="main_plot")
                            
                            # Download section
                            with st.expander("📥 **Downloads & Information**", expanded=False):
                                if os.path.exists(stl_path):
                                    file_size = os.path.getsize(stl_path) / 1024
                                    st.info(f"📁 **STL File:** {Path(stl_path).name} ({file_size:.1f} KB)")
                                
                                # Download buttons
                                col_dl1, col_dl2, col_dl3 = st.columns(3)
                                
                                safe_name = "".join(c for c in description if c.isalnum() or c in (' ', '-', '_'))
                                safe_name = safe_name.replace(' ', '_').lower()[:25]
                                
                                with col_dl1:
                                    st.download_button(
                                        "📄 Code",
                                        data=current_code,
                                        file_name=f"cad_model_{safe_name}.py",
                                        mime="text/plain",
                                        use_container_width=True
                                    )
                                
                                with col_dl2:
                                    if os.path.exists(stl_path):
                                        with open(stl_path, "rb") as file:
                                            st.download_button(
                                                "📁 STL",
                                                data=file.read(),
                                                file_name=f"model_{safe_name}.stl",
                                                mime="application/octet-stream",
                                                use_container_width=True
                                            )
                                
                                with col_dl3:
                                    html_content = fig.to_html(include_plotlyjs='cdn')
                                    st.download_button(
                                        "🌐 Viewer",
                                        data=html_content,
                                        file_name=f"viewer_{safe_name}.html",
                                        mime="text/html",
                                        use_container_width=True
                                    )
                            
                            # Code display
                            with st.expander("💻 **Generated Code**", expanded=False):
                                st.code(current_code, language="python")
                            
                            # Add to history
                            st.session_state.generation_history.append({
                                'title': description,
                                'time': generation_time,
                                'quality': result.quality_score,
                                'timestamp': time.time()
                            })
                            
                        else:
                            # Error state
                            st.markdown(f"""
                            <div class="alert alert-error">
                                <h3 style="margin: 0 0 0.5rem 0;">❌ Generation Failed</h3>
                                <p style="margin: 0;">{viz_error if viz_error else 'Unknown error occurred'}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            with st.expander("💻 **Debug Code**"):
                                st.code(current_code, language="python")
                        
                    except Exception as e:
                        progress_placeholder.empty()
                        st.markdown(f"""
                        <div class="alert alert-error">
                            <h3 style="margin: 0 0 0.5rem 0;">❌ Error Occurred</h3>
                            <p style="margin: 0;">{str(e)}</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        else:
            # Placeholder state
            st.markdown("""
            <div class="progress-section">
                <h3 style="color: #4a5568; margin-bottom: 1rem;">🎯 Ready to Generate</h3>
                <p style="color: #718096; margin-bottom: 1.5rem;">
                    Enter a description and click "Generate" to create your 3D model
                </p>
                <div style="background: white; padding: 1.5rem; border-radius: 12px; text-align: left; display: inline-block; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h4 style="color: #2d3748; margin-bottom: 1rem;">🚀 What You'll Get:</h4>
                    <ul style="color: #4a5568; line-height: 1.6; margin: 0; padding-left: 1.2rem;">
                        <li>✅ Professional CADQuery Python code</li>
                        <li>✅ Interactive 3D Plotly visualization</li>
                        <li>✅ STL files for 3D printing</li>
                        <li>✅ STEP files for CAD software</li>
                        <li>✅ Automatic error detection & fixing</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; color: #718096; padding: 3rem 0 2rem 0; margin-top: 3rem; border-top: 1px solid #e2e8f0;">
        <h3 style="color: #2d3748; margin-bottom: 0.5rem;">🎓 AI-Powered CAD Generator</h3>
        <p style="margin: 0; font-size: 0.875rem;">College Project • Computer Science Engineering</p>
        <p style="margin: 0.25rem 0 0 0; font-size: 0.75rem; opacity: 0.7;">
            Powered by CrewAI • CADQuery • Plotly • Google Gemini
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
