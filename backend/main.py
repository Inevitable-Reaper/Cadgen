"""
FastAPI Backend for AI CAD Generator
Production-ready REST API for React frontend
Updated to use modern lifespan event handlers
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import time
import uuid
import os
import json
from typing import Optional, List
from datetime import datetime
from contextlib import asynccontextmanager

# Import CAD system
from ai_cad_system import CADQueryCrewAI, CADDesignRequest
from visualization_engine import STLPlotlyVisualizer

# Global instances
cad_system = None
visualizer = None
generation_jobs = {}  # Store job status

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    global cad_system, visualizer
    try:
        print("🚀 Initializing AI CAD Generator...")
        
        # Check API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ GEMINI_API_KEY not found!")
            print("💡 Please add your Google AI API key to the .env file")
        else:
            print(f"✅ API key found (length: {len(api_key)})")
            
            # Initialize CAD system
            cad_system = CADQueryCrewAI()
            visualizer = STLPlotlyVisualizer()
            print("✅ CAD system initialized successfully")
        
    except Exception as e:
        print(f"❌ Failed to initialize CAD system: {e}")
        print("💡 Check your API key and dependencies")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down AI CAD Generator...")
    # Cleanup if needed
    cad_system = None
    visualizer = None

# Create FastAPI app with lifespan
app = FastAPI(
    title="AI CAD Generator API", 
    version="1.0.0",
    description="Professional AI-powered CAD model generation API",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:3001",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create output directories
output_htmls_dir = Path("output_htmls")
outputs_dir = Path("outputs")
output_htmls_dir.mkdir(exist_ok=True)
outputs_dir.mkdir(exist_ok=True)

# Static file serving
app.mount("/output_htmls", StaticFiles(directory="output_htmls"), name="html_files")
app.mount("/outputs", StaticFiles(directory="outputs"), name="cad_files")

# Request/Response models
class CADGenerationRequest(BaseModel):
    description: str
    complexity: str = "medium"
    manufacturing_process: str = "general"

class JobStatus(BaseModel):
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: int  # 0-100
    message: str
    created_at: str
    completed_at: Optional[str] = None

class CADModel(BaseModel):
    id: str
    name: str
    description: str
    created_at: str
    html_url: str
    stl_url: Optional[str] = None
    step_url: Optional[str] = None
    vertex_count: Optional[int] = None
    generation_time: Optional[float] = None
    file_size_kb: Optional[int] = None

class CADGenerationResponse(BaseModel):
    success: bool
    job_id: str
    message: str
    model: Optional[CADModel] = None

# Health check
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cad_system_ready": cad_system is not None,
        "visualizer_ready": visualizer is not None,
        "api_key_configured": os.getenv("GEMINI_API_KEY") is not None,
        "fastapi_version": "Modern lifespan handlers",
        "existing_models": len(list(output_htmls_dir.glob("*_metadata.json")))
    }

# Generate CAD model
@app.post("/api/generate", response_model=CADGenerationResponse)
async def generate_cad_model(request: CADGenerationRequest, background_tasks: BackgroundTasks):
    """Generate CAD model from description"""
    if not cad_system or not visualizer:
        raise HTTPException(
            status_code=503, 
            detail="CAD system not initialized. Check API key configuration."
        )
    
    job_id = str(uuid.uuid4())[:8]
    
    # Initialize job tracking
    generation_jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        progress=0,
        message="Job created",
        created_at=datetime.now().isoformat()
    )
    
    # Start background generation
    background_tasks.add_task(process_cad_generation, job_id, request)
    
    return CADGenerationResponse(
        success=True,
        job_id=job_id,
        message="Generation job started",
        model=None
    )

async def process_cad_generation(job_id: str, request: CADGenerationRequest):
    """Background task to process CAD generation with 3-attempt retry"""
    max_code_attempts = 3
    
    for code_attempt in range(max_code_attempts):
        try:
            # Update status
            generation_jobs[job_id].status = "processing"
            if code_attempt == 0:
                generation_jobs[job_id].progress = 25
                generation_jobs[job_id].message = "Generating CAD code..."
            else:
                generation_jobs[job_id].progress = 25 + (code_attempt * 10)
                generation_jobs[job_id].message = f"Retry {code_attempt + 1}/{max_code_attempts}: Generating CAD code..."
            
            start_time = time.time()
            
            # Generate CAD code
            cad_request = CADDesignRequest(
                description=request.description,
                complexity_target=request.complexity,
                manufacturing_process=request.manufacturing_process
            )
            
            # Add retry context to the request if this is a retry
            if code_attempt > 0:
                cad_request.description += f" (Retry {code_attempt + 1}: CRITICAL - Previous attempt failed. Use ONLY simple box(), circle(), extrude() operations. NO fillets, NO chamfers, NO complex curves. Keep it simple!)"
            
            result = cad_system.generate_cad_model(cad_request)
            
            # Update progress
            generation_jobs[job_id].progress = 60
            generation_jobs[job_id].message = "Creating 3D visualization..."
            
            # Clean and execute code
            cleaned_code = result.cadquery_code.replace("```python", "").replace("```", "").strip()
            
            # Create visualization with job_id for consistent naming
            result_tuple = visualizer.visualize_from_code(
                cleaned_code,
                f"Generated: {request.description[:30]}...",
                job_id  # Pass job_id for consistent file naming
            )
            
            # Handle return values (should always be 4 now)
            if len(result_tuple) == 4:
                fig, stl_path, viz_error, html_path = result_tuple
            else:
                # Fallback for old API
                fig, stl_path, viz_error = result_tuple
                html_path = None
            
            if viz_error:
                print(f"❌ Code attempt {code_attempt + 1} failed: {viz_error}")
                if code_attempt < max_code_attempts - 1:
                    print(f"🔄 Retrying code generation... ({code_attempt + 2}/{max_code_attempts})")
                    continue
                else:
                    generation_jobs[job_id].status = "failed"
                    generation_jobs[job_id].message = f"All {max_code_attempts} attempts failed. Last error: {viz_error}"
                    return
            
            # Success! Break out of retry loop
            print(f"✅ Code generation successful on attempt {code_attempt + 1}")
            break
            
        except Exception as e:
            print(f"❌ Code generation attempt {code_attempt + 1} failed: {str(e)}")
            if code_attempt < max_code_attempts - 1:
                print(f"🔄 Retrying code generation... ({code_attempt + 2}/{max_code_attempts})")
                continue
            else:
                generation_jobs[job_id].status = "failed"
                generation_jobs[job_id].message = f"All {max_code_attempts} code generation attempts failed. Last error: {str(e)}"
                return
    
    try:
        generation_time = time.time() - start_time
        
        # Get model info
        vertex_count = 0
        if fig and hasattr(fig, 'data') and len(fig.data) > 0:
            mesh_data = fig.data[0]
            if hasattr(mesh_data, 'x') and mesh_data.x is not None:
                vertex_count = len(mesh_data.x)
        
        # Create model metadata
        model_data = {
            "id": job_id,
            "name": f"Model_{job_id}",
            "description": request.description,
            "created_at": datetime.now().isoformat(),
            "generation_time": generation_time,
            "vertex_count": vertex_count,
            "html_path": html_path,
            "stl_path": stl_path
        }
        
        # Save metadata
        metadata_path = output_htmls_dir / f"{job_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        # Update job status
        generation_jobs[job_id].status = "completed"
        generation_jobs[job_id].progress = 100
        generation_jobs[job_id].message = "Generation completed successfully"
        generation_jobs[job_id].completed_at = datetime.now().isoformat()
        
        print(f"✅ Job {job_id} completed successfully")
        
    except Exception as e:
        generation_jobs[job_id].status = "failed"
        generation_jobs[job_id].message = f"Generation failed: {str(e)}"
        print(f"❌ Job {job_id} failed: {e}")

# Get job status
@app.get("/api/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get generation job status"""
    if job_id not in generation_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return generation_jobs[job_id]

# Get all models
@app.get("/api/models", response_model=List[CADModel])
async def get_all_models():
    """Get list of all generated models"""
    models = []
    
    for metadata_file in output_htmls_dir.glob("*_metadata.json"):
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            
            job_id = data["id"]
            html_file = output_htmls_dir / f"cad_model_{job_id}.html"
            stl_file = outputs_dir / f"cad_model_{job_id}.stl"
            step_file = outputs_dir / f"cad_model_{job_id}.step"
            
            # Check file sizes
            file_size_kb = html_file.stat().st_size // 1024 if html_file.exists() else 0
            
            model = CADModel(
                id=job_id,
                name=data.get("name", f"Model_{job_id}"),
                description=data.get("description", ""),
                created_at=data.get("created_at", ""),
                html_url=f"/output_htmls/cad_model_{job_id}.html",
                stl_url=f"/outputs/cad_model_{job_id}.stl" if stl_file.exists() else None,
                step_url=f"/outputs/cad_model_{job_id}.step" if step_file.exists() else None,
                vertex_count=data.get("vertex_count"),
                generation_time=data.get("generation_time"),
                file_size_kb=file_size_kb
            )
            models.append(model)
            
        except Exception as e:
            print(f"Error loading model metadata: {e}")
            continue
    
    # Sort by creation date (newest first)
    models.sort(key=lambda x: x.created_at, reverse=True)
    return models

# Get specific model
@app.get("/api/models/{model_id}", response_model=CADModel)
async def get_model(model_id: str):
    """Get specific model details"""
    metadata_path = output_htmls_dir / f"{model_id}_metadata.json"
    
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    
    with open(metadata_path, 'r') as f:
        data = json.load(f)
    
    html_file = output_htmls_dir / f"cad_model_{model_id}.html"
    stl_file = outputs_dir / f"cad_model_{model_id}.stl"
    step_file = outputs_dir / f"cad_model_{model_id}.step"
    
    file_size_kb = html_file.stat().st_size // 1024 if html_file.exists() else 0
    
    return CADModel(
        id=model_id,
        name=data.get("name", f"Model_{model_id}"),
        description=data.get("description", ""),
        created_at=data.get("created_at", ""),
        html_url=f"/output_htmls/cad_model_{model_id}.html",
        stl_url=f"/outputs/cad_model_{model_id}.stl" if stl_file.exists() else None,
        step_url=f"/outputs/cad_model_{model_id}.step" if step_file.exists() else None,
        vertex_count=data.get("vertex_count"),
        generation_time=data.get("generation_time"),
        file_size_kb=file_size_kb
    )

# Delete model
@app.delete("/api/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a generated model"""
    metadata_path = output_htmls_dir / f"{model_id}_metadata.json"
    html_path = output_htmls_dir / f"cad_model_{model_id}.html"
    stl_path = outputs_dir / f"cad_model_{model_id}.stl"
    step_path = outputs_dir / f"cad_model_{model_id}.step"
    
    deleted_files = []
    for file_path in [metadata_path, html_path, stl_path, step_path]:
        if file_path.exists():
            file_path.unlink()
            deleted_files.append(str(file_path.name))
    
    # Remove from jobs if exists
    if model_id in generation_jobs:
        del generation_jobs[model_id]
    
    return {
        "success": True,
        "message": f"Deleted model {model_id}",
        "deleted_files": deleted_files
    }

# Get model preview HTML
@app.get("/api/models/{model_id}/preview")
async def get_model_preview(model_id: str):
    """Get model preview HTML"""
    html_path = output_htmls_dir / f"cad_model_{model_id}.html"
    
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Model HTML not found")
    
    return FileResponse(html_path, media_type="text/html")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI CAD Generator API...")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/api/health")
    uvicorn.run(app, host="0.0.0.0", port=8000)
