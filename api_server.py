#!/usr/bin/env python3
"""
FastAPI server for Tax Copilot with WebSocket support for real-time progress updates.
This replaces the slow polling approach with instant state updates.
"""

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import asyncio
import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional
import threading
from contextlib import asynccontextmanager
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
import os
from typing import List

# Add allowed origins configuration
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
PRODUCTION = os.getenv("PRODUCTION", "false").lower() == "true"

# Initialize OpenAI client
client_2 = OpenAI(
    api_key=os.getenv("openai_api_key"),
    base_url="https://oai.helicone.ai/v1",
    default_headers={
        "Helicone-Auth": f"Bearer {os.getenv('HELICONE_API_KEY')}",
    }
)
# Import our analysis functions
from rag_enhanced import run_tax_copilot_analysis, update_progress, get_progress

# Global storage for analysis results
analysis_results_storage = {}

# Cleanup old analysis results (run periodically)
def cleanup_old_results():
    """Remove analysis results older than 1 hour"""
    current_time = datetime.now()
    sessions_to_remove = []
    
    for session_id, data in analysis_results_storage.items():
        if 'timestamp' in data:
            timestamp = datetime.fromisoformat(data['timestamp'])
            if (current_time - timestamp).total_seconds() > 3600:  # 1 hour
                sessions_to_remove.append(session_id)
    
    for session_id in sessions_to_remove:
        del analysis_results_storage[session_id]
        print(f"🧹 Cleaned up old results for session {session_id}")

# Background task to cleanup periodically
async def periodic_cleanup():
    while True:
        await asyncio.sleep(1800)  # Run every 30 minutes
        cleanup_old_results()

# Global WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.main_loop = None

    def set_main_loop(self, loop):
        """Store reference to the main event loop"""
        self.main_loop = loop

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        print(f"🔌 WebSocket connected for session: {session_id}")

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            print(f"🔌 WebSocket disconnected for session: {session_id}")

    async def send_progress_update(self, session_id: str, step: str, status: str, details: str = "", data: dict = None):
        """Send progress update via WebSocket"""
        if session_id in self.active_connections:
            try:
                # Create message with proper data handling
                message = {
                    "step": step,
                    "status": status,
                    "details": details,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Handle data separately to ensure it's properly included
                if data is not None:
                    message["data"] = data
                else:
                    message["data"] = {}
                
                # Special handling for large data
                if step == "legal-analysis" and status == "complete" and data:
                    print(f"📡 Preparing to send legal analysis data...")
                    print(f"📡 Data keys in message: {list(message['data'].keys())}")
                    if 'final_comprehensive_analysis' in message['data']:
                        analysis_len = len(message['data']['final_comprehensive_analysis'])
                        print(f"📡 Analysis length in message: {analysis_len}")
                
                # Serialize and check size
                message_json = json.dumps(message)
                message_size = len(message_json)
                print(f"📡 Total message size: {message_size} bytes")
                
                # Send the message
                await self.active_connections[session_id].send_text(message_json)
                print(f"📡 Sent update to {session_id}: {step} -> {status}")
                
                # Debug confirmation for legal analysis
                if step == "legal-analysis" and status == "complete":
                    print(f"✅ Legal analysis data successfully sent via WebSocket")
                    
            except Exception as e:
                print(f"❌ Error sending WebSocket message: {e}")
                import traceback
                traceback.print_exc()
                self.disconnect(session_id)

    def send_from_thread(self, session_id: str, step: str, status: str, details: str = "", data: dict = None):
        """Send progress update from a background thread"""
        print(f"🔄 send_from_thread called for session {session_id}, step: {step}")
        print(f"🔄 Main loop exists: {self.main_loop is not None}")
        print(f"🔄 Session in active connections: {session_id in self.active_connections}")
        print(f"🔄 Active sessions: {list(self.active_connections.keys())}")
        
        # Debug data being sent
        if data:
            print(f"🔄 Data keys being sent: {list(data.keys())}")
            if step == "legal-analysis" and "final_comprehensive_analysis" in data:
                print(f"🔄 Analysis data length: {len(data['final_comprehensive_analysis'])}")
        
        if self.main_loop and session_id in self.active_connections:
            try:
                # Create a coroutine that captures all parameters
                async def send_update():
                    await self.send_progress_update(session_id, step, status, details, data)
                
                future = asyncio.run_coroutine_threadsafe(
                    send_update(),
                    self.main_loop
                )
                # Wait for the coroutine to complete (with timeout)
                future.result(timeout=10.0)  # Increased timeout for large messages
                print(f"✅ WebSocket message sent successfully from thread")
            except Exception as e:
                print(f"⚠️  Could not send WebSocket update from thread: {e}")
                import traceback
                traceback.print_exc()
        else:
            if not self.main_loop:
                print(f"❌ No main loop available!")
            if session_id not in self.active_connections:
                print(f"❌ Session {session_id} not in active connections!")

manager = ConnectionManager()

# Override the update_progress function to send WebSocket updates
original_update_progress = update_progress

def websocket_update_progress(session_id: str, step: str, status: str, details: str = "", data: dict = None):
    """Enhanced update_progress that sends WebSocket updates"""
    # Call original function first
    original_update_progress(session_id, step, status, details, data)
    
    if step == "document-processing" and status == "complete" and data:
        if "processed_documents" in data:
            # Store documents in global storage
            if session_id not in analysis_results_storage:
                analysis_results_storage[session_id] = {}
            analysis_results_storage[session_id]['relevant_documents'] = data["processed_documents"]
            analysis_results_storage[session_id]['timestamp'] = datetime.now().isoformat()
            
            print(f"📚 Stored {len(data['processed_documents'])} documents for session {session_id}")
            
            # IMPORTANT: Send the actual documents, not just a notification
            documents_to_send = []
            for doc in data["processed_documents"][:10]:  # Send top 10
                documents_to_send.append({
                    "document_id": doc.get("document_id", ""),
                    "case_name": doc.get("case_name", "Unknown Case"),
                    "court_type": doc.get("court_type", ""),
                    "relevance_score": doc.get("relevance_score", 0),
                    "search_phrase": doc.get("search_phrase", ""),
                    "full_text": doc.get("full_text", "") # Limit text size
                })
            
            # Send documents directly
            websocket_data = {
                "processed_documents": documents_to_send,  # <-- This is what's missing!
                "documents_ready": True,
                "document_count": len(data["processed_documents"])
            }
            
            manager.send_from_thread(session_id, step, status, details, websocket_data)
            return
    # Store large data separately
    if step == "legal-analysis" and status == "complete" and data:
        if "final_comprehensive_analysis" in data:
            # Store the analysis in global storage
            if session_id not in analysis_results_storage:
                analysis_results_storage[session_id] = {}
            analysis_results_storage[session_id]['final_analysis'] = data["final_comprehensive_analysis"]
            analysis_results_storage[session_id]['timestamp'] = datetime.now().isoformat()
            
            print(f"📦 Stored analysis in results storage for session {session_id}")
            print(f"📦 Analysis length: {len(data['final_comprehensive_analysis'])}")
            
            # Send a lightweight notification via WebSocket
            notification_data = {
                "analysis_ready": True,
                "analysis_length": len(data["final_comprehensive_analysis"]),
                "fetch_url": f"/api/analysis/{session_id}"
            }
            manager.send_from_thread(session_id, step, status, details, notification_data)
            return
    
    # Send WebSocket update normally for other updates
    manager.send_from_thread(session_id, step, status, details, data)

# Replace the global update_progress function
import rag_enhanced
rag_enhanced.update_progress = websocket_update_progress

# Lifespan context manager to store the event loop
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Store the running event loop
    loop = asyncio.get_running_loop()
    manager.set_main_loop(loop)
    print("📡 Event loop stored for WebSocket updates")
    
    # Start periodic cleanup task
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    yield
    
    # Cleanup
    cleanup_task.cancel()
    print("🔄 Shutting down...")

app = FastAPI(
    title="Tax Copilot API", 
    description="AI-powered legal analysis with real-time updates",
    lifespan=lifespan
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if PRODUCTION else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add production logging
if PRODUCTION:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
else:
    logger = None

# Serve static files (React build) - only if build directory exists
if os.path.exists("build/static"):
    app.mount("/static", StaticFiles(directory="build/static"), name="static")

@app.get("/")
async def read_root():
    """Serve React app or API info if build doesn't exist"""
    if os.path.exists('build/index.html'):
        return FileResponse('build/index.html')
    else:
        return {
            "message": "Tax Copilot API Server",
            "status": "running",
            "frontend": "React app not built yet",
            "docs": "Visit /docs for API documentation",
            "websocket": "ws://localhost:8001/ws/analysis/{session_id}"
        }

@app.post("/api/chat")
async def chat_with_analysis(request: dict):
    """Chat endpoint that uses the analysis context to answer questions"""
    
    session_id = request.get('session_id')
    user_message = request.get('message')
    context = request.get('context', {})
    
    if not session_id or not user_message:
        raise HTTPException(status_code=400, detail="Missing session_id or message")
    
    try:
        # Prepare the context for the AI
        system_prompt = """You are a legal analysis assistant helping users understand their tax case analysis. 
        You have access to the complete analysis including primary issues, court findings, and relevant precedents.
        Be concise, accurate, and helpful. Focus on explaining legal concepts in simple terms when asked.
        Always base your answers on the provided context."""
        
        context_prompt = f"""
Context from the analysis:

PRIMARY ISSUES:
{context.get('primary_issues', 'Not available')}

RELEVANT CASES: {context.get('relevant_documents_count', 0)} cases found
Top relevant cases:
{json.dumps(context.get('relevant_documents_summary', []), indent=2)}

COMPREHENSIVE ANALYSIS:
{context.get('final_analysis', 'Not available')[:2000]}...  # Limit context size

Based on this context, please answer the user's question.
"""
        
        # Make the OpenAI call
        response = client_2.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "assistant", "content": context_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=1,
            max_tokens=500
        )
        
        return {
            "response": response.choices[0].message.content,
            "session_id": session_id
        }
        
    except Exception as e:
        print(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.websocket("/ws/analysis/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time progress updates"""
    await manager.connect(websocket, session_id)
    try:
        while True:
            # Keep connection alive and handle client messages
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send periodic ping to keep connection alive
                try:
                    await websocket.send_text(json.dumps({"type": "ping"}))
                except:
                    break
    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        manager.disconnect(session_id)

@app.post("/api/analyze")
async def analyze_document(
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    """Start document analysis with real-time progress updates"""
    
    # Validate file
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Save uploaded file
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, f"{session_id}_{file.filename}")
    
    try:
        # Save file
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Start analysis in background thread
        def run_analysis():
            try:
                # Test WebSocket communication first
                websocket_update_progress(
                    session_id, 
                    "test", 
                    "active",
                    "Testing WebSocket communication",
                    {"test_data": "This is a test message"}
                )
                
                results = run_tax_copilot_analysis(file_path, session_id)
                
                # Send final completion update using the fixed websocket_update_progress
                if results:
                    base_name = os.path.splitext(file.filename)[0]
                    output_filename = f"{base_name}.txt"
                    
                    # Extract the final analysis from results
                    final_analysis = results.get('final_comprehensive_analysis', 'No analysis generated')
                    
                    print(f"📤 Sending completion update with analysis of length: {len(final_analysis)}")
                    
                    websocket_update_progress(
                        session_id, 
                        "analysis-complete", 
                        "complete",
                        f"Analysis completed successfully. Results saved to {output_filename}",
                        {
                            "results": {
                                "final_comprehensive_analysis": final_analysis,
                                "analysis_metadata": results.get('analysis_metadata', {})
                            }, 
                            "output_filename": output_filename
                        }
                    )
                else:
                    websocket_update_progress(
                        session_id, 
                        "analysis-complete", 
                        "failed",
                        "Analysis failed. Please try again."
                    )
                    
            except Exception as e:
                print(f"❌ Analysis error: {e}")
                import traceback
                traceback.print_exc()
                websocket_update_progress(
                    session_id, 
                    "analysis-complete", 
                    "failed",
                    f"Analysis failed: {str(e)}"
                )
        
        # Start analysis in background
        analysis_thread = threading.Thread(target=run_analysis)
        analysis_thread.daemon = True
        analysis_thread.start()
        
        return {
            "message": "Analysis started",
            "session_id": session_id,
            "filename": file.filename,
            "status": "processing"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

@app.get("/api/progress/{session_id}")
async def get_analysis_progress(session_id: str):
    """Get current analysis progress (fallback for non-WebSocket clients)"""
    progress = get_progress(session_id)
    return {"session_id": session_id, "progress": progress}

@app.get("/api/analysis/{session_id}")
async def get_analysis_result(session_id: str):
    """Get the complete analysis result for a session"""
    # First check the global storage
    if session_id in analysis_results_storage and 'final_analysis' in analysis_results_storage[session_id]:
        return {
            "status": "complete",
            "analysis": analysis_results_storage[session_id]['final_analysis'],
            "timestamp": analysis_results_storage[session_id].get('timestamp')
        }
    
    # Fallback to progress tracker
    progress = get_progress(session_id)
    
    # Check if analysis is complete
    if 'legal-analysis' in progress and progress['legal-analysis']['status'] == 'complete':
        # Get stored analysis
        stored_analysis = progress.get('stored_analysis', {}).get('final', None)
        
        if stored_analysis:
            return {
                "status": "complete",
                "analysis": stored_analysis,
                "metadata": progress['legal-analysis'].get('data', {}).get('analysis_metadata', {})
            }
        else:
            # Try to get from the step data
            analysis_data = progress['legal-analysis'].get('data', {})
            if 'final_comprehensive_analysis' in analysis_data:
                return {
                    "status": "complete", 
                    "analysis": analysis_data['final_comprehensive_analysis'],
                    "metadata": analysis_data.get('analysis_metadata', {})
                }
    
    return {
        "status": "not_ready",
        "message": "Analysis not yet complete or not found"
    }

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload file and return session ID"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    session_id = str(uuid.uuid4())
    return {
        "session_id": session_id,
        "filename": file.filename,
        "message": "File uploaded successfully"
    }

@app.get("/api/analysis/{session_id}/documents")
async def get_analysis_documents(session_id: str):
    """Get relevant documents for a specific analysis session"""
    progress = get_progress(session_id)
    
    # Check document-processing step for documents
    if 'document-processing' in progress:
        doc_data = progress['document-processing'].get('data', {})
        processed_docs = doc_data.get('processed_documents', [])
        
        # Also check if stored separately
        if session_id in analysis_results_storage:
            stored_docs = analysis_results_storage[session_id].get('relevant_documents', [])
            if stored_docs:
                return {"status": "complete", "documents": stored_docs}
        
        if processed_docs:
            return {"status": "complete", "documents": processed_docs}
    
    return {"status": "not_ready", "documents": []}

@app.get("/api/download/{filename}")
async def download_analysis(filename: str):
    """Download analysis result file"""
    if not filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        filename,
        media_type='text/plain',
        filename=filename
    )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# Error handlers
from fastapi.responses import JSONResponse

@app.exception_handler(404)
async def not_found_handler(request, exc):
    if os.path.exists('build/index.html'):
        return FileResponse('build/index.html')
    else:
        return JSONResponse(
            status_code=404,
            content={
                "error": "Not found",
                "message": "React app not built yet. Visit /docs for API documentation."
            }
        )

if __name__ == "__main__":
    import uvicorn
    
    if PRODUCTION:
        print("🚀 Starting Tax Copilot API Server in PRODUCTION mode")
        uvicorn.run(
            "api_server:app",
            host="0.0.0.0",
            port=int(os.getenv("PORT", 8001)),
            log_level="info",
            access_log=True
        )
    else:
        print("🚀 Starting Tax Copilot API Server in DEVELOPMENT mode")
        print("📱 Frontend: http://localhost:8001")
        print("📡 WebSocket: ws://localhost:8001/ws/analysis/{session_id}")
        print("🔧 API Docs: http://localhost:8001/docs")
        
        uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")