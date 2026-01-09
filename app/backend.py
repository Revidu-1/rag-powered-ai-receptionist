"""
FastAPI Backend for Appointment Booking System
Handles appointment booking and retrieval for the AI receptionist
"""

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import json
import os
from pathlib import Path
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

app = FastAPI(title="Appointment Booking API", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    """Log startup to verify app is loading."""
    try:
        log_path = r"c:\Users\revid\RAG\.cursor\debug.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"STARTUP","location":"app/backend.py:startup_event","message":"FastAPI app started","data":{"frontend_dir":FRONTEND_DIR,"base_dir":BASE_DIR},"timestamp":int(time.time()*1000)}) + "\n")
    except Exception as e:
        print(f"[STARTUP LOG ERROR] {e}")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend (serves files from frontend directory)
# This will be set up after BASE_DIR is defined

# Data storage file - relative to project root
# Get the project root directory (parent of app/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
APPOINTMENTS_FILE = os.path.join(DATA_DIR, "appointments.json")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FRONTEND_DIR, exist_ok=True)

# Debug: Print paths (can be removed later)
print(f"[BACKEND] BASE_DIR: {BASE_DIR}")
print(f"[BACKEND] FRONTEND_DIR: {FRONTEND_DIR}")
print(f"[BACKEND] Dashboard exists: {os.path.exists(os.path.join(FRONTEND_DIR, 'dashboard.html'))}")
print(f"[BACKEND] Chat exists: {os.path.exists(os.path.join(FRONTEND_DIR, 'chat.html'))}")

# Request logging middleware
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        # #region agent log
        try:
            log_path = r"c:\Users\revid\RAG\.cursor\debug.log"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"app/backend.py:RequestLoggingMiddleware","message":"Request received","data":{"method":request.method,"path":str(request.url.path)},"timestamp":int(time.time()*1000)}) + "\n")
        except Exception as e:
            print(f"[LOG ERROR] {e}")
        # #endregion
        response = await call_next(request)
        # #region agent log
        try:
            log_path = r"c:\Users\revid\RAG\.cursor\debug.log"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"app/backend.py:RequestLoggingMiddleware","message":"Response sent","data":{"method":request.method,"path":str(request.url.path),"status_code":response.status_code},"timestamp":int(time.time()*1000)}) + "\n")
        except Exception as e:
            print(f"[LOG ERROR] {e}")
        # #endregion
        return response

app.add_middleware(RequestLoggingMiddleware)

# Mount static files directory for frontend assets (optional, for serving static assets)
if os.path.exists(FRONTEND_DIR):
    try:
        app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
        print(f"[BACKEND] Mounted static files from: {FRONTEND_DIR}")
    except Exception as e:
        print(f"[WARNING] Could not mount static files: {e}")


# Pydantic models
class AppointmentRequest(BaseModel):
    name: str = Field(..., description="Name of the person booking the appointment")
    email: str = Field(..., description="Email address")
    date: str = Field(..., description="Date of appointment (YYYY-MM-DD)")
    time: str = Field(..., description="Time of appointment (HH:MM)")
    purpose: Optional[str] = Field(None, description="Purpose/reason for appointment")
    service: Optional[str] = Field(None, description="Service type (e.g., haircut, manicure, hair color, pedicure)")


class Appointment(BaseModel):
    id: int
    name: str
    email: str
    date: str
    time: str
    purpose: Optional[str] = None
    service: Optional[str] = None
    created_at: str
    status: str = "scheduled"  # scheduled, completed, cancelled


class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None  # To maintain conversation context


class ChatResponse(BaseModel):
    response: str
    session_id: str
    appointment_booked: bool = False
    appointment_id: Optional[int] = None


def load_appointments() -> List[Appointment]:
    """Load appointments from JSON file."""
    if os.path.exists(APPOINTMENTS_FILE):
        try:
            with open(APPOINTMENTS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Handle backward compatibility: convert 'department' to 'service' if present
                appointments = []
                for item in data:
                    if 'department' in item and 'service' not in item:
                        item['service'] = item.pop('department')
                    appointments.append(Appointment(**item))
                return appointments
        except (json.JSONDecodeError, KeyError):
            return []
    return []


def save_appointments(appointments: List[Appointment]):
    """Save appointments to JSON file."""
    # #region agent log
    import json as json_module
    import os
    with open(r'c:\Users\revid\RAG\.cursor\debug.log', 'a', encoding='utf-8') as f:
        f.write(json_module.dumps({"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"backend.py:save_appointments:entry","message":"Saving appointments to file","data":{"file_path":APPOINTMENTS_FILE,"count":len(appointments),"file_exists":os.path.exists(APPOINTMENTS_FILE)}}) + '\n')
    # #endregion
    with open(APPOINTMENTS_FILE, 'w', encoding='utf-8') as f:
        json.dump([apt.dict() for apt in appointments], f, indent=2, ensure_ascii=False)
    # #region agent log
    with open(r'c:\Users\revid\RAG\.cursor\debug.log', 'a', encoding='utf-8') as f:
        f.write(json_module.dumps({"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"backend.py:save_appointments:after_write","message":"File written successfully","data":{"file_path":APPOINTMENTS_FILE,"file_size":os.path.getsize(APPOINTMENTS_FILE) if os.path.exists(APPOINTMENTS_FILE) else 0}}) + '\n')
    # #endregion


def get_next_id(appointments: List[Appointment]) -> int:
    """Get the next available appointment ID."""
    if not appointments:
        return 1
    return max(apt.id for apt in appointments) + 1


@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify server is working."""
    # #region agent log
    try:
        log_path = r"c:\Users\revid\RAG\.cursor\debug.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"TEST","location":"app/backend.py:test_endpoint","message":"Test endpoint hit","data":{},"timestamp":int(time.time()*1000)}) + "\n")
    except Exception as e:
        print(f"[LOG ERROR] {e}")
    # #endregion
    return {"status": "ok", "message": "Server is working", "frontend_dir": FRONTEND_DIR}

@app.get("/")
async def root():
    """Root endpoint - serves the dashboard."""
    # #region agent log
    try:
        log_path = r"c:\Users\revid\RAG\.cursor\debug.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"app/backend.py:root","message":"Root route hit","data":{"frontend_dir":FRONTEND_DIR,"base_dir":BASE_DIR},"timestamp":int(time.time()*1000)}) + "\n")
    except Exception as e:
        print(f"[LOG ERROR] {e}")
    # #endregion
    dashboard_path = os.path.join(FRONTEND_DIR, "dashboard.html")
    dashboard_path = os.path.abspath(dashboard_path)
    # #region agent log
    try:
        log_path = r"c:\Users\revid\RAG\.cursor\debug.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"app/backend.py:root","message":"Dashboard path resolved","data":{"dashboard_path":dashboard_path,"exists":os.path.exists(dashboard_path)},"timestamp":int(time.time()*1000)}) + "\n")
    except Exception as e:
        print(f"[LOG ERROR] {e}")
    # #endregion
    if os.path.exists(dashboard_path):
        # #region agent log
        try:
            log_path = r"c:\Users\revid\RAG\.cursor\debug.log"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"app/backend.py:root","message":"Returning FileResponse","data":{"dashboard_path":dashboard_path},"timestamp":int(time.time()*1000)}) + "\n")
        except Exception as e:
            print(f"[LOG ERROR] {e}")
        # #endregion
        try:
            return FileResponse(dashboard_path, media_type="text/html")
        except Exception as e:
            # #region agent log
            try:
                log_path = r"c:\Users\revid\RAG\.cursor\debug.log"
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"F","location":"app/backend.py:root","message":"FileResponse exception","data":{"error":str(e),"dashboard_path":dashboard_path},"timestamp":int(time.time()*1000)}) + "\n")
            except:
                pass
            # #endregion
            print(f"[ERROR] FileResponse failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to serve dashboard: {str(e)}")
    # #region agent log
    try:
        log_path = r"c:\Users\revid\RAG\.cursor\debug.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"app/backend.py:root","message":"Dashboard file not found","data":{"dashboard_path":dashboard_path,"frontend_dir":FRONTEND_DIR,"base_dir":BASE_DIR},"timestamp":int(time.time()*1000)}) + "\n")
    except Exception as e:
        print(f"[LOG ERROR] {e}")
    # #endregion
    print(f"[ERROR] Dashboard not found at: {dashboard_path}")
    print(f"[ERROR] FRONTEND_DIR: {FRONTEND_DIR}")
    print(f"[ERROR] BASE_DIR: {BASE_DIR}")
    return {
        "message": "Appointment Booking API",
        "docs": "/docs",
        "chat": "/chat",
        "twilio_voice": "/twilio/voice/incoming",
        "error": f"Dashboard not found at {dashboard_path}",
        "frontend_dir": FRONTEND_DIR,
        "base_dir": BASE_DIR
    }


@app.get("/chat")
async def chat_page():
    """Serve the chat interface."""
    # #region agent log
    try:
        log_path = r"c:\Users\revid\RAG\.cursor\debug.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"app/backend.py:chat_page","message":"Chat route hit","data":{"frontend_dir":FRONTEND_DIR},"timestamp":int(time.time()*1000)}) + "\n")
    except Exception as e:
        print(f"[LOG ERROR] {e}")
    # #endregion
    chat_path = os.path.join(FRONTEND_DIR, "chat.html")
    chat_path = os.path.abspath(chat_path)
    # #region agent log
    try:
        log_path = r"c:\Users\revid\RAG\.cursor\debug.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"app/backend.py:chat_page","message":"Chat path resolved","data":{"chat_path":chat_path,"exists":os.path.exists(chat_path)},"timestamp":int(time.time()*1000)}) + "\n")
    except Exception as e:
        print(f"[LOG ERROR] {e}")
    # #endregion
    if os.path.exists(chat_path):
        # #region agent log
        try:
            log_path = r"c:\Users\revid\RAG\.cursor\debug.log"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"app/backend.py:chat_page","message":"Returning FileResponse for chat","data":{"chat_path":chat_path},"timestamp":int(time.time()*1000)}) + "\n")
        except Exception as e:
            print(f"[LOG ERROR] {e}")
        # #endregion
        return FileResponse(chat_path, media_type="text/html")
    # #region agent log
    try:
        log_path = r"c:\Users\revid\RAG\.cursor\debug.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"app/backend.py:chat_page","message":"Chat file not found","data":{"chat_path":chat_path,"frontend_dir":FRONTEND_DIR},"timestamp":int(time.time()*1000)}) + "\n")
    except Exception as e:
        print(f"[LOG ERROR] {e}")
    # #endregion
    print(f"[ERROR] Chat not found at: {chat_path}")
    print(f"[ERROR] FRONTEND_DIR: {FRONTEND_DIR}")
    raise HTTPException(status_code=404, detail=f"Chat interface not found at {chat_path}")


@app.get("/api/appointments", response_model=List[Appointment])
async def get_appointments():
    """Get all appointments."""
    # #region agent log
    try:
        log_path = r"c:\Users\revid\RAG\.cursor\debug.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"app/backend.py:get_appointments","message":"API appointments endpoint hit","data":{},"timestamp":int(time.time()*1000)}) + "\n")
    except Exception as e:
        print(f"[LOG ERROR] {e}")
    # #endregion
    appointments = load_appointments()
    # Sort by date and time (most recent first)
    appointments.sort(key=lambda x: (x.date, x.time), reverse=True)
    return appointments


@app.get("/api/appointments/{appointment_id}", response_model=Appointment)
async def get_appointment(appointment_id: int):
    """Get a specific appointment by ID."""
    appointments = load_appointments()
    appointment = next((apt for apt in appointments if apt.id == appointment_id), None)
    if not appointment:
        raise HTTPException(status_code=404, detail="Appointment not found")
    return appointment


@app.post("/api/appointments", response_model=Appointment)
async def create_appointment(appointment_request: AppointmentRequest):
    """Create a new appointment."""
    # #region agent log
    import json
    with open(r'c:\Users\revid\RAG\.cursor\debug.log', 'a', encoding='utf-8') as f:
        f.write(json.dumps({"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"backend.py:create_appointment:entry","message":"Received appointment request","data":{"name":appointment_request.name,"email":appointment_request.email,"date":appointment_request.date,"time":appointment_request.time,"service":appointment_request.service}}) + '\n')
    # #endregion
    appointments = load_appointments()
    
    # Validate date and time format
    try:
        datetime.strptime(appointment_request.date, "%Y-%m-%d")
        datetime.strptime(appointment_request.time, "%H:%M")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid date or time format. Use YYYY-MM-DD for date and HH:MM for time."
        )
    
    # Create new appointment
    new_appointment = Appointment(
        id=get_next_id(appointments),
        name=appointment_request.name,
        email=appointment_request.email,
        date=appointment_request.date,
        time=appointment_request.time,
        purpose=appointment_request.purpose,
        service=appointment_request.service,
        created_at=datetime.now().isoformat(),
        status="scheduled"
    )
    
    appointments.append(new_appointment)
    print(f"[API] POST /api/appointments - Created appointment #{new_appointment.id} for {new_appointment.name}")
    # #region agent log
    import json
    with open(r'c:\Users\revid\RAG\.cursor\debug.log', 'a', encoding='utf-8') as f:
        f.write(json.dumps({"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"backend.py:create_appointment:before_save","message":"About to save appointment","data":{"appointment_id":new_appointment.id,"total_appointments":len(appointments)}}) + '\n')
    # #endregion
    save_appointments(appointments)
    print(f"[API] POST /api/appointments - Saved {len(appointments)} total appointments to file")
    # #region agent log
    with open(r'c:\Users\revid\RAG\.cursor\debug.log', 'a', encoding='utf-8') as f:
        f.write(json.dumps({"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"backend.py:create_appointment:after_save","message":"Appointment saved","data":{"appointment_id":new_appointment.id}}) + '\n')
    # #endregion
    
    return new_appointment


@app.delete("/api/appointments/{appointment_id}")
async def delete_appointment(appointment_id: int):
    """Delete an appointment."""
    appointments = load_appointments()
    appointment = next((apt for apt in appointments if apt.id == appointment_id), None)
    
    if not appointment:
        raise HTTPException(status_code=404, detail="Appointment not found")
    
    appointments.remove(appointment)
    save_appointments(appointments)
    
    return {"message": "Appointment deleted successfully", "id": appointment_id}


@app.patch("/api/appointments/{appointment_id}", response_model=Appointment)
async def update_appointment_status(
    appointment_id: int,
    status: str = Query(..., description="New status: scheduled, completed, or cancelled")
):
    """Update appointment status."""
    if status not in ["scheduled", "completed", "cancelled"]:
        raise HTTPException(
            status_code=400,
            detail="Status must be one of: scheduled, completed, cancelled"
        )
    
    appointments = load_appointments()
    appointment = next((apt for apt in appointments if apt.id == appointment_id), None)
    
    if not appointment:
        raise HTTPException(status_code=404, detail="Appointment not found")
    
    appointment.status = status
    save_appointments(appointments)
    
    return appointment


@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_receptionist(chat_message: ChatMessage):
    """Chat endpoint - processes messages through the AI receptionist."""
    print(f"[CHAT] Received message: {chat_message.message[:50]}...")
    try:
        from app.ai_receptionist import create_receptionist_graph, ReceptionistState
        from langchain_core.messages import HumanMessage, AIMessage
        print("[CHAT] Successfully imported receptionist modules")
        
        # Use session_id to maintain conversation state per user
        session_id = chat_message.session_id or "default"
        
        # Get or create conversation state for this session
        if not hasattr(chat_with_receptionist, 'conversation_states'):
            chat_with_receptionist.conversation_states = {}
        
        if session_id not in chat_with_receptionist.conversation_states:
            chat_with_receptionist.conversation_states[session_id] = {
                "messages": [],
                "documents": [],
                "context": "",
                "intent": "",
                "routing_info": {},
                "conversation_stage": "greeting",
                "appointment_data": {}
            }
        
        state = chat_with_receptionist.conversation_states[session_id]
        
        # Create the receptionist graph (cached after first call)
        if not hasattr(chat_with_receptionist, 'graph_app'):
            print("[CHAT] Creating receptionist graph (first time)...")
            chat_with_receptionist.graph_app = create_receptionist_graph()
            print("[CHAT] Graph created successfully")
        app = chat_with_receptionist.graph_app
        
        # Add user message
        state["messages"].append(HumanMessage(content=chat_message.message))
        
        # Process through LangGraph
        print("[CHAT] Processing message through LangGraph...")
        state = app.invoke(state)
        print("[CHAT] LangGraph processing complete")
        
        # Get AI response
        ai_response = state["messages"][-1].content
        print(f"[CHAT] Generated response: {ai_response[:100]}...")
        
        # Update stored state
        chat_with_receptionist.conversation_states[session_id] = state
        
        # Check if appointment was booked
        appointment_booked = state.get("appointment_data", {}).get("booked", False)
        appointment_id = state.get("appointment_data", {}).get("appointment_id")
        
        print(f"[CHAT] Appointment status - booked: {appointment_booked}, id: {appointment_id}")
        print(f"[CHAT] Appointment data keys: {list(state.get('appointment_data', {}).keys())}")
        
        # Verify appointment exists in database if booked
        if appointment_booked and appointment_id:
            try:
                appointments = load_appointments()
                found = any(apt.id == appointment_id for apt in appointments)
                print(f"[CHAT] Verification - Appointment #{appointment_id} found in database: {found}")
                if not found:
                    print(f"[CHAT] WARNING: Appointment was marked as booked but not found in database!")
            except Exception as e:
                print(f"[CHAT] Error verifying appointment: {e}")
        
        response = ChatResponse(
            response=ai_response,
            session_id=session_id,
            appointment_booked=appointment_booked,
            appointment_id=appointment_id
        )
        print(f"[CHAT] Returning response - booked: {appointment_booked}, id: {appointment_id}")
        return response
        
    except Exception as e:
        import traceback
        error_msg = f"Error processing chat message: {str(e)}"
        print(f"[CHAT ERROR] {error_msg}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "appointment-booking-api"}


# Add Twilio Voice integration with Whisper if available
try:
    from integrations.twilio_voice import add_twilio_voice_routes
    add_twilio_voice_routes(app)
    print("[OK] Twilio Voice integration with Whisper enabled")
except ImportError:
    print("[INFO] Twilio Voice integration not available (integrations/twilio_voice.py not found)")
except Exception as e:
    print(f"[WARNING] Failed to load Twilio Voice integration: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

