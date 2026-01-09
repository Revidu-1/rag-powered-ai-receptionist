"""
Twilio Voice Integration with OpenAI Whisper for AI Salon Receptionist
Handles phone calls, uses Whisper for speech-to-text, and processes through AI receptionist
"""

from fastapi import FastAPI, Request, Form
from fastapi.responses import Response, PlainTextResponse
from pydantic import BaseModel
from typing import Optional
import requests
import os
from dotenv import load_dotenv
import json
from datetime import datetime
from twilio.twiml.voice_response import VoiceResponse, Gather
import logging

load_dotenv()

# Import the receptionist
from app.ai_receptionist import create_receptionist_graph, ReceptionistState
from langchain_core.messages import HumanMessage, AIMessage

# Twilio configuration
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")

# OpenAI configuration for Whisper
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Store conversation states per phone number
conversation_states = {}

# Cache the receptionist graph for better performance
_receptionist_graph_cache = None


def get_receptionist_graph():
    """Get or create the receptionist graph (cached for performance)."""
    global _receptionist_graph_cache
    if _receptionist_graph_cache is None:
        try:
            print("[TWILIO] Initializing receptionist graph...")
            _receptionist_graph_cache = create_receptionist_graph()
            print("[TWILIO] Receptionist graph initialized successfully")
        except Exception as e:
            print(f"[ERROR] Failed to initialize receptionist graph: {e}")
            import traceback
            traceback.print_exc()
            raise
    return _receptionist_graph_cache


class TwilioVoiceMessage(BaseModel):
    """Twilio voice call model"""
    CallSid: str
    From: str
    To: str
    CallStatus: Optional[str] = None


def get_or_create_conversation_state(phone_number: str) -> ReceptionistState:
    """Get or create conversation state for a phone number."""
    if phone_number not in conversation_states:
        conversation_states[phone_number] = {
            "messages": [],
            "documents": [],
            "context": "",
            "intent": "",
            "routing_info": {},
            "conversation_stage": "greeting",
            "appointment_data": {},
            "is_voice_call": True  # Mark as voice call for voice-friendly responses
        }
    else:
        # Ensure voice flag is set
        conversation_states[phone_number]["is_voice_call"] = True
    return conversation_states[phone_number]


def transcribe_audio_with_whisper(audio_url: str) -> str:
    """
    Transcribe audio using OpenAI Whisper API.
    
    Args:
        audio_url: URL of the audio file to transcribe
    
    Returns:
        str: Transcribed text
    """
    if not OPENAI_API_KEY:
        print("⚠ OpenAI API key not configured for Whisper")
        return ""
    
    try:
        # Download the audio file from Twilio
        audio_response = requests.get(audio_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=30)
        audio_response.raise_for_status()
        
        # Send to OpenAI Whisper API
        files = {'file': ('audio.wav', audio_response.content, 'audio/wav')}
        data = {'model': 'whisper-1'}
        headers = {'Authorization': f'Bearer {OPENAI_API_KEY}'}
        
        response = requests.post(
            'https://api.openai.com/v1/audio/transcriptions',
            headers=headers,
            files=files,
            data=data,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        transcript = result.get('text', '')
        print(f"✓ Whisper transcription: {transcript}")
        return transcript
        
    except Exception as e:
        print(f"✗ Error transcribing with Whisper: {e}")
        return ""


def text_to_speech_with_openai(text: str) -> Optional[str]:
    """
    Convert text to speech using OpenAI TTS API.
    
    Args:
        text: Text to convert to speech
    
    Returns:
        str: URL of the generated audio file, or None if error
    """
    if not OPENAI_API_KEY:
        print("⚠ OpenAI API key not configured for TTS")
        return None
    
    try:
        # Clean text for voice (remove markdown, emojis)
        import re
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.+?)\*', r'\1', text)  # Italic
        text = re.sub(r'`(.+?)`', r'\1', text)  # Code
        text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)  # Links
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Use OpenAI TTS API
        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': 'tts-1',  # or 'tts-1-hd' for higher quality
            'input': text,
            'voice': 'alloy',  # Options: alloy, echo, fable, onyx, nova, shimmer
            'response_format': 'mp3'
        }
        
        response = requests.post(
            'https://api.openai.com/v1/audio/speech',
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        
        # Save audio to a temporary file or return as base64
        # For now, we'll use Twilio's built-in TTS which is simpler
        # This function is here for future use if needed
        return None
        
    except Exception as e:
        print(f"✗ Error with OpenAI TTS: {e}")
        return None


def process_voice_message(phone_number: str, transcript: str) -> str:
    """
    Process transcribed voice message through the AI receptionist.
    
    Args:
        phone_number: Caller's phone number
        transcript: Transcribed text from speech
    
    Returns:
        str: Response text to be converted to speech
    """
    if not transcript or not transcript.strip():
        return "I didn't catch that. Could you please repeat?"
    
    # Get or create conversation state
    state = get_or_create_conversation_state(phone_number)
    
    # Get the receptionist graph
    app = get_receptionist_graph()
    
    # Add user message
    state["messages"].append(HumanMessage(content=transcript.strip()))
    
    print(f"[TWILIO] Processing voice message from {phone_number}: {transcript[:50]}...")
    
    # Process through LangGraph
    try:
        state = app.invoke(state)
    except Exception as e:
        print(f"✗ Error processing message through receptionist: {e}")
        return "I'm sorry, I encountered an error. Please try again."
    
    # Get AI response
    ai_response = state["messages"][-1].content if state["messages"] else "I'm sorry, I didn't understand that."
    
    # Update stored state
    conversation_states[phone_number] = state
    
    # Check if appointment was booked
    # Note: The confirmation is already included in the AI response via the system prompt,
    # but we ensure it's voice-friendly here as well (no # symbols, natural language)
    if state.get("appointment_data", {}).get("booked"):
        appointment_id = state.get("appointment_data", {}).get("appointment_id")
        # Use voice-friendly format (no # symbol, spell out numbers naturally)
        ai_response += f" Great news! Your appointment has been booked successfully. Your appointment number is {appointment_id}."
        ai_response += " You can view all appointments on our website."
    
    # Clean up response for voice (remove markdown, emojis, formatting, etc.)
    import re
    
    # Remove markdown headers (# ## ### etc.)
    ai_response = re.sub(r'^#{1,6}\s+', '', ai_response, flags=re.MULTILINE)
    
    # Remove markdown bold (**text** or __text__)
    ai_response = re.sub(r'\*\*(.+?)\*\*', r'\1', ai_response)
    ai_response = re.sub(r'__(.+?)__', r'\1', ai_response)
    
    # Remove markdown italic (*text* or _text_)
    ai_response = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\1', ai_response)
    ai_response = re.sub(r'(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', r'\1', ai_response)
    
    # Remove markdown code blocks and inline code
    ai_response = re.sub(r'```[\s\S]*?```', '', ai_response)
    ai_response = re.sub(r'`([^`]+)`', r'\1', ai_response)
    
    # Remove markdown links [text](url) -> text
    ai_response = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', ai_response)
    
    # Remove markdown lists (bullets and numbers)
    ai_response = re.sub(r'^\s*[-*+]\s+', '', ai_response, flags=re.MULTILINE)
    ai_response = re.sub(r'^\s*\d+\.\s+', '', ai_response, flags=re.MULTILINE)
    
    # Remove markdown horizontal rules
    ai_response = re.sub(r'^---+$', '', ai_response, flags=re.MULTILINE)
    
    # Remove markdown blockquotes
    ai_response = re.sub(r'^>\s+', '', ai_response, flags=re.MULTILINE)
    
    # Remove emojis and replace with text where appropriate
    emoji_map = {
        '✅': 'Success',
        '✓': 'Success',
        '❌': 'Error',
        '✗': 'Error',
        '💇‍♀️': '',
        '💅': '',
        '📞': '',
        '📱': '',
        '📅': '',
        '🕐': '',
        '📧': '',
        '⚠️': 'Warning',
        '🎉': '',
        '👍': '',
        '👎': '',
    }
    for emoji, replacement in emoji_map.items():
        ai_response = ai_response.replace(emoji, replacement)
    
    # Remove any remaining special markdown characters that might be read aloud
    ai_response = re.sub(r'[#*_`~]', '', ai_response)
    
    # Clean up extra whitespace
    ai_response = re.sub(r'\s+', ' ', ai_response)
    ai_response = re.sub(r'\n+', '. ', ai_response)  # Replace newlines with periods
    ai_response = ai_response.strip()
    
    # Remove any trailing periods/spaces that might cause awkward pauses
    ai_response = re.sub(r'\.{2,}', '.', ai_response)
    
    return ai_response


def add_twilio_voice_routes(app: FastAPI):
    """Add Twilio Voice webhook routes to FastAPI app."""
    
    @app.get("/twilio/voice/incoming")
    @app.post("/twilio/voice/incoming")
    async def twilio_voice_incoming(request: Request):
        """
        Handle incoming Twilio voice calls.
        This is called when someone calls your Twilio number.
        Supports both GET and POST (Twilio uses POST, but GET is useful for testing).
        """
        try:
            print(f"[TWILIO] Received {request.method} request to /twilio/voice/incoming")
            print(f"[TWILIO] Headers: {dict(request.headers)}")
            print(f"[TWILIO] URL: {request.url}")
            
            # Parse form data safely (Twilio sends form-urlencoded data)
            call_sid = ""
            from_number = ""
            to_number = ""
            
            try:
                if request.method == "POST":
                    form_data = await request.form()
                    call_sid = form_data.get("CallSid", "")
                    from_number = form_data.get("From", "")
                    to_number = form_data.get("To", "")
                    print(f"[TWILIO] Form data received - CallSid: {call_sid}, From: {from_number}, To: {to_number}")
                else:
                    # GET request (for testing)
                    call_sid = request.query_params.get("CallSid", "")
                    from_number = request.query_params.get("From", "")
                    to_number = request.query_params.get("To", "")
            except Exception as form_error:
                print(f"[ERROR] Failed to parse form/query data: {form_error}")
                import traceback
                traceback.print_exc()
                # Return valid TwiML even if form parsing fails
                response = VoiceResponse()
                response.say("I'm sorry, there was an error processing your call. Please try again later.", voice='alice')
                response.hangup()
                twiml_content = str(response)
                return Response(
                    content=twiml_content, 
                    media_type='application/xml; charset=utf-8',
                    headers={"Content-Type": "application/xml; charset=utf-8"}
                )
            
            print(f"[TWILIO] Incoming call from {from_number} to {to_number} (CallSid: {call_sid})")
            
            # Create TwiML response
            response = VoiceResponse()
            
            # Initial greeting
            greeting = "Hello! Welcome to our salon. I'm your AI receptionist. How can I help you today?"
            response.say(greeting, voice='alice', language='en-US')
            
            # Get the base URL from the request (for webhook callback)
            # More robust URL construction
            try:
                # Try to get from headers first (for reverse proxies like ngrok)
                host = request.headers.get("X-Forwarded-Host") or request.headers.get("Host")
                scheme = request.headers.get("X-Forwarded-Proto") or request.url.scheme
                
                if host:
                    base_url = f"{scheme}://{host}".rstrip('/')
                else:
                    # Fallback to request URL
                    base_url = str(request.base_url).rstrip('/')
            except Exception as url_error:
                print(f"[WARNING] Error constructing base URL: {url_error}, using fallback")
                # Last resort: use request URL components
                try:
                    base_url = f"{request.url.scheme}://{request.url.netloc}".rstrip('/')
                except:
                    # If all else fails, use a placeholder (Twilio will need the full URL configured)
                    base_url = "https://your-domain.com"  # This should be replaced with actual domain
                    print(f"[WARNING] Using placeholder base URL: {base_url}")
            
            process_url = f"{base_url}/twilio/voice/process"
            print(f"[TWILIO] Using process URL: {process_url}")
            
            # Gather speech input (using Twilio's speech recognition)
            gather = Gather(
                input='speech',
                action=process_url,
                method='POST',
                speech_timeout='auto',
                language='en-US',
                timeout=10
            )
            gather.say('Please speak your message.', voice='alice')
            response.append(gather)
            
            # If no input after gather, say goodbye and hangup
            response.say("Thank you for calling. Goodbye.", voice='alice')
            response.hangup()
            
            twiml_content = str(response)
            print(f"[TWILIO] Returning TwiML response (length: {len(twiml_content)})")
            print(f"[TWILIO] TwiML preview: {twiml_content[:200]}...")
            
            # Return with proper headers
            return Response(
                content=twiml_content, 
                media_type='application/xml; charset=utf-8',
                headers={
                    "Content-Type": "application/xml; charset=utf-8",
                    "Cache-Control": "no-cache"
                }
            )
            
        except Exception as e:
            print(f"[ERROR] Error handling incoming call: {e}")
            import traceback
            traceback.print_exc()
            try:
                response = VoiceResponse()
                response.say("I'm sorry, there was an error. Please try again later.", voice='alice')
                response.hangup()
                twiml_content = str(response)
                return Response(
                    content=twiml_content, 
                    media_type='application/xml; charset=utf-8',
                    headers={"Content-Type": "application/xml; charset=utf-8"}
                )
            except Exception as e2:
                print(f"[ERROR] Failed to create error response: {e2}")
                # Return minimal valid TwiML
                return Response(
                    content='<?xml version="1.0" encoding="UTF-8"?><Response><Say voice="alice">Error occurred. Goodbye.</Say><Hangup/></Response>', 
                    media_type='application/xml; charset=utf-8',
                    headers={"Content-Type": "application/xml; charset=utf-8"}
                )
    
    # Add a root POST handler in case webhook URL is set to root
    @app.post("/")
    async def root_post_handler(request: Request):
        """Handle POST to root - forward to incoming handler."""
        print("[TWILIO] Received POST to root path, forwarding to incoming handler")
        return await twilio_voice_incoming(request)
    
    @app.get("/twilio/voice/process")
    @app.post("/twilio/voice/process")
    async def twilio_voice_process(request: Request):
        """
        Process speech input from Twilio voice call.
        Uses Twilio's built-in speech recognition or Whisper for transcription.
        """
        try:
            # Parse form data safely
            call_sid = ""
            from_number = ""
            speech_result = ""
            recording_url = ""
            
            try:
                if request.method == "POST":
                    form_data = await request.form()
                    call_sid = form_data.get("CallSid", "")
                    from_number = form_data.get("From", "")
                    speech_result = form_data.get("SpeechResult", "")
                    recording_url = form_data.get("RecordingUrl", "")
                else:
                    # GET request (for testing)
                    call_sid = request.query_params.get("CallSid", "")
                    from_number = request.query_params.get("From", "")
                    speech_result = request.query_params.get("SpeechResult", "")
                    recording_url = request.query_params.get("RecordingUrl", "")
            except Exception as form_error:
                print(f"[ERROR] Failed to parse form/query data: {form_error}")
                import traceback
                traceback.print_exc()
                response = VoiceResponse()
                response.say("I'm sorry, there was an error processing your message. Please try again.", voice='alice')
                response.hangup()
                return Response(
                    content=str(response), 
                    media_type='application/xml; charset=utf-8',
                    headers={"Content-Type": "application/xml; charset=utf-8"}
                )
            
            print(f"[TWILIO] Processing speech from {from_number}: {speech_result[:50] if speech_result else 'No speech result'}...")
            
            # Create TwiML response
            response = VoiceResponse()
            
            # Get transcript - use Twilio's speech result or Whisper if available
            transcript = speech_result
            
            # If we have a recording URL and no speech result, use Whisper
            if recording_url and not transcript:
                print(f"[TWILIO] Using Whisper to transcribe recording: {recording_url}")
                try:
                    transcript = transcribe_audio_with_whisper(recording_url)
                except Exception as whisper_error:
                    print(f"[ERROR] Whisper transcription failed: {whisper_error}")
                    transcript = ""
            
            # Helper function to get base URL
            def get_base_url():
                try:
                    host = request.headers.get("X-Forwarded-Host") or request.headers.get("Host")
                    scheme = request.headers.get("X-Forwarded-Proto") or request.url.scheme
                    if host:
                        return f"{scheme}://{host}".rstrip('/')
                    return str(request.base_url).rstrip('/')
                except:
                    try:
                        return f"{request.url.scheme}://{request.url.netloc}".rstrip('/')
                    except:
                        return "https://your-domain.com"  # Placeholder
            
            if not transcript or not transcript.strip():
                # No transcript - ask to repeat
                process_url = f"{get_base_url()}/twilio/voice/process"
                response.say("I didn't catch that. Could you please repeat?", voice='alice')
                gather = Gather(
                    input='speech',
                    action=process_url,
                    method='POST',
                    speech_timeout='auto',
                    language='en-US'
                )
                gather.say('Please speak your message.', voice='alice')
                response.append(gather)
                return Response(
                    content=str(response), 
                    media_type='application/xml; charset=utf-8',
                    headers={
                        "Content-Type": "application/xml; charset=utf-8",
                        "Cache-Control": "no-cache"
                    }
                )
            
            # Process through AI receptionist
            try:
                ai_response = process_voice_message(from_number, transcript)
                if not ai_response or not ai_response.strip():
                    ai_response = "I'm sorry, I didn't understand that. Could you please repeat?"
            except Exception as process_error:
                print(f"[ERROR] Error processing voice message: {process_error}")
                import traceback
                traceback.print_exc()
                ai_response = "I'm sorry, I encountered an error processing your request. Please try again."
            
            # Speak the response (limit length for voice)
            if len(ai_response) > 500:
                ai_response = ai_response[:500] + "..."
            
            response.say(ai_response, voice='alice', language='en-US')
            
            # Continue conversation - gather next input (silently, let AI response handle the flow)
            process_url = f"{get_base_url()}/twilio/voice/process"
            gather = Gather(
                input='speech',
                action=process_url,
                method='POST',
                speech_timeout='auto',
                language='en-US',
                timeout=10  # Wait 10 seconds for user input
            )
            # Don't add any prompt here - let the AI's natural response guide the conversation
            response.append(gather)
            
            # If no input after timeout, end call gracefully
            response.say("Thank you for calling. Have a great day!", voice='alice')
            response.hangup()
            
            twiml_content = str(response)
            print(f"[TWILIO] Returning TwiML response (length: {len(twiml_content)})")
            
            # Return with proper headers
            return Response(
                content=twiml_content, 
                media_type='application/xml; charset=utf-8',
                headers={
                    "Content-Type": "application/xml; charset=utf-8",
                    "Cache-Control": "no-cache"
                }
            )
            
        except Exception as e:
            print(f"[ERROR] Error processing voice call: {e}")
            import traceback
            traceback.print_exc()
            try:
                response = VoiceResponse()
                response.say("I'm sorry, I encountered an error. Please try again later.", voice='alice')
                response.hangup()
                return Response(
                    content=str(response), 
                    media_type='application/xml; charset=utf-8',
                    headers={"Content-Type": "application/xml; charset=utf-8"}
                )
            except Exception as e2:
                print(f"[ERROR] Failed to create error response: {e2}")
                # Return minimal valid TwiML
                return Response(
                    content='<?xml version="1.0" encoding="UTF-8"?><Response><Say voice="alice">Error occurred. Goodbye.</Say><Hangup/></Response>', 
                    media_type='application/xml'
                )
    
    @app.get("/twilio/voice/status")
    async def twilio_voice_status():
        """Check Twilio Voice integration status."""
        twilio_configured = bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN)
        openai_configured = bool(OPENAI_API_KEY)
        active_conversations = len(conversation_states)
        
        # Test receptionist graph initialization
        receptionist_working = False
        try:
            graph = get_receptionist_graph()
            receptionist_working = graph is not None
        except Exception as e:
            print(f"[ERROR] Receptionist graph test failed: {e}")
        
        return {
            "twilio_configured": twilio_configured,
            "openai_whisper_configured": openai_configured,
            "active_conversations": active_conversations,
            "phone_number": TWILIO_PHONE_NUMBER if TWILIO_PHONE_NUMBER else "Not configured",
            "receptionist_working": receptionist_working
        }
    
    @app.get("/twilio/voice/test")
    async def twilio_voice_test():
        """Simple test endpoint to verify the server is responding."""
        return {
            "status": "ok",
            "message": "Twilio Voice integration is running",
            "endpoints": {
                "incoming": "/twilio/voice/incoming",
                "process": "/twilio/voice/process",
                "status": "/twilio/voice/status"
            }
        }
    
    @app.get("/twilio/voice/simple")
    @app.post("/twilio/voice/simple")
    async def twilio_voice_simple():
        """
        Minimal TwiML endpoint for testing webhook connectivity.
        Returns a simple TwiML response that just says hello and hangs up.
        Use this to verify Twilio can reach your webhook.
        """
        response = VoiceResponse()
        response.say("Hello, this is a test. The webhook is working correctly.", voice='alice')
        response.hangup()
        
        twiml_content = str(response)
        print(f"[TWILIO TEST] Simple endpoint called, returning: {twiml_content}")
        
        return Response(
            content=twiml_content,
            media_type='application/xml; charset=utf-8',
            headers={
                "Content-Type": "application/xml; charset=utf-8",
                "Cache-Control": "no-cache"
            }
        )


# Standalone Twilio Voice server (alternative to integrating with main backend)
def create_twilio_voice_server():
    """Create a standalone FastAPI server for Twilio Voice integration."""
    app = FastAPI(title="Twilio Voice Receptionist API")
    
    # Add CORS
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add Twilio Voice routes
    add_twilio_voice_routes(app)
    
    return app


if __name__ == "__main__":
    import uvicorn
    print("Starting Twilio Voice Receptionist Server...")
    print("Webhook URL: http://your-domain.com/twilio/voice/incoming")
    print("Status: http://localhost:8003/twilio/voice/status")
    app = create_twilio_voice_server()
    uvicorn.run(app, host="0.0.0.0", port=8003)


