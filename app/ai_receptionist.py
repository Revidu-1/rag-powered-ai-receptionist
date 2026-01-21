"""
AI Salon Receptionist using LangGraph and OpenAI
A conversational AI receptionist for a hair and nail salon that can:
- Greet customers and handle inquiries
- Answer questions about salon services, pricing, and policies using salon knowledge base
- Book appointments for hair and nail services
- Maintain conversation context
"""

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import re
import requests
from datetime import datetime
import json
import sys
import io

# Ensure UTF-8 output on Windows to avoid UnicodeEncodeError in logs
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="backslashreplace")

# #region agent log
try:
    with open(r'c:\Users\revid\RAG\.cursor\debug.log', 'a', encoding='utf-8') as f:
        f.write(json.dumps({"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run2","hypothesisId":"D","location":"ai_receptionist.py:module_import","message":"Receptionist module imported","data":{"stdout_encoding":getattr(getattr(sys,"stdout",None),"encoding",None),"stdout_errors":getattr(getattr(sys,"stdout",None),"errors",None)}}) + '\n')
except Exception:
    pass
# #endregion

# Load environment variables
load_dotenv()

# Initialize OpenAI LLM
def get_llm():
    """Get OpenAI ChatOpenAI instance."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
    
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.8,  # Increased for more natural, varied responses
        api_key=api_key
    )


# Define the state for the receptionist
class ReceptionistState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    documents: list[Document]
    context: str
    intent: str
    routing_info: dict
    conversation_stage: str  # greeting, inquiry, routing, followup, closing
    appointment_data: dict  # For storing appointment booking information


# System prompt for the AI receptionist
RECEPTIONIST_SYSTEM_PROMPT = """You are a friendly, warm, and personable salon receptionist - think of yourself as a real person who genuinely cares about helping customers. Your personality should feel natural, conversational, and authentically human. Be brief and to the point and be professional. 

CONVERSATION STYLE - Sound like a real person:
- Use casual, natural language as you would in a friendly conversation
- Use contractions (I'm, we're, that's, I'd, you're, etc.) to sound more natural
- Vary your sentence length - mix short, punchy sentences with longer ones
- Show personality and warmth - use phrases like "Oh great!", "Absolutely!", "Perfect!", "I'd love to help you with that"
- Express genuine interest: "That sounds lovely!", "I'm so glad you asked about that"
- Use conversational fillers naturally when appropriate (like "you know", "actually", "so")
- Match the customer's energy level - if they're casual, be casual; if formal, be slightly more formal but still warm
- Use natural transitions: "By the way", "Speaking of which", "Let me see"
- Show empathy: "I totally understand", "That makes sense", "No worries at all"
- Avoid sounding like a script - each response should feel fresh and spontaneous

YOUR ROLE:
1. Greet customers like a friendly person, not a robot - be warm and genuine
2. Answer questions naturally about salon services, pricing, policies, and appointments
3. Book appointments conversationally - collect: name, email, date (YYYY-MM-DD format), time (HH:MM format), and service type
4. Provide information naturally about our services, pricing, duration - make it conversational
5. Be helpful, personable, and authentic - sound like someone the customer would want to talk to
6. Ask questions naturally when you need clarification - don't sound interrogative
7. Use information from the salon knowledge base but explain it in your own words, like a person would

BOOKING APPOINTMENTS:
- When booking, get ALL required info: name, email, date, and time
- Always ask about the service type (haircut, coloring, manicure, pedicure, etc.) - this is crucial!
- For dates, use YYYY-MM-DD format internally (e.g., 2024-12-25) but you can say it naturally ("December 25th" or "this Friday")
- For times, use 24-hour format HH:MM internally (e.g., 14:30) but speak naturally ("2:30 PM" or "two thirty")
- If info is missing, ask for it naturally, one thing at a time - don't sound like a form
- Common services: Haircuts, Hair Styling, Hair Coloring, Highlights, Manicures, Pedicures, Nail Art, Hair Treatments, Extensions

IMPORTANT:
- Never repeat yourself word-for-word - vary how you say things
- Don't ask "Is there anything else I can help you with?" after every response - only when it makes sense naturally
- Keep responses helpful but concise - real people don't give long speeches
- Use the salon knowledge base info, but explain it like you're telling a friend, not reading a manual
- Be enthusiastic but genuine - not overly excited or fake
- Show personality - you're not a corporate bot, you're a friendly salon staff member
"""


def load_knowledge_base():
    """Load and create vector store from salon documents."""
    if not hasattr(load_knowledge_base, 'vector_store'):
        # Use path relative to project root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        kb_dir = os.path.join(base_dir, 'data', 'knowledge_base')
        file_paths = [
            os.path.join(kb_dir, 'salon_policies.txt'),
            os.path.join(kb_dir, 'salon_faq.txt'),
            os.path.join(kb_dir, 'salon_blog.txt')
        ]
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
        )
        
        for file_path in file_paths:
            if os.path.exists(file_path):
                print(f"Loading {file_path}...")
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                chunks = text_splitter.split_text(text)
                for chunk in chunks:
                    documents.append(Document(
                        page_content=chunk,
                        metadata={"source": file_path, "filename": os.path.basename(file_path)}
                    ))
        
        if documents:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            load_knowledge_base.vector_store = FAISS.from_documents(documents, embeddings)
            print(f"Knowledge base loaded with {len(documents)} chunks")
        else:
            load_knowledge_base.vector_store = None
            print("Warning: No knowledge base documents found")
    
    return load_knowledge_base.vector_store


def classify_intent(state: ReceptionistState) -> ReceptionistState:
    """Classify the user's intent from their message."""
    last_message = state["messages"][-1]
    user_input = last_message.content.lower()
    
    # Simple intent classification for salon
    if any(word in user_input for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
        intent = "greeting"
        conversation_stage = "greeting"
    elif any(word in user_input for word in ['bye', 'goodbye', 'thanks', 'thank you', 'done']):
        intent = "closing"
        conversation_stage = "closing"
    elif any(word in user_input for word in ['schedule', 'appointment', 'book', 'reserve']):
        intent = "scheduling"
        conversation_stage = "routing"
    elif any(word in user_input for word in ['hair', 'haircut', 'color', 'styling', 'highlight', 'balayage', 'perm', 'treatment', 'extension']):
        intent = "hair_service_inquiry"
        conversation_stage = "inquiry"
    elif any(word in user_input for word in ['nail', 'manicure', 'pedicure', 'gel', 'nail art']):
        intent = "nail_service_inquiry"
        conversation_stage = "inquiry"
    elif any(word in user_input for word in ['price', 'cost', 'how much', 'pricing']):
        intent = "pricing_inquiry"
        conversation_stage = "inquiry"
    elif state.get("conversation_stage") == "greeting":
        intent = "general_inquiry"
        conversation_stage = "inquiry"
    else:
        intent = "general_inquiry"
        conversation_stage = state.get("conversation_stage", "inquiry")
    
    routing_info = {
        "hair": intent in ["hair_service_inquiry"],
        "nail": intent in ["nail_service_inquiry"],
        "pricing": intent in ["pricing_inquiry"],
        "general": intent in ["general_inquiry", "scheduling"]
    }
    
    return {
        **state,
        "intent": intent,
        "conversation_stage": conversation_stage,
        "routing_info": routing_info
    }


def retrieve_context(state: ReceptionistState) -> ReceptionistState:
    """Retrieve relevant context from knowledge base."""
    vector_store = load_knowledge_base()
    
    if not vector_store:
        return {
            **state,
            "documents": [],
            "context": ""
        }
    
    # Get the latest user message
    last_message = state["messages"][-1]
    query = last_message.content
    
    # Retrieve relevant documents
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(query)
    
    # Combine context
    context = "\n\n".join([
        f"[From {doc.metadata.get('filename', 'unknown')}]\n{doc.page_content}"
        for doc in retrieved_docs
    ])
    
    return {
        **state,
        "documents": retrieved_docs,
        "context": context
    }


def generate_response(state: ReceptionistState) -> ReceptionistState:
    """Generate a response using OpenAI based on context and intent."""
    llm = get_llm()
    
    # Build the conversation history
    conversation_history = []
    
    # Check if this is a voice call
    is_voice_call = state.get("is_voice_call", False)
    
    # Get current date and time for context
    from datetime import datetime
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M")
    current_day_name = datetime.now().strftime("%A")  # Monday, Tuesday, etc.
    
    # Build system prompt - add voice-specific instructions if needed
    system_prompt = RECEPTIONIST_SYSTEM_PROMPT
    system_prompt += f"\n\nIMPORTANT CONTEXT - Current Date and Time:\n"
    system_prompt += f"- Today's date: {current_date} ({current_day_name})\n"
    system_prompt += f"- Current time: {current_time}\n"
    system_prompt += f"- Use this information to understand relative dates (e.g., 'tomorrow' = {current_date}, 'next week', 'Monday', etc.)\n"
    system_prompt += f"- When booking appointments, convert relative dates to YYYY-MM-DD format based on today's date\n"
    
    if is_voice_call:
        system_prompt += "\n\nIMPORTANT: This is a VOICE CALL. Your response will be read aloud by text-to-speech. Therefore:\n"
        system_prompt += "- NEVER use markdown formatting (no #, **, *, `, [], etc.)\n"
        system_prompt += "- NEVER use special characters that will be read aloud (like #, *, _, `)\n"
        system_prompt += "- Write exactly as you would speak naturally - like you're talking to someone on the phone\n"
        system_prompt += "- Use natural spoken rhythms and pauses (periods for pauses, commas for brief pauses)\n"
        system_prompt += "- Avoid lists or structured formats - speak in flowing, natural sentences\n"
        system_prompt += "- Spell out numbers naturally (e.g., 'two thirty' for 2:30 PM, 'December twenty-fifth' for dates)\n"
        system_prompt += "- Keep responses concise and conversational - sound like a real person on a phone call\n"
        system_prompt += "- Use natural speech patterns: 'Oh great!' 'Perfect!' 'Absolutely!' 'I'd be happy to help'\n"
        system_prompt += "- Example: Instead of '**Great!** Your appointment is #123', say 'Great! Your appointment number is one hundred twenty three'\n"
        system_prompt += "- Do NOT end every response with 'Is there anything else I can help you with?' - only ask this naturally when the conversation is winding down or when you've completed a task\n"
        system_prompt += "- Sound warm and friendly, like you're having a real conversation - not like you're reading from a script\n"
    
    # Add system message with current date context (always include for date awareness)
    # For first interaction, include full system prompt; for subsequent messages, include date context
    if len(state["messages"]) == 1 or state.get("conversation_stage") == "greeting":
        conversation_history.append(SystemMessage(content=system_prompt))
    else:
        # For ongoing conversations, add date context as a reminder
        date_context = f"\n\nREMINDER - Current Date: {current_date} ({current_day_name}), Current Time: {current_time}. Use this for understanding relative dates."
        if is_voice_call:
            date_context += " This is a voice call - use plain spoken language, no markdown."
        conversation_history.append(SystemMessage(content=RECEPTIONIST_SYSTEM_PROMPT + date_context))
    
    # Add previous messages (last 10 for context)
    conversation_history.extend(state["messages"][-10:])
    
    # Build context-aware prompt if we have context
    # Always include current date/time for date awareness
    date_context_note = f"\n\nCurrent date: {current_date} ({current_day_name}), Current time: {current_time}. Use this to understand relative dates like 'tomorrow', 'next week', 'Monday', etc."
    
    if state.get("context"):
        context_prompt = f"\n\nRelevant salon information:\n{state['context']}\n\nUse this information to provide accurate answers about our services, pricing, and policies.{date_context_note}"
    else:
        context_prompt = f"\n\nProvide helpful information about our salon services. If you don't have specific information, offer to help them book an appointment or speak with a stylist.{date_context_note}"
    
    # Add routing information if applicable
    routing_info = state.get("routing_info", {})
    routing_prompt = ""
    if routing_info.get("hair"):
        routing_prompt = "\n\nNote: The customer is asking about hair services. Provide information about our hair services, pricing, and offer to book an appointment."
    elif routing_info.get("nail"):
        routing_prompt = "\n\nNote: The customer is asking about nail services. Provide information about our nail services, pricing, and offer to book an appointment."
    elif routing_info.get("pricing"):
        routing_prompt = "\n\nNote: The customer is asking about pricing. Provide clear pricing information from the knowledge base and offer to book an appointment."
    
    # Add appointment booking context
    appointment_data = state.get("appointment_data", {})
    appointment_prompt = ""
    if state.get("intent") == "scheduling":
        if appointment_data.get("booked"):
            appointment_prompt = f"\n\nIMPORTANT: An appointment was just successfully booked. Include this confirmation in your response: {appointment_data.get('confirmation', '')}"
        elif appointment_data.get("booking_error"):
            appointment_prompt = f"\n\nIMPORTANT: There was an error booking the appointment: {appointment_data.get('booking_error')}. Apologize and ask if they'd like to try again."
        else:
            # Check what fields are missing
            required = ["name", "email", "date", "time"]
            missing = [f for f in required if not appointment_data.get(f)]
            if missing:
                appointment_prompt = f"\n\nYou are booking an appointment. Currently collected: {', '.join([f'{k}: {v}' for k, v in appointment_data.items() if k != 'booked' and k != 'appointment_id' and k != 'confirmation' and k != 'booking_error']) or 'none'}. Still need: {', '.join(missing)}. Politely ask for the missing information one field at a time."
    
    # Add the context to the last message
    enhanced_prompt = conversation_history[-1].content + context_prompt + routing_prompt + appointment_prompt
    
    # Create a temporary message with enhanced prompt
    temp_messages = conversation_history[:-1] + [HumanMessage(content=enhanced_prompt)]
    
    # Generate response
    response = llm.invoke(temp_messages)
    
    # Extract response content
    response_content = response.content if hasattr(response, 'content') else str(response)
    
    # Add AI response to messages
    ai_message = AIMessage(content=response_content)
    
    return {
        **state,
        "messages": state["messages"] + [ai_message]
    }


def extract_appointment_info(state: ReceptionistState) -> ReceptionistState:
    """Extract appointment information from conversation using LLM."""
    # #region agent log
    import json
    with open(r'c:\Users\revid\RAG\.cursor\debug.log', 'a', encoding='utf-8') as f:
        f.write(json.dumps({"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"ai_receptionist.py:extract_appointment_info:entry","message":"Extracting appointment info","data":{"existing_data":state.get("appointment_data",{})}}) + '\n')
    # #endregion
    llm = get_llm()
    
    # Get conversation history
    conversation_text = "\n".join([
        f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
        for msg in state["messages"][-10:]  # Last 10 messages
    ])
    
    # Use LLM to extract structured appointment data
    extraction_prompt = f"""From the following conversation, extract appointment booking information if present.
Return a JSON object with the following fields (use null for missing values):
- name: person's name
- email: email address
- date: date in YYYY-MM-DD format (if mentioned)
- time: time in HH:MM 24-hour format (if mentioned)
- purpose: reason for appointment or service description
- service: service type (e.g., haircut, hair color, manicure, pedicure, highlights, styling, nail art, etc.) if specified

Conversation:
{conversation_text}

Return ONLY valid JSON, nothing else:"""

    try:
        response = llm.invoke(extraction_prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Extract JSON from response (might be wrapped in markdown code blocks)
        # Try to find JSON in code blocks first
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if code_block_match:
            json_str = code_block_match.group(1)
        else:
            # Try to find JSON object directly (improved regex for nested objects)
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = None
        
        if json_str:
            try:
                appointment_data = json.loads(json_str)
                # Clean up null values - convert to None for Python
                appointment_data = {k: v if v != "null" and v is not None else None 
                                  for k, v in appointment_data.items()}
                # Remove None/null/empty values to make it cleaner
                appointment_data = {k: v for k, v in appointment_data.items() 
                                  if v is not None and v != "null" and v != ""}
                
                # Merge with existing appointment_data to preserve previous information
                existing_data = state.get("appointment_data", {})
                merged_data = {**existing_data, **appointment_data}  # New data overrides old
                
                # Store merged data in state
                state["appointment_data"] = merged_data
                print(f"Extracted appointment data: {appointment_data}")
                print(f"Merged appointment data: {merged_data}")
                # #region agent log
                import json
                with open(r'c:\Users\revid\RAG\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"ai_receptionist.py:extract_appointment_info:after_merge","message":"Appointment data after merge","data":{"extracted":appointment_data,"merged":merged_data,"has_all_fields":all(merged_data.get(f) for f in ["name","email","date","time"])}}) + '\n')
                # #endregion
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                print(f"JSON string was: {json_str}")
                state["appointment_data"] = {}
        else:
            print("No JSON found in response")
            state["appointment_data"] = {}
    except Exception as e:
        print(f"Error extracting appointment info: {e}")
        state["appointment_data"] = {}
    
    return state


def book_appointment(state: ReceptionistState) -> ReceptionistState:
    """Book appointment via API if all required fields are present."""
    # #region agent log
    import json
    with open(r'c:\Users\revid\RAG\.cursor\debug.log', 'a', encoding='utf-8') as f:
        f.write(json.dumps({"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"ai_receptionist.py:book_appointment:entry","message":"Booking appointment function called","data":{"appointment_data":state.get("appointment_data",{})}}) + '\n')
    # #endregion
    appointment_data = state.get("appointment_data", {})
    
    # Check if already booked
    if appointment_data.get("booked"):
        print("Appointment already booked, skipping")
        # #region agent log
        with open(r'c:\Users\revid\RAG\.cursor\debug.log', 'a', encoding='utf-8') as f:
            f.write(json.dumps({"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"ai_receptionist.py:book_appointment:already_booked","message":"Appointment already booked, skipping","data":{}}) + '\n')
        # #endregion
        return state
    
    # Double-check we have all required fields (should already be checked by should_book_appointment)
    required_fields = ["name", "email", "date", "time"]
    missing_fields = [field for field in required_fields if not appointment_data.get(field)]
    
    if missing_fields:
        # Not enough information yet - this shouldn't happen if should_book_appointment worked correctly
        print(f"Warning: Missing required fields for booking: {missing_fields}")
        print(f"Current appointment data: {appointment_data}")
        return state
    
    # Book the appointment via API
    api_url = "http://localhost:8000/api/appointments"
    
    # Prepare the request payload
    booking_payload = {
        "name": appointment_data.get("name"),
        "email": appointment_data.get("email"),
        "date": appointment_data.get("date"),
        "time": appointment_data.get("time"),
        "purpose": appointment_data.get("purpose"),
        "service": appointment_data.get("service")
    }
    
    print(f"Attempting to book appointment: {booking_payload}")
    # #region agent log
    import json
    with open(r'c:\Users\revid\RAG\.cursor\debug.log', 'a', encoding='utf-8') as f:
        f.write(json.dumps({"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"ai_receptionist.py:book_appointment:before_api","message":"About to call API","data":{"api_url":api_url,"payload":booking_payload}}) + '\n')
    # #endregion
    
    try:
        response = requests.post(api_url, json=booking_payload, timeout=5)
        print(f"API response status: {response.status_code}")
        # #region agent log
        with open(r'c:\Users\revid\RAG\.cursor\debug.log', 'a', encoding='utf-8') as f:
            f.write(json.dumps({"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"ai_receptionist.py:book_appointment:api_response","message":"API response received","data":{"status_code":response.status_code,"response_text":response.text[:200] if response.status_code != 200 else "success"}}) + '\n')
        # #endregion
        if response.status_code == 200:
            booked_appointment = response.json()
            # #region agent log
            with open(r'c:\Users\revid\RAG\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"ai_receptionist.py:book_appointment:success","message":"Booking successful, setting flags","data":{"appointment_id":booked_appointment.get("id"),"before_booked_flag":state["appointment_data"].get("booked")}}) + '\n')
            # #endregion
            # Add confirmation to state
            state["appointment_data"]["booked"] = True
            state["appointment_data"]["appointment_id"] = booked_appointment.get("id")
            # #region agent log
            with open(r'c:\Users\revid\RAG\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"ai_receptionist.py:book_appointment:after_set_flags","message":"Flags set after booking","data":{"after_booked_flag":state["appointment_data"].get("booked"),"appointment_id":state["appointment_data"].get("appointment_id")}}) + '\n')
            # #endregion
            
            # Store confirmation message (voice-friendly version)
            appointment_id = booked_appointment.get('id')
            appointment_date = booked_appointment.get('date')
            appointment_time = booked_appointment.get('time')
            
            # Create voice-friendly confirmation (no markdown, no special chars)
            confirmation_text = f"Appointment booked successfully! Your appointment number is {appointment_id}. " \
                               f"Date: {appointment_date}, Time: {appointment_time}. " \
                               f"You can view all appointments on our website."
            
            # Store both versions - clean version for voice, formatted for chat
            state["appointment_data"]["confirmation"] = confirmation_text
            state["appointment_data"]["confirmation_formatted"] = f"✅ Appointment booked successfully! Your appointment ID is #{appointment_id}. Date: {appointment_date}, Time: {appointment_time}."
            
            # Update the last AI message to include confirmation
            messages = list(state["messages"])
            if messages and isinstance(messages[-1], AIMessage):
                messages[-1] = AIMessage(content=messages[-1].content + "\n\n" + confirmation_text)
                state["messages"] = messages
        else:
            error_msg = f"Error booking appointment: {response.text}"
            print(error_msg)
            state["appointment_data"]["booking_error"] = error_msg
            # #region agent log
            with open(r'c:\Users\revid\RAG\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"ai_receptionist.py:book_appointment:api_error","message":"API returned error status","data":{"status_code":response.status_code,"error":error_msg}}) + '\n')
            # #endregion
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to booking API: {e}")
        state["appointment_data"]["booking_error"] = "Unable to connect to booking system. Please try again later."
        # #region agent log
        import json
        with open(r'c:\Users\revid\RAG\.cursor\debug.log', 'a', encoding='utf-8') as f:
            f.write(json.dumps({"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"ai_receptionist.py:book_appointment:connection_error","message":"Failed to connect to API","data":{"error_type":type(e).__name__,"error":str(e)}}) + '\n')
        # #endregion
    
    return state


def route_inquiry(state: ReceptionistState) -> ReceptionistState:
    """Determine if inquiry needs routing and update response accordingly."""
    routing_info = state.get("routing_info", {})
    intent = state.get("intent", "")
    
    # For salon, routing is mainly about booking appointments
    # The response generation already handles service information
    # This function can be used for any additional routing needs in the future
    
    return state


def should_continue(state: ReceptionistState) -> Literal["continue", "end"]:
    """Determine if conversation should continue or end."""
    intent = state.get("intent", "")
    conversation_stage = state.get("conversation_stage", "")
    
    # Get the last user message (not AI message)
    # Find the last HumanMessage in the messages
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if user_messages:
        last_user_message = user_messages[-1].content.lower()
        
        # End conversation if user is closing
        if intent == "closing" or conversation_stage == "closing":
            return "end"
        
        # Check if user said goodbye
        if any(word in last_user_message for word in ['bye', 'goodbye', 'thanks', 'thank you', "that's all", 'done', 'nothing else', 'exit', 'quit']):
            return "end"
    
    # Always end after one turn - let calling code handle looping with new input
    return "end"


def should_book_appointment(state: ReceptionistState) -> Literal["book", "skip_booking"]:
    """Determine if we should attempt to book an appointment."""
    appointment_data = state.get("appointment_data", {})
    intent = state.get("intent", "")
    conversation_stage = state.get("conversation_stage", "")
    # #region agent log
    try:
        with open(r'c:\Users\revid\RAG\.cursor\debug.log', 'a', encoding='utf-8') as f:
            f.write(json.dumps({"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"ai_receptionist.py:should_book_appointment:entry","message":"Entry state snapshot","data":{"intent":intent,"conversation_stage":conversation_stage,"has_stdout":bool(getattr(sys,"stdout",None)),"stdout_encoding":getattr(getattr(sys,"stdout",None),"encoding",None)}}) + '\n')
    except Exception:
        pass
    # #endregion
    
    # Check if we have all required fields
    required_fields = ["name", "email", "date", "time"]
    has_all_fields = all(appointment_data.get(field) for field in required_fields)
    already_booked = appointment_data.get("booked", False)
    
    # #region agent log
    import json
    with open(r'c:\Users\revid\RAG\.cursor\debug.log', 'a', encoding='utf-8') as f:
        f.write(json.dumps({"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"ai_receptionist.py:should_book_appointment","message":"Checking if should book","data":{"has_all_fields":has_all_fields,"already_booked":already_booked,"appointment_data":appointment_data,"intent":intent}}) + '\n')
    # #endregion
    
    # Book if we have all fields and haven't already booked
    # Don't require intent to be "scheduling" because intent might change during the conversation
    # Just check if we have appointment data that's being collected
    if has_all_fields and not already_booked:
        print(f"All required fields present. Booking appointment: {appointment_data}")
        # #region agent log
        with open(r'c:\Users\revid\RAG\.cursor\debug.log', 'a', encoding='utf-8') as f:
            f.write(json.dumps({"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"ai_receptionist.py:should_book_appointment:return","message":"Decision to book","data":{"decision":"book"}}) + '\n')
        # #endregion
        return "book"
    
    # Debug info
    if intent == "scheduling" or conversation_stage in ["routing", "inquiry"]:
        missing = [f for f in required_fields if not appointment_data.get(f)]
        if missing:
            # #region agent log
            try:
                with open(r'c:\Users\revid\RAG\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"ai_receptionist.py:should_book_appointment:missing","message":"Missing fields computed","data":{"missing":missing,"stdout_encoding":getattr(getattr(sys,"stdout",None),"encoding",None)}}) + '\n')
            except Exception:
                pass
            # #endregion
            print(f"Still need fields for booking: {missing}")
    
    # #region agent log
    with open(r'c:\Users\revid\RAG\.cursor\debug.log', 'a', encoding='utf-8') as f:
        f.write(json.dumps({"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"ai_receptionist.py:should_book_appointment:return","message":"Decision to skip booking","data":{"decision":"skip_booking","reason":"missing_fields" if not has_all_fields else "already_booked" if already_booked else "other"}}) + '\n')
    # #endregion
    return "skip_booking"


def create_receptionist_graph():
    """Create the LangGraph workflow for the AI receptionist."""
    workflow = StateGraph(ReceptionistState)
    
    # Add nodes
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("extract_appointment_info", extract_appointment_info)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("book_appointment", book_appointment)
    workflow.add_node("route_inquiry", route_inquiry)
    
    # Set entry point
    workflow.set_entry_point("classify_intent")
    
    # Add edges
    workflow.add_edge("classify_intent", "retrieve_context")
    workflow.add_edge("retrieve_context", "extract_appointment_info")
    workflow.add_edge("extract_appointment_info", "generate_response")
    
    # Conditional edge: book appointment if scheduling with all info, otherwise skip
    workflow.add_conditional_edges(
        "generate_response",
        should_book_appointment,
        {
            "book": "book_appointment",
            "skip_booking": "route_inquiry"
        }
    )
    
    workflow.add_edge("book_appointment", "route_inquiry")
    
    # Always end after processing one turn - calling code handles looping
    workflow.add_edge("route_inquiry", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app


def chat_with_receptionist():
    """Interactive chat interface with the AI receptionist."""
    import sys
    # Set UTF-8 encoding for Windows console
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    # Check if backend is running
    print("="*80)
    print("AI SALON RECEPTIONIST")
    print("="*80)
    print("\nChecking backend connection...")
    
    backend_running = False
    
    # First, check if port is open (fast socket check)
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)  # Very short timeout
        result = sock.connect_ex(('127.0.0.1', 8000))
        sock.close()
        if result == 0:
            # Port is open, try HTTP request
            try:
                response = requests.get("http://127.0.0.1:8000/api/health", timeout=2)
                if response.status_code == 200:
                    backend_running = True
                    print("Backend is running and ready!")
                else:
                    backend_running = True
                    print("Backend is running (port 8000 is open)!")
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                # Port is open but HTTP not responding - might be starting up
                backend_running = True
                print("Backend port is open (may still be starting up)")
            except Exception:
                backend_running = True
                print("Backend port is open")
    except Exception:
        pass
    
    if not backend_running:
        print("WARNING: Cannot connect to backend on port 8000!")
        print("   Please start the backend first with: python scripts/start_backend.py")
        print("   Appointments cannot be booked without the backend running.\n")
        user_response = input("Continue anyway? (y/n): ").strip().lower()
        if user_response != 'y':
            print("Exiting. Please start the backend and try again.")
            return
        print("   Continuing without backend - appointment booking will fail.\n")
    
    print("\nWelcome! I'm your AI salon receptionist. How can I help you today?")
    print("Type 'exit', 'quit', or 'bye' to end the conversation.\n")
    
    # Create the graph
    app = create_receptionist_graph()
    
    # Initialize state
    state = {
        "messages": [HumanMessage(content="Hello")],
        "documents": [],
        "context": "",
        "intent": "",
        "routing_info": {},
        "conversation_stage": "greeting",
        "appointment_data": {}
    }
    
    # Process initial greeting
    state = app.invoke(state)
    print(f"Receptionist: {state['messages'][-1].content}\n")
    
    # Conversation loop
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
            # Process closing
            state["messages"].append(HumanMessage(content="Goodbye"))
            state = app.invoke(state)
            print(f"\nReceptionist: {state['messages'][-1].content}\n")
            break
        
        # Add user message
        state["messages"].append(HumanMessage(content=user_input))
        
        # Process through graph
        state = app.invoke(state)
        
        # Print AI response
        response_content = state['messages'][-1].content
        try:
            print(f"\nReceptionist: {response_content}\n")
        except UnicodeEncodeError:
            # Fallback: remove problematic characters
            safe_content = response_content.encode('ascii', 'ignore').decode('ascii')
            print(f"\nReceptionist: {safe_content}\n")
        
        # Check if appointment was just booked
        if state.get("appointment_data", {}).get("booked"):
            print("="*80)
            print("Appointment successfully booked!")
            print(f"   View all appointments at: http://localhost:8000")
            print("="*80 + "\n")
    
    print("Thank you for using the AI Receptionist. Have a great day!")


def demo_receptionist():
    """Run a demo conversation with the AI receptionist."""
    import sys
    # Set UTF-8 encoding for Windows console
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    print("="*80)
    print("AI RECEPTIONIST - DEMO MODE")
    print("="*80)
    
    app = create_receptionist_graph()
    
    # Demo conversation flow
    demo_conversations = [
        "Hello",
        "What services do you offer?",
        "How much does a haircut cost?",
        "I'd like to book an appointment for a haircut",
        "Sarah Johnson",
        "sarah@example.com",
        "2024-12-20",
        "14:30",
        "Thank you!"
    ]
    
    state = {
        "messages": [],
        "documents": [],
        "context": "",
        "intent": "",
        "routing_info": {},
        "conversation_stage": "greeting",
        "appointment_data": {}
    }
    
    for user_input in demo_conversations:
        print(f"\n{'='*80}")
        print(f"User: {user_input}")
        print(f"{'='*80}")
        
        state["messages"].append(HumanMessage(content=user_input))
        state = app.invoke(state)
        
        # Get response content and handle encoding issues
        response_content = state['messages'][-1].content
        try:
            print(f"\nReceptionist: {response_content}")
        except UnicodeEncodeError:
            # Fallback: remove problematic characters
            safe_content = response_content.encode('ascii', 'ignore').decode('ascii')
            print(f"\nReceptionist: {safe_content}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_receptionist()
    else:
        chat_with_receptionist()
