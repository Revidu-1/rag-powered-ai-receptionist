# AI Salon Receptionist with Appointment Booking

A LangGraph-based AI receptionist for a hair and nail salon, powered by OpenAI. The system can answer questions about salon services, pricing, and policies, and book appointments with a FastAPI backend and dashboard.

## Features

- 💇‍♀️ **AI Salon Receptionist**: Conversational AI that answers questions about salon services, pricing, and policies
- 📅 **Appointment Booking**: Books salon appointments and saves them to the backend
- 📊 **Dashboard**: Real-time dashboard to view and manage salon appointments
- 🔍 **RAG System**: Retrieval-Augmented Generation using salon knowledge base
- 💬 **Web Chat Interface**: Interactive web-based chat interface for customers
- 📞 **Voice Calls (Twilio)**: Customers can call and speak with the AI receptionist (see `TWILIO_VOICE_SETUP.md`)

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up OpenAI API key** (already in `.env`):
   The API key is already configured in your `.env` file.

## Running the System

### 1. Start the Backend Server

In one terminal, start the FastAPI backend:

```bash
python start_backend.py
```

Or directly:
```bash
python backend.py
```

The backend will be available at:
- **Dashboard**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **API Base**: http://localhost:8000/api

### 2. Run the AI Receptionist

In another terminal, run the receptionist:

**Interactive Mode** (recommended for booking appointments):
```bash
python ai_receptionist.py
```

**Demo Mode** (for testing):
```bash
python ai_receptionist.py demo
```

## Using the Receptionist

The AI salon receptionist can:

1. **Answer Questions**: Ask about salon services, pricing, policies, hours, and procedures
2. **Book Appointments**: Say something like:
   - "I'd like to book an appointment"
   - "Can I schedule a haircut?"
   - "I need to book a manicure"
   - The receptionist will ask for: name, email, date (YYYY-MM-DD), time (HH:MM), and service type (haircut, manicure, pedicure, coloring, etc.)

### Example Booking Conversation:

```
You: I'd like to book an appointment
Receptionist: I'd be happy to help you book an appointment! What service would you like? (haircut, manicure, pedicure, hair color, etc.)

You: I need a haircut
Receptionist: Great! Could you please provide your name?

You: Sarah Johnson
Receptionist: Thank you, Sarah. What's your email address?

You: sarah@example.com
Receptionist: Perfect! What date would you like? Please provide it in YYYY-MM-DD format.

You: 2024-12-20
Receptionist: Great! What time would work for you? Please use 24-hour format (HH:MM).

You: 14:30
Receptionist: ✅ Appointment booked successfully! Your appointment ID is #1. Date: 2024-12-20, Time: 14:30, Service: haircut...
```

## Viewing Appointments

Open your browser and go to:
```
http://localhost:8000
```

The dashboard will show:
- All scheduled appointments
- Statistics (total, scheduled, completed, cancelled)
- Ability to update appointment status
- Delete appointments

The dashboard auto-refreshes every 5 seconds to show new bookings.

## API Endpoints

### Appointment Endpoints
- `GET /api/appointments` - Get all appointments
- `POST /api/appointments` - Create a new appointment
- `GET /api/appointments/{id}` - Get a specific appointment
- `PATCH /api/appointments/{id}?status={status}` - Update appointment status
- `DELETE /api/appointments/{id}` - Delete an appointment

### Chat Endpoints
- `POST /api/chat` - Send a message to the AI receptionist (web chat)
- `GET /chat` - Web chat interface

### Twilio Voice Endpoints (if configured)
- `POST /twilio/voice/incoming` - Receive Twilio voice call webhooks
- `POST /twilio/voice/process` - Process speech input from voice calls
- `GET /twilio/voice/status` - Check Twilio Voice integration status

## Files Structure

- `ai_receptionist.py` - Main salon receptionist with LangGraph workflow
- `backend.py` - FastAPI backend server
- `dashboard.html` - Salon appointment dashboard interface
- `chat.html` - Web chat interface for customers
- `start_backend.py` - Script to start the backend
- `appointments.json` - Stores appointment data (auto-created)
- `salon_policies.txt`, `salon_faq.txt`, `salon_blog.txt` - Salon knowledge base documents
- `twilio_voice_integration.py` - Twilio Voice integration module (see `TWILIO_VOICE_SETUP.md`)
- `TWILIO_VOICE_SETUP.md` - Twilio Voice integration setup guide

## Integration Options

### Web Chat
The easiest way to interact with the receptionist is through the web chat interface:
- Open http://localhost:8000/chat in your browser
- Start chatting with the AI receptionist
- Book appointments directly through the chat

### Voice Calls (Twilio)
Enable voice calls so customers can call and speak with the AI receptionist:
- See `TWILIO_VOICE_SETUP.md` for detailed setup instructions
- Requires Twilio account and phone number
- Customers can book appointments via phone calls
- Uses OpenAI Whisper for speech-to-text and Twilio for text-to-speech

## Notes

- Make sure the backend is running before booking appointments through the receptionist
- The backend stores appointments in `appointments.json`
- The receptionist uses OpenAI's GPT-4o-mini model (configurable in code)
- All salon appointments are visible on the dashboard in real-time
- Operating hours: Monday-Saturday 9 AM-7 PM, Sunday 10 AM-5 PM
- Services include: Haircuts, Styling, Coloring, Highlights, Manicures, Pedicures, Nail Art, Extensions, and more
- Multiple channels (web chat, voice) all use the same AI receptionist and appointment booking system

