# Twilio Voice Integration with OpenAI Whisper

The AI salon receptionist can now handle voice calls using Twilio Voice API with OpenAI Whisper for high-quality speech transcription!

## How It Works

1. **Customer calls** your Twilio phone number
2. **Twilio receives call** → sends webhook to your backend
3. **Backend processes** through AI receptionist (existing LangGraph)
4. **Twilio converts** AI response text to speech
5. **Customer hears** the response

## Features

- ✅ Phone call handling via Twilio
- ✅ Speech-to-text using Twilio's built-in recognition (primary)
- ✅ OpenAI Whisper fallback for better accuracy (optional)
- ✅ Text-to-speech via Twilio (natural voices)
- ✅ Full conversation support
- ✅ Appointment booking through voice calls
- ✅ Uses your existing AI receptionist

## Setup Steps

### 1. Get Twilio Phone Number

1. **Log into Twilio Console**: https://console.twilio.com
2. **Go to Phone Numbers** → **Buy a Number**
3. **Choose your country/region**
4. **Purchase a voice-capable number**
5. **Note the phone number** (format: +1234567890)

### 2. Configure Environment Variables

Add to your `.env` file:

```bash
# Twilio Configuration (you may already have these for WhatsApp)
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER=+1234567890  # Your Twilio voice number

# OpenAI Configuration (for Whisper - you already have this)
OPENAI_API_KEY=your_openai_api_key
```

### 3. Set Up Webhook in Twilio

1. **In Twilio Console** → **Phone Numbers** → Click your number
2. **Voice & Fax Configuration**:
   - **A CALL COMES IN**: Set to `Webhook`
   - **URL**: `https://your-ngrok-url.ngrok.io/twilio/voice/incoming`
   - **HTTP Method**: `POST`
3. **Save** the configuration

### 4. Set Up ngrok (for Local Testing)

1. **Start your backend**:
   ```bash
   python start_backend.py
   ```

2. **In a new terminal, start ngrok**:
   ```bash
   ngrok http 8000
   ```

3. **Copy the HTTPS URL** (e.g., `https://abc123.ngrok.io`)

4. **Update Twilio webhook URL** with your ngrok URL:
   ```
   https://abc123.ngrok.io/twilio/voice/incoming
   ```

### 5. Test the Integration

1. **Call your Twilio phone number**
2. **You should hear**: "Hello! Welcome to our salon. I'm your AI receptionist. How can I help you today?"
3. **Speak your message** (e.g., "I'd like to book an appointment")
4. **The AI receptionist will respond** and guide you through booking

## API Endpoints

- `POST /twilio/voice/incoming` - Handles incoming calls
- `POST /twilio/voice/process` - Processes speech input
- `GET /twilio/voice/status` - Check integration status

## How Speech Recognition Works

### Primary: Twilio Built-in Speech Recognition

- Twilio has built-in speech recognition
- Works automatically with the `Gather` verb
- Fast and reliable
- No additional API calls needed

### Fallback: OpenAI Whisper (Optional)

- If Twilio's recognition doesn't work well, we can use Whisper
- Higher accuracy for complex speech
- Requires OpenAI API key (you already have this)
- Automatically used if recording URL is available

## Example Conversation

```
Customer: *calls phone number*
Receptionist: "Hello! Welcome to our salon. I'm your AI receptionist. How can I help you today?"

Customer: "I'd like to book an appointment for a haircut"
Receptionist: "Great! I'd be happy to help you book an appointment. Could you please provide your name?"

Customer: "Sarah Johnson"
Receptionist: "Thank you, Sarah. What's your email address?"

Customer: "sarah@example.com"
Receptionist: "Perfect! What date would you like? Please provide it in YYYY-MM-DD format."

Customer: "December 20th, 2024"
Receptionist: "I understand December 20th, 2024. What time would work for you? Please use 24-hour format."

Customer: "Two thirty PM"
Receptionist: "Great news! Your appointment has been booked successfully. Your appointment ID is 5. Date: 2024-12-20, Time: 14:30, Service: haircut. Is there anything else I can help you with?"

Customer: "No, that's all. Thank you!"
Receptionist: "Thank you for calling. Have a great day!"
```

## Troubleshooting

### Calls Not Connecting

**Problem**: Calls to Twilio number don't connect.

**Solutions**:
- Verify Twilio phone number is active
- Check webhook URL is correct in Twilio console
- Ensure ngrok is running (for local testing)
- Verify backend is running on port 8000

### No Response from Receptionist

**Problem**: Call connects but no AI response.

**Solutions**:
- Check backend logs for errors
- Verify `OPENAI_API_KEY` is set correctly
- Test the AI receptionist directly: `python ai_receptionist.py demo`
- Check Twilio call logs in dashboard

### Speech Not Recognized

**Problem**: AI doesn't understand what you say.

**Solutions**:
- Speak clearly and slowly
- Check Twilio call logs for speech recognition results
- Whisper fallback will be used automatically if available
- Test with simple phrases first

### Appointments Not Booking

**Problem**: Conversation works but appointments don't appear.

**Solutions**:
- Verify backend `/api/appointments` endpoint is accessible
- Check backend logs for appointment creation errors
- Ensure all required appointment fields are provided
- Test appointment booking via web chat to verify backend works

## Cost Information

- **Twilio Voice**: Pay-per-minute pricing
  - Check current pricing: https://www.twilio.com/voice/pricing
  - Incoming calls: ~$0.0085/minute (US)
  - Outbound calls: ~$0.013/minute (US)

- **OpenAI Whisper**: Pay-per-minute of audio
  - $0.006 per minute of audio transcribed
  - Only used as fallback, so minimal cost

- **Phone Number**: Monthly rental fee
  - ~$1-2/month per number (varies by country)

## Production Considerations

### Security

1. **Webhook Verification**:
   - Twilio sends webhook signatures for verification
   - Implement signature verification in production (not included in basic setup)

2. **HTTPS Required**:
   - Twilio requires HTTPS for webhooks in production
   - Use ngrok for local testing (provides HTTPS)
   - For production, use a proper domain with SSL certificate

### Performance

1. **Response Time**:
   - Keep responses concise for better user experience
   - Twilio has timeout limits for webhook responses

2. **Error Handling**:
   - All errors are caught and return safe responses
   - Consider adding more detailed error logging for production

### Scaling

1. **Database Storage**:
   - Currently, conversation states are stored in memory
   - For production, move to a database (Redis, PostgreSQL, etc.)

2. **Multiple Phone Numbers**:
   - You can configure multiple numbers to use the same webhook
   - Useful for different regions or languages

## Comparison with Vapi

| Feature | Twilio Voice + Whisper | Vapi |
|---------|------------------------|------|
| Setup Complexity | Medium | Easy |
| Cost | Pay-per-minute | Pay-per-minute |
| Speech Recognition | Twilio + Whisper | Built-in |
| Text-to-Speech | Twilio (good) | High quality |
| Integration | Webhook-based | Webhook/API |
| Reliability | Very high | High |

## Next Steps

Once voice integration is working:

1. **Customize the Voice**: Choose different Twilio voices (alice, man, woman, etc.)
2. **Add Greetings**: Customize the greeting message
3. **Handle Edge Cases**: Add handling for specific scenarios
4. **Analytics**: Track call metrics and customer satisfaction
5. **Multi-language**: Consider adding support for multiple languages

## Support

- **Twilio Documentation**: https://www.twilio.com/docs/voice
- **OpenAI Whisper**: https://platform.openai.com/docs/guides/speech-to-text
- **Local Issues**: Check backend logs and Twilio call logs


