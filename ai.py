import os
import sys
import asyncio
import base64
import json
from typing import Optional
from fastapi import FastAPI, WebSocket, Request, Form
from fastapi.responses import Response
import uvicorn
from twilio.rest import Client

from pipecat.frames.frames import (
    Frame, 
    AudioRawFrame, 
    TranscriptionFrame, 
    TextFrame, 
    EndFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.azure import AzureTTSService, AzureSTTService
from pipecat.services.openai import OpenAILLMService
from pipecat.processors.aggregators.llm_response import LLMAssistantResponseAggregator, LLMUserResponseAggregator
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from loguru import logger
from dotenv import load_dotenv

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


# FastAPI application for handling Twilio webhooks
app = FastAPI()


@app.post("/make-call")
async def make_outbound_call(to_number: str = Form(...)):
    """
    Initiate an outbound call to an Indian number
    Usage: POST to /make-call with form data: to_number=+91XXXXXXXXXX
    """
    try:
        # Initialize Twilio client
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        twilio_number = os.getenv("TWILIO_PHONE_NUMBER")
        
        if not all([account_sid, auth_token, twilio_number]):
            return {"error": "Twilio credentials not configured"}
        
        client = Client(account_sid, auth_token)
        
        # Get the base URL for webhooks
        base_url = os.getenv("BASE_URL", "https://your-ngrok-url.ngrok.io")
        
        # Make the call
        call = client.calls.create(
            to=to_number,  # Indian number in format: +91XXXXXXXXXX
            from_=twilio_number,
            url=f"{base_url}/outbound-voice",
            status_callback=f"{base_url}/call-status",
            status_callback_event=['initiated', 'ringing', 'answered', 'completed']
        )
        
        logger.info(f"Call initiated to {to_number}, Call SID: {call.sid}")
        
        return {
            "success": True,
            "call_sid": call.sid,
            "to": to_number,
            "status": call.status
        }
        
    except Exception as e:
        logger.error(f"Error making call: {e}")
        return {"error": str(e)}


@app.post("/outbound-voice")
async def outbound_voice_webhook(request: Request):
    """
    Handle the outbound call when recipient picks up
    This webhook is called when the Indian number answers
    """
    try:
        logger.info("=" * 50)
        logger.info("📞 WEBHOOK CALLED: /outbound-voice")
        logger.info("=" * 50)
        
        # Log request details
        logger.info(f"Request URL: {request.url}")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request headers: {dict(request.headers)}")
        
        form_data = await request.form()
        logger.info(f"Form data: {dict(form_data)}")
        
        call_status = form_data.get('CallStatus')
        call_sid = form_data.get('CallSid')
        from_number = form_data.get('From')
        to_number = form_data.get('To')
        
        logger.info(f"✅ Call Status: {call_status}")
        logger.info(f"✅ Call SID: {call_sid}")
        logger.info(f"✅ From: {from_number}")
        logger.info(f"✅ To: {to_number}")
        
        # Get WebSocket URL
        base_url = os.getenv("BASE_URL")
        if not base_url:
            logger.error("❌ BASE_URL not set in environment!")
            base_url = str(request.base_url).rstrip('/')
            logger.info(f"Using request base URL: {base_url}")
        
        logger.info(f"Base URL: {base_url}")
        
        # Create WebSocket URL
        ws_url = base_url.replace("https://", "wss://").replace("http://", "ws://") + "/ws"
        logger.info(f"WebSocket URL: {ws_url}")
        
        # Return TwiML to connect to AI assistant when call is answered
        twiml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice">Connecting you to our AI assistant. Please wait.</Say>
    <Connect>
        <Stream url="{ws_url}" />
    </Connect>
</Response>'''
        
        logger.info("✅ Returning TwiML:")
        logger.info(twiml)
        logger.info("=" * 50)
        
        return Response(content=twiml, media_type="application/xml")
        
    except Exception as e:
        logger.error(f"❌ ERROR in outbound_voice_webhook: {e}")
        import traceback
        traceback.print_exc()
        
        # Return error TwiML
        error_twiml = '''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice">Sorry, there was an error connecting to the AI assistant.</Say>
    <Hangup/>
</Response>'''
        return Response(content=error_twiml, media_type="application/xml")


@app.post("/call-status")
async def call_status_webhook(request: Request):
    """Track call status updates"""
    form_data = await request.form()
    call_sid = form_data.get('CallSid')
    call_status = form_data.get('CallStatus')
    
    logger.info(f"📞 Call {call_sid} status: {call_status}")
    
    return {"status": "received"}


class TranscriptionLogger(FrameProcessor):
    """Custom processor to log transcriptions"""
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TranscriptionFrame):
            logger.info(f"🎤 USER SAID: '{frame.text}'")
            print(f"\n🎤 USER TRANSCRIPTION: {frame.text}\n")
        elif isinstance(frame, TextFrame):
            logger.info(f"🤖 AI RESPONSE: '{frame.text}'")
            print(f"\n🤖 AI TEXT RESPONSE: {frame.text}\n")
        
        await self.push_frame(frame, direction)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connection from Twilio"""
    await websocket.accept()
    logger.info("✅ WebSocket connection established")
    
    stream_sid = None
    
    try:
        logger.info("⏳ Initializing Azure STT...")
        stt = AzureSTTService(
            api_key=os.getenv("AZURE_SPEECH_API_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION"),
            language="en-US",
        )
        logger.info("✅ Azure STT initialized")
        
        logger.info("⏳ Initializing OpenAI LLM...")
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
        )
        logger.info("✅ OpenAI LLM initialized")
        
        logger.info("⏳ Initializing Azure TTS...")
        tts = AzureTTSService(
            api_key=os.getenv("AZURE_SPEECH_API_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION"),
            voice="en-US-JennyNeural",
        )
        logger.info("✅ Azure TTS initialized")
        
        # Set up conversation context
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI voice assistant on a phone call. Keep responses very brief and conversational (1-2 sentences max). Be friendly and professional.",
            },
        ]
        
        # Create aggregators
        user_response = LLMUserResponseAggregator(messages)
        assistant_response = LLMAssistantResponseAggregator(messages)
        transcription_logger = TranscriptionLogger()
        
        # Create pipeline with logger
        pipeline = Pipeline([
            stt,
            transcription_logger,  # Log transcriptions
            user_response,
            llm,
            tts,
            assistant_response,
        ])
        
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
            ),
        )
        
        logger.info("✅ Pipeline created")
        
        # Queue to collect output frames
        output_queue = asyncio.Queue()
        
        # Task to handle output frames
        async def handle_output():
            try:
                while True:
                    frame = await output_queue.get()
                    
                    if isinstance(frame, EndFrame):
                        logger.info("🛑 End frame received")
                        break
                    
                    # Send TTS audio back to Twilio
                    if isinstance(frame, TTSAudioRawFrame) or isinstance(frame, AudioRawFrame):
                        if stream_sid and hasattr(frame, 'audio'):
                            try:
                                audio_payload = base64.b64encode(frame.audio).decode('utf-8')
                                message = {
                                    "event": "media",
                                    "streamSid": stream_sid,
                                    "media": {
                                        "payload": audio_payload
                                    }
                                }
                                await websocket.send_json(message)
                                logger.debug("🔊 Sent audio chunk to Twilio")
                            except Exception as e:
                                logger.error(f"❌ Error sending audio: {e}")
            except Exception as e:
                logger.error(f"❌ Error in output handler: {e}")
        
        # Start output handler
        output_task = asyncio.create_task(handle_output())
        
        # Run the pipeline in background
        runner = PipelineRunner()
        pipeline_task = asyncio.create_task(runner.run(task))
        
        logger.info("✅ Pipeline started")
        
        # Flag for greeting
        greeting_sent = False
        
        # Handle incoming audio stream from Twilio
        try:
            while True:
                data = await websocket.receive_json()
                
                if data['event'] == 'start':
                    stream_sid = data['start']['streamSid']
                    call_sid = data['start'].get('callSid', 'unknown')
                    logger.info(f"✅ Stream started - StreamSID: {stream_sid}, CallSID: {call_sid}")
                    
                    # Wait for connection to stabilize
                    await asyncio.sleep(1.0)
                    
                    # Send greeting via TTS
                    if not greeting_sent:
                        greeting = "Hello! I'm your AI assistant. How can I help you today?"
                        logger.info(f"📢 Sending greeting: {greeting}")
                        
                        greeting_frame = TextFrame(text=greeting)
                        await task.queue_frame(greeting_frame)
                        greeting_sent = True
                
                elif data['event'] == 'media':
                    # Receive audio from user
                    audio_payload = data['media']['payload']
                    audio_bytes = base64.b64decode(audio_payload)
                    
                    # Send to STT pipeline
                    audio_frame = AudioRawFrame(
                        audio=audio_bytes,
                        sample_rate=8000,
                        num_channels=1
                    )
                    await task.queue_frame(audio_frame)
                
                elif data['event'] == 'stop':
                    logger.info("🛑 Stream stopped by Twilio")
                    await task.queue_frame(EndFrame())
                    await output_queue.put(EndFrame())
                    break
        
        except Exception as e:
            logger.error(f"❌ Error in WebSocket receive loop: {e}")
            import traceback
            traceback.print_exc()
        
        # Cleanup
        logger.info("⏳ Waiting for tasks to complete...")
        await asyncio.gather(pipeline_task, output_task, return_exceptions=True)
        logger.info("✅ Tasks completed")
        
    except Exception as e:
        logger.error(f"❌ Error in WebSocket handler: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            await websocket.close()
        except:
            pass
        logger.info("❌ WebSocket connection closed")


@app.get("/")
async def root():
    """Simple status page"""
    return {
        "status": "running",
        "endpoints": {
            "make_call": "POST /make-call (form data: to_number)",
            "outbound_voice": "POST /outbound-voice (Twilio webhook)",
            "call_status": "POST /call-status (Twilio webhook)",
            "websocket": "WS /ws"
        },
        "info": "AI Voice Assistant is ready to make calls"
    }


@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify server is reachable"""
    base_url = os.getenv("BASE_URL", "not_set")
    return {
        "status": "✅ Server is working!",
        "base_url": base_url,
        "ws_url": base_url.replace("https://", "wss://").replace("http://", "ws://") + "/ws" if base_url != "not_set" else "not_set"
    }


@app.get("/health")
async def health_check():
    """Simple health check"""
    return {"status": "ok", "service": "ai-voice-assistant"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )