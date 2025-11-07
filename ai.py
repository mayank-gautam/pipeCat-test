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
    LLMMessagesFrame,
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


# Store active websocket connections
active_connections = {}


# FastAPI application for handling Twilio webhooks
app = FastAPI()


class TwilioAudioSender(FrameProcessor):
    """Send audio frames back to Twilio WebSocket"""
    
    def __init__(self, websocket: WebSocket, stream_sid: str):
        super().__init__()
        self.websocket = websocket
        self.stream_sid = stream_sid
        logger.info(f"TwilioAudioSender initialized with stream_sid: {stream_sid}")
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        # Send audio frames to Twilio
        if isinstance(frame, AudioRawFrame):
            try:
                # Encode audio to base64 for Twilio
                audio_payload = base64.b64encode(frame.audio).decode('utf-8')
                message = {
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {
                        "payload": audio_payload
                    }
                }
                await self.websocket.send_json(message)
                logger.debug(f"Sent {len(frame.audio)} bytes of audio to Twilio")
            except Exception as e:
                logger.error(f"Error sending audio to Twilio: {e}")
        
        await self.push_frame(frame, direction)


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
    # Get form data from Twilio
    form_data = await request.form()
    call_status = form_data.get('CallStatus')
    call_sid = form_data.get('CallSid')
    
    logger.info(f"Outbound call webhook - CallSID: {call_sid}, Status: {call_status}")
    
    # Get WebSocket URL
    base_url = os.getenv("BASE_URL", "https://your-ngrok-url.ngrok.io")
    ws_url = base_url.replace("https://", "wss://").replace("http://", "ws://") + "/ws"
    
    logger.info(f"Connecting to WebSocket: {ws_url}")
    
    # Return TwiML to connect to AI assistant when call is answered
    twiml = f'''<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Connect>
            <Stream url="{ws_url}" />
        </Connect>
    </Response>'''
    
    return Response(content=twiml, media_type="application/xml")


@app.post("/call-status")
async def call_status_webhook(request: Request):
    """Track call status updates"""
    form_data = await request.form()
    call_sid = form_data.get('CallSid')
    call_status = form_data.get('CallStatus')
    
    logger.info(f"Call {call_sid} status: {call_status}")
    
    return {"status": "received"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connection from Twilio"""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    stream_sid = None
    task = None
    pipeline_task = None
    
    try:
        # Initialize Azure STT
        stt = AzureSTTService(
            api_key=os.getenv("AZURE_SPEECH_API_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION"),
            language="en-US",
        )
        logger.info("STT initialized")
        
        # Initialize OpenAI LLM
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
        )
        logger.info("LLM initialized")
        
        # Initialize Azure TTS
        tts = AzureTTSService(
            api_key=os.getenv("AZURE_SPEECH_API_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION"),
            voice="en-US-JennyNeural",
        )
        logger.info("TTS initialized")
        
        # Set up conversation context
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI voice assistant on a phone call. Keep responses very brief and conversational (1-2 sentences max). Be friendly and professional.",
            },
        ]
        
        # Create aggregators to handle conversation flow
        user_response = LLMUserResponseAggregator(messages)
        assistant_response = LLMAssistantResponseAggregator(messages)
        
        # Pipeline will be created after we get stream_sid from Twilio
        pipeline = None
        
        # Flag to track if we've sent the greeting
        greeting_sent = False
        
        # Handle incoming audio stream from Twilio
        async for message in websocket.iter_json():
            try:
                event = message.get('event')
                logger.debug(f"Received event: {event}")
                
                if event == 'start':
                    stream_sid = message['start']['streamSid']
                    call_sid = message['start'].get('callSid', 'unknown')
                    logger.info(f"Stream started - StreamSID: {stream_sid}, CallSID: {call_sid}")
                    
                    # Create Twilio audio sender now that we have stream_sid
                    twilio_sender = TwilioAudioSender(websocket, stream_sid)
                    
                    # Create pipeline with Twilio sender
                    pipeline = Pipeline([
                        stt,
                        user_response,
                        llm,
                        tts,
                        twilio_sender,  # Add this to send audio back
                        assistant_response,
                    ])
                    
                    task = PipelineTask(
                        pipeline,
                        params=PipelineParams(
                            allow_interruptions=True,
                            enable_metrics=True,
                            enable_usage_metrics=True,
                        ),
                    )
                    
                    # Start the pipeline runner
                    runner = PipelineRunner()
                    pipeline_task = asyncio.create_task(runner.run(task))
                    logger.info("Pipeline started")
                    
                    # Wait a moment for connection to stabilize
                    await asyncio.sleep(1.0)
                    
                    # Send initial greeting by queuing a text frame
                    if not greeting_sent:
                        greeting = "Hello! I'm your AI assistant. How can I help you today?"
                        logger.info(f"Sending greeting: {greeting}")
                        
                        # Queue greeting as a text frame that will go through TTS
                        await task.queue_frames([
                            TextFrame(greeting)
                        ])
                        greeting_sent = True
                    
                elif event == 'media':
                    # Process incoming audio from user
                    if task is None:
                        logger.warning("Received media before stream started")
                        continue
                        
                    audio_payload = message['media']['payload']
                    audio_bytes = base64.b64decode(audio_payload)
                    
                    # Twilio sends μ-law encoded audio at 8kHz
                    audio_frame = AudioRawFrame(
                        audio=audio_bytes,
                        sample_rate=8000,
                        num_channels=1
                    )
                    await task.queue_frame(audio_frame)
                    
                elif event == 'stop':
                    logger.info("Stream stopped by Twilio")
                    if task:
                        await task.queue_frame(EndFrame())
                    break
                    
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Cancel pipeline task if it's running
        if pipeline_task and not pipeline_task.done():
            logger.info("Cancelling pipeline task")
            pipeline_task.cancel()
            try:
                await pipeline_task
            except asyncio.CancelledError:
                logger.info("Pipeline task cancelled")
                pass
                
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if task:
            try:
                await task.queue_frame(EndFrame())
            except Exception as e:
                logger.debug(f"Error sending EndFrame: {e}")
        
        try:
            await websocket.close()
        except Exception as e:
            logger.debug(f"Error closing websocket: {e}")
            
        logger.info("WebSocket connection closed")


# API endpoint to trigger a call programmatically
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


if __name__ == "__main__":
    # Run the FastAPI server
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
