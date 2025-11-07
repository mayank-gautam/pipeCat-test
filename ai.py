import os
import asyncio
import base64
import audioop
from typing import Optional
from fastapi import FastAPI, WebSocket, Request, Form
from fastapi.responses import Response
import uvicorn
from twilio.rest import Client

from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)

# --- Pipecat imports ---
from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    TextFrame,
    EndFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    InputAudioRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask

# Service imports
from pipecat.services.azure.tts import AzureTTSService
from pipecat.services.azure.stt import AzureSTTService
from pipecat.services.openai.llm import OpenAILLMService

from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

# FastAPI app
app = FastAPI()


class TwilioAudioSender(FrameProcessor):
    """Send TTS audio frames back to Twilio WebSocket with proper format conversion"""

    def __init__(self, websocket: WebSocket, stream_sid: str):
        super().__init__()
        self.websocket = websocket
        self.stream_sid = stream_sid
        self.is_speaking = False

    async def clear_audio_buffer(self):
        """Clear Twilio's audio buffer (used for interruptions)"""
        try:
            message = {
                "event": "clear",
                "streamSid": self.stream_sid,
            }
            await self.websocket.send_json(message)
            logger.debug("Cleared Twilio audio buffer")
        except Exception as e:
            logger.error(f"Error clearing Twilio buffer: {e}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSStartedFrame):
            self.is_speaking = True
        elif isinstance(frame, TTSStoppedFrame):
            self.is_speaking = False

        # Send audio frames from TTS
        if isinstance(frame, AudioRawFrame) and self.is_speaking and direction == FrameDirection.DOWNSTREAM:
            try:
                # Convert audio to Twilio format (mulaw, 8kHz, mono)
                audio_bytes = frame.audio
                
                # Skip empty audio
                if not audio_bytes or len(audio_bytes) == 0:
                    await self.push_frame(frame, direction)
                    return
                
                # Resample to 8kHz if necessary
                if frame.sample_rate != 8000:
                    audio_bytes, _ = audioop.ratecv(
                        audio_bytes,
                        2,  # 16-bit samples
                        frame.num_channels,
                        frame.sample_rate,
                        8000,
                        None
                    )
                
                # Convert stereo to mono if necessary
                if frame.num_channels == 2:
                    audio_bytes = audioop.tomono(audio_bytes, 2, 1, 1)
                
                # Convert to mulaw
                mulaw_audio = audioop.lin2ulaw(audio_bytes, 2)
                
                # Base64 encode
                audio_payload = base64.b64encode(mulaw_audio).decode("utf-8")
                
                message = {
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {"payload": audio_payload},
                }
                await self.websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending audio to Twilio: {e}")

        await self.push_frame(frame, direction)


class TwilioAudioReceiver(FrameProcessor):
    """Receive audio from Twilio and convert to PCM format"""

    def __init__(self):
        super().__init__()
        self._audio_buffer = bytearray()
        self._chunk_size = 320  # 20ms at 8kHz mulaw = 160 bytes, but PCM will be double

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

    async def push_twilio_audio(self, audio_bytes: bytes):
        """Convert Twilio mulaw audio to PCM and push to pipeline"""
        try:
            # Skip empty audio
            if not audio_bytes or len(audio_bytes) == 0:
                return
            
            # Convert mulaw to PCM 16-bit
            pcm_audio = audioop.ulaw2lin(audio_bytes, 2)
            
            # Add to buffer
            self._audio_buffer.extend(pcm_audio)
            
            # Process in chunks to avoid overwhelming the pipeline
            while len(self._audio_buffer) >= self._chunk_size:
                chunk = bytes(self._audio_buffer[:self._chunk_size])
                self._audio_buffer = self._audio_buffer[self._chunk_size:]
                
                # Create InputAudioRawFrame for STT
                audio_frame = InputAudioRawFrame(
                    audio=chunk,
                    sample_rate=8000,
                    num_channels=1,
                )
                
                await self.push_frame(audio_frame, FrameDirection.DOWNSTREAM)
        except Exception as e:
            logger.error(f"Error processing Twilio audio: {e}")


@app.post("/make-call")
async def make_outbound_call(to_number: str = Form(...)):
    """
    Initiate an outbound call to an Indian number
    Usage: POST to /make-call with form data: to_number=+91XXXXXXXXXX
    """
    try:
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        twilio_number = os.getenv("TWILIO_PHONE_NUMBER")

        if not all([account_sid, auth_token, twilio_number]):
            return {"error": "Twilio credentials not configured"}

        client = Client(account_sid, auth_token)
        base_url = os.getenv("BASE_URL", "https://your-ngrok-url.ngrok.io")

        call = client.calls.create(
            to=to_number,
            from_=twilio_number,
            url=f"{base_url}/outbound-voice",
            status_callback=f"{base_url}/call-status",
            status_callback_event=["initiated", "ringing", "answered", "completed"],
        )

        return {"success": True, "call_sid": call.sid, "to": to_number, "status": call.status}

    except Exception as e:
        return {"error": str(e)}


@app.post("/outbound-voice")
async def outbound_voice_webhook(request: Request):
    """Handle the outbound call when recipient picks up"""
    form_data = await request.form()
    call_sid = form_data.get("CallSid")

    base_url = os.getenv("BASE_URL", "https://your-ngrok-url.ngrok.io")
    ws_url = base_url.replace("https://", "wss://").replace("http://", "ws://") + "/ws"

    # Use inbound_track only to prevent audio feedback loop
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Connect>
            <Stream url="{ws_url}" track="inbound_track" />
        </Connect>
    </Response>"""

    return Response(content=twiml, media_type="application/xml")


@app.post("/call-status")
async def call_status_webhook(request: Request):
    """Track call status updates"""
    form_data = await request.form()
    call_sid = form_data.get("CallSid")
    call_status = form_data.get("CallStatus")
    logger.info(f"Call {call_sid} status: {call_status}")
    return {"status": "received"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connection from Twilio"""
    await websocket.accept()
    stream_sid = None
    task: Optional[PipelineTask] = None
    pipeline_task = None
    runner = None
    audio_receiver = None

    try:
        # Initialize Azure STT
        stt = AzureSTTService(
            api_key=os.getenv("AZURE_SPEECH_API_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION"),
            language="en-US",
        )

        # Initialize OpenAI LLM
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"),
        )

        # Initialize Azure TTS
        tts = AzureTTSService(
            api_key=os.getenv("AZURE_SPEECH_API_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION"),
            voice=os.getenv("AZURE_TTS_VOICE", "en-US-JennyNeural"),
        )

        # Initial conversation messages
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI voice assistant on a phone call. Keep responses very brief and conversational (1-2 sentences max). Be friendly and professional.",
            },
        ]

        # Create context object FIRST, then pass it to create_context_aggregator
        context = OpenAILLMContext(messages=messages)
        context_aggregator = llm.create_context_aggregator(context)
        
        greeting_sent = False

        async for message in websocket.iter_json():
            try:
                event = message.get("event")
                logger.debug(f"WS event: {event}")

                if event == "start":
                    stream_sid = message["start"]["streamSid"]
                    call_sid = message["start"].get("callSid", "unknown")

                    # Initialize audio processors
                    audio_receiver = TwilioAudioReceiver()
                    twilio_sender = TwilioAudioSender(websocket, stream_sid)

                    # Build pipeline
                    pipeline = Pipeline(
                        [
                            audio_receiver,         # Receive and convert Twilio audio
                            stt,                    # Speech to text
                            context_aggregator.user(),  # User context
                            llm,                    # LLM processing
                            tts,                    # Text to speech
                            twilio_sender,          # Send audio to Twilio
                            context_aggregator.assistant(),  # Assistant context
                        ]
                    )

                    task = PipelineTask(
                        pipeline,
                        params=PipelineParams(
                            allow_interruptions=True,
                            enable_metrics=False,
                            enable_usage_metrics=False,
                        ),
                    )

                    runner = PipelineRunner()
                    pipeline_task = asyncio.create_task(runner.run(task))

                    # Wait for pipeline to be ready
                    await asyncio.sleep(0.5)

                    # Send greeting
                    if not greeting_sent:
                        greeting = "Hello! I'm your AI assistant. How can I help you today?"
                        try:
                            await task.queue_frame(TextFrame(greeting))
                            greeting_sent = True
                        except Exception as e:
                            logger.error(f"Error queuing greeting: {e}")

                elif event == "media":
                    if audio_receiver is None:
                        logger.warning("Received media before pipeline created")
                        continue

                    try:
                        audio_payload = message["media"]["payload"]
                        audio_bytes = base64.b64decode(audio_payload)
                        
                        # Push audio through the receiver processor
                        await audio_receiver.push_twilio_audio(audio_bytes)

                    except Exception as e:
                        logger.exception(f"Error processing audio media: {e}")

                elif event == "stop":
                    if task:
                        try:
                            await task.queue_frame(EndFrame())
                        except Exception as e:
                            logger.debug(f"Error sending EndFrame: {e}")
                    break

            except Exception as e:
                logger.exception(f"Error in websocket message loop: {e}")
                continue

        # Cleanup
        if pipeline_task and not pipeline_task.done():
            pipeline_task.cancel()
            try:
                await pipeline_task
            except asyncio.CancelledError:
                pass

    except Exception as e:
        logger.exception(f"Critical error in WebSocket handler: {e}")
    finally:
        if task:
            try:
                await task.queue_frame(EndFrame())
            except Exception as e:
                logger.debug(f"Error sending final EndFrame: {e}")

        try:
            await websocket.close()
        except Exception as e:
            logger.debug(f"Error closing websocket: {e}")

        logger.info("WebSocket connection closed completely")


@app.get("/")
async def root():
    """Simple status page"""
    return {
        "status": "running",
        "endpoints": {
            "make_call": "POST /make-call (form data: to_number)",
            "outbound_voice": "POST /outbound-voice (Twilio webhook)",
            "call_status": "POST /call-status (Twilio webhook)",
            "websocket": "WS /ws",
        },
        "info": "AI Voice Assistant is ready to make calls",
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
