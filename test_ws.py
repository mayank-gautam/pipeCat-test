import asyncio
import websockets
import json

async def test():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as ws:
        print("Connected!")
        
        # Simulate Twilio start event
        await ws.send(json.dumps({
            "event": "start",
            "start": {
                "streamSid": "MZ_test",
                "callSid": "CA_test"
            }
        }))
        
        # Wait for response
        response = await ws.recv()
        print(f"Received: {response}")

asyncio.run(test())



AZURE_SPEECH_API_KEY
="Ccv9z2NcajfPfsRwr1k0Zs7UMb4RGy6BqC8PM3rt5wsFdPVYrOuLJQQJ99BKACYeBjFXJ3w3AAAYACOGNUfO"
AZURE_SPEECH_REGION
="eastus"

OPENAI_API_KEY
="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"

TWILIO_ACCOUNT_SID
="ACa201b0614e265ffe57d279c0e7bf4813"
TWILIO_AUTH_TOKEN
="d826872051661f129e626664e8ed4bed"
TWILIO_PHONE_NUMBER
="+16626727576"
# Server port
PORT=8000
BASE_URL="https://tidy-boxes-cheer.loca.lt"
