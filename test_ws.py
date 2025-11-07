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