import asyncio
import websockets
from PIL import Image
import io

async def process_frame(frame):
    # Process the frame here
    image = Image.open(io.BytesIO(frame))
    # Perform image processing as needed
    

async def handle_client(websocket, path):
    async for message in websocket:
        await process_frame(message)

start_server = websockets.serve(handle_client, 'localhost', 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
