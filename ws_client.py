# ws_client.py
import asyncio
import websockets
import json

async def run():
    uri = "ws://127.0.0.1:8765"
    try:
        async with websockets.connect(uri) as ws:
            print("[WS CLIENT] Connected to", uri)
            while True:
                msg = await ws.recv()
                try:
                    obj = json.loads(msg)
                    print("[WS CLIENT] Got message:", json.dumps(obj, indent=2)[:1000])
                except Exception:
                    print("[WS CLIENT] Raw message:", msg)
    except Exception as e:
        print("[WS CLIENT] Connection error:", e)

if __name__ == "__main__":
    asyncio.run(run())
