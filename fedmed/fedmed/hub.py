"""WebSocket broadcaster feeding the React dashboard. Keeps history so late joiners see every round."""
import asyncio
import json
import threading

import websockets


class MetricsHub:
    def __init__(self, host, port):
        self.history, self.clients = [], set()
        self.loop = asyncio.new_event_loop()
        ready = threading.Event()
        threading.Thread(target=self._run, args=(host, port, ready), daemon=True).start()
        ready.wait(5)

    def _run(self, host, port, ready):
        asyncio.set_event_loop(self.loop)

        async def handler(ws):
            self.clients.add(ws)
            try:
                await ws.send(json.dumps({"type": "history", "events": self.history}))
                async for _ in ws:
                    pass
            finally:
                self.clients.discard(ws)

        async def start():
            await websockets.serve(handler, host, port)

        self.loop.run_until_complete(start())
        ready.set()
        self.loop.run_forever()

    def publish(self, event: dict):
        self.history.append(event)
        msg = json.dumps(event)

        async def send():
            websockets.broadcast(self.clients, msg)

        asyncio.run_coroutine_threadsafe(send(), self.loop)
