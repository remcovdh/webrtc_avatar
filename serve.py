"""Serve the avatar app over HTTP and, when configured, HTTPS in one process.

Browsers only allow microphone access on localhost or HTTPS, so opening the
page from another machine needs HTTPS. Both servers share the same `app` (and
so the same loaded models); only the HTTP server runs the app's lifespan.
"""

from __future__ import annotations

import asyncio
import os

import uvicorn

from server import app


async def main() -> None:
    http = uvicorn.Server(
        uvicorn.Config(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
    )
    servers = [http.serve()]
    https_port = os.getenv("HTTPS_PORT", "").strip()
    if https_port:
        https = uvicorn.Server(
            uvicorn.Config(
                app,
                host="0.0.0.0",
                port=int(https_port),
                ssl_certfile=os.environ["HTTPS_CERT_FILE"],
                ssl_keyfile=os.environ["HTTPS_KEY_FILE"],
                lifespan="off",
            )
        )
        servers.append(https.serve())
    await asyncio.gather(*servers)


if __name__ == "__main__":
    asyncio.run(main())
