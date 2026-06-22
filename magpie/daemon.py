"""magpie daemon: watch an inbox for audio, transcribe it, archive, notify.

Bare-metal because mlx_whisper needs Metal -- same carve-out as paling serve. The
in-cluster Go sidecar (good-citizen) bridges this host process to the fleet and
will own the Kafka notification; until that lands (blm#11) this is a minimal
Python watch loop that logs instead of emitting.
"""

import logging
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from . import pipeline

log = logging.getLogger(__name__)

_AUDIO_EXTS = {".m4a", ".wav", ".mp3", ".aac", ".flac"}

app = FastAPI(title="magpie")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "magpie"}


def _watch(inbox: Path) -> None:
    # poll the inbox; process each new audio file once. In-memory "seen" set, so a
    # restart reprocesses what's still in the inbox -- safe, because each run
    # creates a fresh bento and never deletes the source (dup-over-loss). Moving
    # processed files out, or persisting "seen", is a later refinement.
    inbox.mkdir(parents=True, exist_ok=True)
    seen: set[Path] = set()
    log.info("watching %s for audio", inbox)
    while True:
        for f in sorted(inbox.iterdir()):
            if f.is_file() and f.suffix.lower() in _AUDIO_EXTS and f not in seen:
                seen.add(f)
                try:
                    manifest = pipeline.process(f)
                    log.info("transcribed %s -> %s", f.name, manifest["transcript"])
                    # TODO(blm#11): emit a magpie.events banchan lifecycle event via
                    # the good-citizen Go sidecar instead of only logging.
                except Exception as e:  # noqa: BLE001 - one bad file must not stop the loop
                    log.error("failed to process %s: %s", f, e)
        time.sleep(5)


def serve(host: str = "127.0.0.1", port: int = 8092, inbox: Path | None = None) -> None:
    inbox = inbox or (pipeline.DATA_ROOT / "inbox")
    threading.Thread(target=_watch, args=(inbox,), daemon=True).start()
    log.info("magpie serving on %s:%s", host, port)
    uvicorn.run(app, host=host, port=port)
