# magpie daemon: watch an inbox for audio, transcribe it, archive, notify.
#
# bare-metal because mlx_whisper needs Metal -- same carve-out as paling serve. the watch
# loop and the lifecycle emit are now birblib's: service.serve_inbox wires
# good_citizen.watcher to a build-bento-and-drive handler that relays each transition to
# the Go sidecar (the bus). this closes the old TODO -- the daemon emits now, it does not
# only log -- and drops magpie's hand-rolled poll loop. /health stays magpie's own.
#
# dup-over-loss is stronger than before: the watcher never touches the inbox file (the old
# loop renamed it in place); on_noticed COPIES it into the bento under a safe name.

import logging
import threading
from pathlib import Path

import uvicorn
from birblib import service
from fastapi import FastAPI
from good_citizen.provider import FilesystemProvider

from . import pipeline

log = logging.getLogger(__name__)

_AUDIO_EXTS = {".m4a", ".wav", ".mp3", ".aac", ".flac"}

app = FastAPI(title="magpie")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "magpie"}


def _make_bento(source) -> pipeline.AudioBento:
    # build a NOTICED raw-audio bento for an inbox source, honoring a sidecar prompt next to
    # it. the source banchan starts at the inbox file; on_noticed copies it in.
    path = Path(source.location)
    prompt = pipeline._sidecar_prompt(path)
    return pipeline.AudioBento.new(path, prompt)


def serve(host: str = "127.0.0.1", port: int = 8092, inbox: Path | None = None,
          sidecar_url: str | None = None) -> None:
    # run the watch-folder daemon: serve_inbox drives each new audio source through the FSM
    # with the real sidecar emit, on a background thread; /health is served by uvicorn. the
    # FilesystemProvider owns intake, the persistent (restart-surviving) dedup, the
    # partial-write guard, and the terminal notify drop -- swapping the inbox for a synced
    # folder or a tunnel collector is a provider change, not a daemon change.
    inbox = inbox or (pipeline.DATA_ROOT / "inbox")
    provider = FilesystemProvider(
        inbox, suffixes=_AUDIO_EXTS, notify_dir=pipeline.DATA_ROOT / "notifications",
    )
    threading.Thread(
        target=service.serve_inbox,
        args=(provider, pipeline.AudioHandlers, _make_bento),
        kwargs={"sidecar_url": sidecar_url},
        daemon=True,
    ).start()
    log.info("magpie serving on %s:%s", host, port)
    uvicorn.run(app, host=host, port=port)
