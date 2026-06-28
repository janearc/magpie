# magpie CLI: `magpie transcribe <file>` and `magpie serve`.
#
# JSON by default -- magpie is a good agent-citizen, so every command's output is
# machine-readable. `transcribe` returns an ACK (it found your file and where the
# result will land), not the transcript text itself.

import argparse
import json
from pathlib import Path

from . import pipeline


def _transcribe(args) -> None:
    # run the pipeline and return an ACK manifest: where the result IS, not the
    # transcript inline. "graaaak -- found yer file." the transcript is the bento's
    # primary artifact in the birblib manifest envelope.
    manifest = pipeline.process(Path(args.file), prompt=args.prompt)
    print(json.dumps({
        "status": "accepted",
        "message": "graaaak -- found yer file. transcribed, cleaned, archived.",
        "bento_id": manifest["bento_id"],
        "ok": manifest["ok"],
        "transcript": manifest["artifact"],
    }, indent=2))


def _serve(args) -> None:
    from . import daemon
    daemon.serve(host=args.host, port=args.port, inbox=Path(args.inbox).expanduser())


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="magpie", description="voice memos in, clean transcripts out")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe", help="transcribe one audio file")
    t.add_argument("file", help="path to an audio file (m4a, wav, ...)")
    t.add_argument("--prompt", default="",
                   help="per-bento context (names, places, terms, what it's about) to bias whisper")
    t.set_defaults(func=_transcribe)

    s = sub.add_parser("serve", help="run the watch-folder daemon")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8092)
    s.add_argument("--inbox", default=str(pipeline.DATA_ROOT / "inbox"),
                   help="directory to watch for new audio")
    s.set_defaults(func=_serve)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
