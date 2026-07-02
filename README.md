# magpie

Voice memos in, clean transcripts out. magpie watches a folder for audio, runs
it through whisper, cleans up the transcription, and archives both the source and
the result. It is a frood of the mesh: JSON-by-default, an agent skill, and
(soon) Kafka notifications via the shared frood sidecar.

## Pipeline

A run is a **bento** (the shared unit of work): a directory holding the source
audio under `raw_data/` and the transcripts under `outputs/`.

1. **transcribe** — whisper large-v3 (MLX, Apple Silicon) via `mlx_whisper`. The
   model lives in the shared read-only Hugging Face cache; it is not vendored here.
2. **cleanup** — a local model (mistral, via ollama) removes whisper's stutter-loop
   artifacts without rewriting the words. Best-effort: if the model is down, the
   raw transcript is kept. *(Interim: this routes through the frood model
   abstraction once that lands.)*
3. **archive** — the source audio is **copied** into the bento (never moved or
   deleted) and the transcripts written alongside. Duplicates over loss.

## Usage

```sh
magpie transcribe path/to/memo.m4a    # returns an ACK + where the result landed
magpie serve                          # watch ~/var/magpie/inbox and process arrivals
```

`transcribe` returns an acknowledgement and the output location — not the
transcript text. The transcript is an artifact under `~/var/magpie`.

## Data lives outside the repo

Code is public; audio and transcripts live under `~/var/magpie` (gitignored,
never committed). magpie only ships code.

## Status

Working today: the transcribe → cleanup → archive pipeline, the watch daemon,
delightd registration (register-on-start, health-gated, with retry while delightd
is unreachable), the agent skill + wrapper, and the kube manifests. Still to wire:
the Go sidecar for Kafka notifications (frood) — magpie declares its emit
contracts but does not yet put events on the bus.
