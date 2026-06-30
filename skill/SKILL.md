---
name: magpie
description: Transcribe audio files to clean transcripts via magpie (voice memos in, clean transcripts out). Use when asked to transcribe an audio file, check the magpie daemon's health, or run its watch-folder daemon. Drives the magpie CLI/daemon through its wrapper -- never invoke whisper or place files by hand.
---

# magpie

magpie turns audio into clean transcripts. A run is a **bento** (the shared unit of
work): a directory holding the source audio under `raw_data/` and the transcripts under
`outputs/`. magpie transcribes with whisper large-v3 (MLX, Apple Silicon), cleans the
stutter-loop artifacts with a local model (degrading to the raw transcript if it is down),
and archives both source and output. Code is public; the audio and transcripts live under
`${HOME}/var/magpie`, never in the repo.

**Always operate through the `magpie` wrapper in this skill.** Transcription is a one-shot
op on the **bare-metal (Metal) host** -- it MUST NOT be modeled as an HTTP call (the daemon
serves only `GET /health`). The wrapper drives the installed `magpie` CLI; override it with
`MAGPIE_BIN`, and the daemon base with `MAGPIE_API`.

## Operations

```sh
./magpie transcribe <file> [prompt]   # transcribe one audio file; returns an ACK + where it landed
./magpie health                       # report the daemon's liveness (GET /health on :8092)
./magpie serve [inbox]                # run the watch-folder daemon (default ~/var/magpie/inbox)
```

`transcribe` returns an acknowledgement and the output location -- not the transcript text.
The transcript is the bento's artifact under `~/var/magpie`.

## The optional prompt (the `{prompt}` guard)

The optional `prompt` biases whisper toward the right proper nouns ("names: Will, Rae" for a
voice memo full of them). It is genuinely optional. The `mcp.json` `transcribe` handler
templates `{prompt}` into the wrapper's argument list, and the MCP harness leaves that token
**unsubstituted** (the literal `{prompt}`, or empty) when the caller omits the prompt. The
wrapper GUARDS this: an empty value OR the literal `{prompt}` means "no prompt", so it runs
the no-prompt path and whisper never receives a junk `--prompt ""` or `--prompt "{prompt}"`.
An omitted prompt therefore behaves identically to never passing one at all.

## Typical flow

```sh
./magpie health                                   # -> {"status":"ok","service":"magpie"}
./magpie transcribe ~/var/magpie/inbox/memo.m4a   # -> ACK manifest (bento_id, transcript path)
```

Each run is a bento walked through the shared FSM; progress shows up as lifecycle events on
the bus (once the Go sidecar lands), which obs-svc aggregates -- so a run is never silent.
