# magpie: an audio-to-text ingest service. voice memos in, clean transcripts out.
#
# a frood of the mesh: it watches an inbox, transcribes with whisper, cleans
# the result, archives both source and output (duplicates over loss), and -- once the
# Go sidecar lands -- notifies the fleet over Kafka. code is public; the audio and
# transcripts live under ${HOME}/var/magpie, never in the repo.

__version__ = "0.1.0"
