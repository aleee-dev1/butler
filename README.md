# Butler

Local semantic-search CLI using Node.js ESM, ChromaDB, and a local Python embedding server running **EmbeddingGemma 2 740M**.

## Model

**EmbeddingGemma 2** (multimodal: text, images, audio, video)
- Hugging Face: [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2)
- Stored locally in `~/.butler/models/embeddinggemma-2`, downloaded automatically by `butler init`
- License: Apache-2.0
- Architecture: 130M transformer + 140M embeddings + 170M vision encoder + 300M audio encoder
- Output: 768-dim vectors (configurable 128/256/512/768)
- Context: 8,192 tokens

## Requirements

- Node.js 22+
- Python 3.13 (creates `~/.butler/venvs/embeddinggemma2-venv`; override the interpreter with `PYTHON=/path/to/python`)
- ChromaDB server CLI available as `chroma` (`pipx install chromadb`)

## Install

```bash
# 1. Install Butler
npm install
npm link

# 2. Initialize (creates ~/.butler/venvs/embeddinggemma2-venv, installs the Python stack,
#    downloads the model to ~/.butler/models, writes ~/.butler/config.json)
butler init

# 3. Start services
butler start
```

`python-multipart`, `fastapi`, and `uvicorn` come from the pip stack above and are installed automatically with the pinned torch/sentence-transformers versions (CPU-only torch from the PyTorch CPU index).

If you already have a venv prepared, copy it to `~/.butler/venvs/embeddinggemma2-venv` and run `butler init --skip-venv` to skip venv creation and pip installs (the model is still verified/downloaded).

`butler init [/path/to/model-dir]` accepts an optional existing model directory instead of the default `~/.butler/models/embeddinggemma-2`.

Example `~/.butler/config.json`:
```json
{
  "host": "127.0.0.1",
  "chromaPort": 8000,
  "embedPort": 8765,
  "imageTokens": 280,
  "videoTokens": 70,
  "videoChunk": 10,
  "audioChunk": 30,
  "modelPath": "~/.butler/models/embeddinggemma-2",
  "chunk": 6000,
  "overlap": 300,
  "batch": 500
}
```

## Commands

```bash
butler start
butler start --image-tokens=70 --video-tokens=70   # persists to config; hot-restarts if budgets changed
butler stop
butler reset   # clear DB + logs, reset config (keeps venv and model)
butler add .
butler add . -r -v
butler add . -r --chunk=2000 --overlap=200 --batch=100
butler add ./README.md ./src/cli.js
butler add ./screenshot.png ./meeting-talk.mp3 ./demo.mov
butler add "This folder contains my company projects"
butler recall "where is the CLI argument parser"
butler recall "screenshots of the desktop" -i -n 5
butler recall "cat meowing" -a
butler recall "drone footage" -v
butler recall "project notes" --toml
butler delete .
butler delete ./README.md
butler flush -v
butler purge   # full wipe of ~/.butler (venv and model included)
```

Supported indexed files: UTF-8 text files, common image formats (PNG, JPEG, WebP, GIF, BMP, TIFF, AVIF), audio (MP3, WAV, FLAC, M4A, AAC, OGG, OPUS) and video (MP4, WebM, MKV, MOV, AVI, M4V, MPEG). Directories are never indexed on their own — `add .` / `add . -r` gather their files, and a directory's description only exists if you create a free-text note with `butler add "…"` from inside it (the `-d` recall filter still matches any legacy dir records).

### Chunking configuration

Text files are split into overlapping chunks for embedding. Configure in `~/.butler/config.json` or override per-command:

| Parameter | Config key | CLI flag | Default | Description |
|-----------|------------|----------|---------|-------------|
| Chunk size | `chunk` | `--chunk=N` | 6000 | Max characters per chunk (~1.5–2k tokens; the server window supports 8192 tokens) |
| Overlap | `overlap` | `--overlap=N` | 300 | Characters overlapping between chunks |
| Batch size | `batch` | `--batch=N` | 500 | Vectors per ChromaDB upsert |

Text chunks are grouped per file and embedded in one batched request to the server (`/embed/text` accepts a list). Anything that exceeds the model's 8192-token context is proactively windowed at 16,000 chars before sending, so no text is silently truncated server-side — the old reactive split-on-server-error logic is gone.

Example:
```bash
butler add . -r --chunk=2000 --overlap=200 --batch=100
```

### Media windowing (audio/video)

Audio and video files are embedded in **time windows**, so `butler recall` can point at a moment in a recording instead of a whole file — video hits display `mm:ss-mm:ss` segments just as text hits display `L<start>-<end>` line ranges.

| Parameter | Config key | Server default | Notes |
|-----------|------------|----------------|-------|
| Video window | `videoChunk` | 10 s | Segment cost ≈ frames (1 fps sampling) × per-frame budget |
| Audio window | `audioChunk` | 30 s | Hard max: the feature extractor truncates at 480,000 samples (30 s @ 16 kHz) |
| Video frame budget | `videoTokens` | 70 | Per-frame soft tokens (70/140/280/560/1120); 140 is the processor's quality default |

The server decodes each file once (torchcodec — no ffmpeg dependency), slices it into windows of the configured length pre-sampled at 1 fps (video) or as 16 kHz mono waveform (audio), and returns one 768-dim vector plus a `(start, end)` pair per window. Audio windows longer than 30 s are rejected with a clear error, and the server raises the processor's `audio_seq_length` to 750 (= 30 s) at load time so full windows are never silently truncated (the shipped config pins it at 280 = 11.2 s). Whole-file requests without an explicit window still auto-window at 30 s for audio longer than 30 s; videos without one are embedded whole (frame-capped at 32 uniformly sampled frames by the processor).

## Runtime notes

- `~/.butler/pids/chroma.pid` and `~/.butler/pids/embedding-server.pid` record child PIDs; logs are in `~/.butler/logs/*.log`.
- Layout of `~/.butler`: `databases/` (ChromaDB), `models/` (downloaded models), `venvs/` (Python venvs, `embeddinggemma2-venv`), `pids/` (pid files), `logs/` (service logs), plus `config.json`.
- The embedding server (`server.py`) runs from `~/.butler/venvs/embeddinggemma2-venv` and serves `/health` plus `/embed/{text,image,audio,video}`. Python bytecode is redirected to `~/.butler/pycache` so nothing litters the repo. `butler start` waits until `/health` reports `ready` — the first load can take a while on CPU.
- EmbeddingGemma 2 uses asymmetric prompt names: documents are embedded with prompt `Document`, queries with `Query`.
- Chroma's score is converted from cosine distance to `1 - distance`; it is a similarity indicator, not a probability.
- `-j` emits JSON; `--toml` emits TOML-like result tables. `-t` is the text-only filter, so TOML uses `--toml`.
- The Chroma server executable must be installed separately; the `chromadb` npm package is only the JavaScript client.
- **Image speed vs. quality:** per-image latency is dominated by the vision encoder, whose input resolution grows with the configured soft-token budget (`imageTokens`, one of 70/140/280/560/1120). On a Zen 3 CPU (AVX2, fp32), 280 ≈ 5.3s per image, 70 ≈ 1.1s (~5x faster) with a small retrievability trade-off — Google's 280 default is the calibrated quality setting, set `imageTokens: 70` in config for bulk photo indexing and restart.
- **Video cost:** at the default 10 s windows and `videoTokens: 70`, a 10 s clip takes ~4 s (10 pre-sampled frames @ 70 ≈ 3.8 s measured); raising the frame budget to 140 more than doubles it (~9.7 s for the same 5-frame window). `butler start --video-tokens=N` switches budgets (hot-restart).
- **Audio cost:** linear in duration — ~1.2 s per 10 s window on the same CPU.
- `butler reset` clears the database, logs, pid files and config but keeps `~/.butler/venvs` and `~/.butler/models`; `butler purge` removes the whole `~/.butler` directory.
