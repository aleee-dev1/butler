
import argparse
import gc
import os
import shutil
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Union

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from torchcodec.decoders import AudioDecoder, VideoDecoder

DEFAULT_MODEL_PATH = Path.home() / ".butler/models/embeddinggemma-2"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_IMAGE_TOKENS = 280
# Supported vision budgets of Gemma4ImageProcessor; smaller = fewer pixels
# into the vision encoder = much faster, slightly lower quality.
SUPPORTED_IMAGE_TOKENS = (70, 140, 280, 560, 1120)
DEFAULT_VIDEO_TOKENS = 70
# VideoProcessor's configured per-frame budget; kwargs applied when the
# requested budget differs from it.
DEFAULT_VIDEO_SOFT_TOKENS = 140
# Gemma4AudioFeatureExtractor hard-truncates at 480,000 samples
# (= 30 s @ 16 kHz): audio windows must never exceed this.
MAX_AUDIO_CHUNK_SECONDS = 30.0
DEFAULT_AUDIO_CHUNK_SECONDS = 30.0

model = None
temp_dir = None
inference_lock = threading.Lock()
image_tokens = DEFAULT_IMAGE_TOKENS
video_tokens = DEFAULT_VIDEO_TOKENS


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, temp_dir

    print("Loading EmbeddingGemma 2...", flush=True)

    temp_dir = tempfile.TemporaryDirectory(prefix="embeddinggemma-")

    model = SentenceTransformer(
        str(MODEL_PATH),
        model_kwargs={"torch_dtype": torch.float32},
        device="cpu",
    )

    # The shipped processor_config.json pins audio_seq_length: 280
    # (= 11.2 s @ 40 ms/token), which would silently truncate 30 s audio
    # windows. The processor source default is 750 (= 30 s) — restore it.
    try:
        processor = model[0].processor
        if getattr(processor, "audio_seq_length", 0) < 750:
            processor.audio_seq_length = 750
    except (AttributeError, IndexError, TypeError):
        pass

    print(
        f"Model loaded. Modalities: {model.modalities} | "
        f"image tokens: {image_tokens} | video tokens/frame: {video_tokens}",
        flush=True,
    )

    yield

    print("Shutting down...", flush=True)

    model = None
    gc.collect()

    if temp_dir:
        temp_dir.cleanup()
        temp_dir = None

    print("Cleanup complete.", flush=True)


app = FastAPI(
    title="EmbeddingGemma 2 Inference Server",
    lifespan=lifespan,
)


# Context is 8192 tokens; the client pre-windowed text above a conservative
# char ceiling so nothing is silently truncated server-side.


class TextRequest(BaseModel):
    # A string embeds one input; a list embeds a batch in a single request.
    text: Union[str, list[str]]
    prompt_name: str = "Document"


def embed(payload, modality, prompt_name=None):
    if model is None:
        raise HTTPException(503, "Model not loaded")

    start = time.perf_counter()

    try:
        with inference_lock:
            kwargs = {
                "normalize_embeddings": True,
            }

            if prompt_name:
                kwargs["prompt_name"] = prompt_name

            if modality == "image" and image_tokens != 280:
                kwargs["processing_kwargs"] = {"image": {"max_soft_tokens": image_tokens}}

            # A list of texts embeds as one batched request through the
            # model; anything else is a single structured input (text /
            # media file).
            if modality == "text" and isinstance(payload, (list, tuple)):
                matrix = model.encode(list(payload), **kwargs)
                vectors = [row.reshape(-1).tolist() for row in matrix]
            else:
                embedding = model.encode(payload, **kwargs)
                vectors = [embedding.reshape(-1).tolist()]

        return {
            "modality": modality,
            "dimensions": len(vectors[0]),
            "embeddings": vectors,
            "count": len(vectors),
            "processing_time_ms": round(
                (time.perf_counter() - start) * 1000, 2
            ),
        }

    except Exception as exc:
        raise HTTPException(422, f"Embedding failed: {exc}") from exc


def embed_file(file: UploadFile, modality: str):
    """Whole-file single-vector embedding (image path)."""
    if model is None or temp_dir is None:
        raise HTTPException(503, "Model not loaded")

    suffix = Path(file.filename or "").suffix

    try:
        with tempfile.NamedTemporaryFile(
            dir=temp_dir.name,
            suffix=suffix,
            delete=False,
        ) as temp:
            temp_path = Path(temp.name)
            shutil.copyfileobj(file.file, temp)

        if temp_path.stat().st_size == 0:
            raise HTTPException(400, "Empty file")

        kwargs = None
        if modality == "image" and image_tokens != 280:
            kwargs = {"image": {"max_soft_tokens": image_tokens}}

        return embed_batch(
            [{modality: str(temp_path)}],
            modality,
            kwargs,
        )

    finally:
        if "temp_path" in locals():
            temp_path.unlink(missing_ok=True)


def embed_batch(payloads, modality, processing_kwargs=None):
    """Embed a list of pre-built single inputs under one lock acquisition."""
    if model is None:
        raise HTTPException(503, "Model not loaded")

    start = time.perf_counter()

    try:
        with inference_lock:
            kwargs = {"normalize_embeddings": True}
            if processing_kwargs:
                kwargs["processing_kwargs"] = processing_kwargs

            vectors = []
            for payload in payloads:
                embedding = model.encode(payload, **kwargs)
                vectors.append(embedding.reshape(-1).tolist())

        return {
            "modality": modality,
            "dimensions": len(vectors[0]) if vectors else 0,
            "embeddings": vectors,
            "count": len(vectors),
            "processing_time_ms": round(
                (time.perf_counter() - start) * 1000, 2
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f"Embedding failed: {exc}") from exc


def audio_windows(path: Path, chunk: float | None):
    """Decode once at 16 kHz mono, then slice into <=30 s windows.

    Returns (windows, payloads) where windows are (start, end) seconds and
    payloads are 1-D numpy waveforms (the shape the audio feature
    extractor expects).
    """
    try:
        decoder = AudioDecoder(str(path), sample_rate=16000, num_channels=1)
        samples = decoder.get_all_samples()
    except Exception as exc:
        raise HTTPException(422, f"Audio decode failed: {exc}") from exc

    # Mimic load_audio: 1-D waveform. [1, N] fails np.pad broadcasting.
    wave = samples.data[0].numpy()
    duration = len(wave) / samples.sample_rate

    if duration <= 0:
        raise HTTPException(400, "Audio contains no samples")

    # No explicit window: default to 30 s chunks — sending audio over
    # 30 s whole would be silently truncated by the feature extractor.
    if chunk is None or chunk <= 0:
        chunk = DEFAULT_AUDIO_CHUNK_SECONDS
    if duration <= chunk + 1e-6:
        return [(0.0, duration)], [wave]

    windows = []
    payloads = []
    start = 0.0
    while start < duration - 1e-6:
        end = min(start + chunk, duration)
        lo, hi = int(start * 16000), int(end * 16000)
        windows.append((start, end))
        payloads.append(wave[lo:hi])
        start = end
    return windows, payloads


def video_windows(path: Path, chunk: float | None):
    """Decode once, then pre-sample each window at 1 fps.

    Returns (windows, payloads) where payloads are [T, C, H, W] frame
    tensors fed directly to the video processor (bypassing its decoder).
    """
    try:
        decoder = VideoDecoder(str(path))
        duration = decoder.metadata.duration_seconds
    except Exception as exc:
        raise HTTPException(422, f"Video decode failed: {exc}") from exc

    if not duration or duration <= 0:
        raise HTTPException(400, "Video has no duration metadata")

    if chunk is None or chunk <= 0 or duration <= chunk + 1e-6:
        spans = [duration]
    else:
        spans = []
        offset = 0.0
        while offset < duration - 1e-6:
            spans.append(min(chunk, duration - offset))
            offset += chunk

    windows = []
    payloads = []
    start = 0.0
    for span in spans:
        end = start + span
        frames = decoder.get_frames_played_in_range(start, end, fps=1.0)
        if frames.data.shape[0] == 0:
            # Shorter than one frame period; fold into the previous window.
            if windows:
                windows[-1] = (windows[-1][0], end)
            else:
                windows.append((start, end))
                payloads.append(frames.data)
        else:
            windows.append((start, end))
            payloads.append(frames.data)
        start = end
    return windows, payloads


def embed_media(file: UploadFile, modality: str, segment: float | None = None):
    if model is None or temp_dir is None:
        raise HTTPException(503, "Model not loaded")

    if (
        modality == "audio"
        and segment is not None
        and segment > MAX_AUDIO_CHUNK_SECONDS + 1e-6
    ):
        raise HTTPException(
            400,
            f"Audio segment must be <= {MAX_AUDIO_CHUNK_SECONDS:g}s "
            "(feature extractor truncates at 480,000 samples @ 16 kHz).",
        )

    suffix = Path(file.filename or "").suffix

    try:
        with tempfile.NamedTemporaryFile(
            dir=temp_dir.name,
            suffix=suffix,
            delete=False,
        ) as temp:
            temp_path = Path(temp.name)
            shutil.copyfileobj(file.file, temp)

        if temp_path.stat().st_size == 0:
            raise HTTPException(400, "Empty file")

        if modality == "audio":
            windows, payloads = audio_windows(temp_path, segment)
            processing_kwargs = None
        else:
            windows, payloads = video_windows(temp_path, segment)
            processing_kwargs = None
            if video_tokens != DEFAULT_VIDEO_SOFT_TOKENS:
                processing_kwargs = {"video": {"max_soft_tokens": video_tokens}}

        result = embed_batch(payloads, modality, processing_kwargs)
        result["segments"] = [
            {"start": round(s, 3), "end": round(e, 3)} for s, e in windows
        ]
        return result

    finally:
        if "temp_path" in locals():
            temp_path.unlink(missing_ok=True)


@app.get("/health")
def health():
    return {
        "status": "ready" if model is not None else "loading",
        "modalities": model.modalities if model else [],
        "image_tokens": image_tokens,
        "video_tokens": video_tokens,
    }


@app.post("/embed/text")
def embed_text(request: TextRequest):
    if isinstance(request.text, list):
        if not request.text or not all(isinstance(t, str) and t.strip() for t in request.text):
            raise HTTPException(400, "Batch texts cannot be empty")
        payload = list(request.text)
    else:
        if not request.text.strip():
            raise HTTPException(400, "Text cannot be empty")
        payload = {"text": request.text}

    return embed(
        payload,
        "text",
        request.prompt_name,
    )


@app.post("/embed/image")
def embed_image(file: UploadFile = File(...)):
    return embed_file(file, "image")


@app.post("/embed/audio")
def embed_audio(
    file: UploadFile = File(...),
    segment: float | None = Form(None),
):
    # segment = window length in seconds; omit to embed the file whole
    # (whole-file audio longer than 30 s is auto-windowed at 30 s so
    # nothing is silently truncated).
    return embed_media(file, "audio", segment)


@app.post("/embed/video")
def embed_video(
    file: UploadFile = File(...),
    segment: float | None = Form(None),
):
    return embed_media(file, "video", segment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EmbeddingGemma 2 multimodal embedding server.",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help=f"Path to the model directory (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Bind address (default: {DEFAULT_HOST}).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to listen on (default: {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--image-tokens",
        type=int,
        default=DEFAULT_IMAGE_TOKENS,
        choices=SUPPORTED_IMAGE_TOKENS,
        help=(
            "Vision soft-token budget per image. 280 (default) is Google's "
            "calibrated quality default; 70 is ~5x faster for bulk photo "
            "indexing with a small quality trade-off."
        ),
    )
    parser.add_argument(
        "--video-tokens",
        type=int,
        default=DEFAULT_VIDEO_TOKENS,
        choices=SUPPORTED_IMAGE_TOKENS,
        help=(
            "Vision soft-token budget per video frame. 70 (default) keeps "
            "segmented video indexing practical on CPU; the processor's "
            "quality default is 140. Applies to pre-sampled frame windows."
        ),
    )
    args = parser.parse_args()

    image_tokens = args.image_tokens
    video_tokens = args.video_tokens

    model_path = Path(args.model).expanduser().resolve()
    if not (model_path / "model.safetensors").is_file() or not (model_path / "config.json").is_file():
        parser.error(f"No model found at {model_path} (missing model.safetensors/config.json).")

    MODEL_PATH = model_path
    HOST = args.host
    PORT = args.port

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
        access_log=False,
    )