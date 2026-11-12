import fs from 'node:fs/promises';
import path from 'node:path';
import { endpoint, mimeFor } from './util.js';

const TIMEOUT = 300000; // CPU media embedding can be slow, give it room

// Prompt names as defined by EmbeddingGemma 2's configured prompts dict
// (case-sensitive): 'query' → "task: search result | query: …",
// 'document' → "title: none | text: …".
const PROMPTS = {
    document: 'document',
    query: 'query',
};

export async function embedTexts(texts, config, mode = 'document') {
    if (!texts.length) throw new Error('No text to embed.');
    const response = await fetch(endpoint(config, '/embed/text'), {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ text: texts, prompt_name: PROMPTS[mode] || PROMPTS.document }),
        signal: AbortSignal.timeout(TIMEOUT)
    });
    if (!response.ok) throw new Error(`Embedding server request failed (${response.status}): ${(await response.text()).slice(0, 500)}`);
    const data = await response.json();
    const vectors = data?.embeddings;
    if (!Array.isArray(vectors) || vectors.length !== texts.length) throw new Error(`Embedding server returned ${vectors ? vectors.length : 'no'} vectors for ${texts.length} texts.`);
    return vectors;
}

export function embedText(text, config, mode = 'document') {
    return embedTexts([text], config, mode).then(vectors => vectors[0]);
}

async function embedMedia(file, config, modality, segmentSeconds) {
    const mime = mimeFor(file);
    const content = await fs.readFile(file);
    const form = new FormData();
    form.append('file', new Blob([content], { type: mime }), path.basename(file));
    // Optional window length in seconds; the server slices long media into
    // per-window vectors and returns matching segment ranges.
    if (segmentSeconds) form.append('segment', String(segmentSeconds));
    const response = await fetch(endpoint(config, `/embed/${modality}`), {
        method: 'POST',
        body: form,
        signal: AbortSignal.timeout(TIMEOUT)
    });
    if (!response.ok) throw new Error(`Embedding server ${modality} request failed (${response.status}): ${(await response.text()).slice(0, 500)}`);
    const data = await response.json();
    const vectors = data?.embeddings;
    if (!Array.isArray(vectors) || !vectors.length || !Array.isArray(vectors[0]) || vectors[0].length === 0) {
        throw new Error(`Embedding server returned no ${modality} embedding.`);
    }
    const segments = Array.isArray(data.segments) ? data.segments : [];
    // One (start, end) window per vector; absent/short arrays degrade to nulls.
    return vectors.map((vector, i) => ({
        vector,
        tStart: segments[i]?.start ?? null,
        tEnd: segments[i]?.end ?? null
    }));
}

export function embedImage(file, config) {
    return embedMedia(file, config, 'image').then(parts => parts[0]);
}

// Audio is windows of at most 30 s (the feature extractor hard-truncates
// at 480,000 samples @ 16 kHz); video defaults to 10 s windows. The server
// decodes once and returns one vector + time range per window.
export function embedMediaChunked(file, config, modality, chunkSeconds) {
    const segment = modality === 'audio' ? Math.min(chunkSeconds ?? 30, 30) : (chunkSeconds ?? 10);
    return embedMedia(file, config, modality, segment);
}
