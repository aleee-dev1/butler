import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';
import readline from 'node:readline/promises';
import { fileURLToPath } from 'node:url';
import { stdin as input, stdout as output } from 'node:process';

export const HOME = os.homedir();
export const ROOT = path.join(HOME, '.butler');
export const CONFIG = path.join(ROOT, 'config.json');
export const DATABASES = path.join(ROOT, 'databases');
export const MODELS = path.join(ROOT, 'models');
export const MODEL_DIR = path.join(MODELS, 'embeddinggemma-2');
export const VENVS = path.join(ROOT, 'venvs');
export const VENV = path.join(VENVS, 'embeddinggemma2-venv');
export const LOGS = path.join(ROOT, 'logs');
export const PIDS = path.join(ROOT, 'pids');
export const HF_REPO = 'google/embeddinggemma-2';
// server.py lives at the package root (src/../server.py), resolved at runtime
// so a global npm install still finds it.
export const SERVER_PY = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'server.py');
export const COLLECTION = 'butler_files';

// Default chunking parameters (can be overridden in config or CLI).
// 6000 chars ≈ 1.5–2k tokens for English — upper range of the retrieval
// sweet spot with EmbeddingGemma 2's native 8192-token context.
export const DEFAULT_CHUNK = 6000;
export const DEFAULT_OVERLAP = 300;
export const DEFAULT_BATCH = 500;

export function error(message) { console.error(message); }
export function log(message) { console.log(message); }
export function verbose(enabled, message) { if (enabled) console.log(message); }

export async function readConfig() {
    try { return JSON.parse(await fs.readFile(CONFIG, 'utf8')); }
    catch { throw new Error('[!] Butler is not initialized. Run `butler init` first.'); }
}

export function endpoint(config, suffix = '') {
    return `http://${config.host || '127.0.0.1'}:${config.embedPort || 8765}${suffix}`;
}

export function chromaUrl(config) {
    return `http://${config.host || '127.0.0.1'}:${config.chromaPort}`;
}

export async function assertRunning(config) {
    const checks = [
        ['Embedding server', endpoint(config, '/health')],
        ['ChromaDB', chromaUrl(config) + '/api/v2/heartbeat']
    ];
    for (const [name, url] of checks) {
        try {
            const r = await fetch(url, { signal: AbortSignal.timeout(1500) });
            if (!r.ok) throw new Error();
        } catch {
            throw new Error(`${name} is not running. Run \`butler start\`.`);
        }
    }
}

export async function confirm(question) {
    const rl = readline.createInterface({ input, output });
    try { return (await rl.question(`${question} [y/N] `)).trim().toLowerCase() === 'y'; }
    finally { rl.close(); }
}

export async function sha256File(file) {
    const hash = crypto.createHash('sha256');
    const stream = (await import('node:fs')).createReadStream(file);
    for await (const chunk of stream) hash.update(chunk);
    return hash.digest('hex');
}

export async function sha256Text(text) {
    return crypto.createHash('sha256').update(text, 'utf8').digest('hex');
}

export function stableId(value) {
    return crypto.createHash('sha256').update(value).digest('hex');
}

export async function readableUtf8(file) {
    const buf = await fs.readFile(file);
    if (buf.includes(0)) return null;
    try {
        const text = new TextDecoder('utf-8', { fatal: true }).decode(buf);
        return text.trim() ? text : null;
    } catch { return null; }
}

export const IMAGE_EXT = new Set([ '.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tif', '.tiff', '.avif']);
export const AUDIO_EXT = new Set([ '.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.oga', '.opus', '.wma']);
export const VIDEO_EXT = new Set([ '.mp4', '.webm', '.mkv', '.mov', '.avi', '.m4v', '.mpg', '.mpeg']);

const MEDIA_MIME = {
    // images
    '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp', '.gif': 'image/gif', '.bmp': 'image/bmp', '.tif': 'image/tiff', '.tiff': 'image/tiff', '.avif': 'image/avif',
    // audio
    '.mp3': 'audio/mpeg', '.wav': 'audio/wav', '.flac': 'audio/flac', '.m4a': 'audio/mp4', '.aac': 'audio/aac', '.ogg': 'audio/ogg', '.oga': 'audio/ogg', '.opus': 'audio/opus', '.wma': 'audio/x-ms-wma',
    // video
    '.mp4': 'video/mp4', '.webm': 'video/webm', '.mkv': 'video/x-matroska', '.mov': 'video/quicktime', '.avi': 'video/x-msvideo', '.m4v': 'video/mp4', '.mpg': 'video/mpeg', '.mpeg': 'video/mpeg'
};

export function isImage(file) { return IMAGE_EXT.has(path.extname(file).toLowerCase()); }
export function isAudio(file) { return AUDIO_EXT.has(path.extname(file).toLowerCase()); }
export function isVideo(file) { return VIDEO_EXT.has(path.extname(file).toLowerCase()); }

export function mimeFor(file) {
    return MEDIA_MIME[path.extname(file).toLowerCase()] || 'application/octet-stream';
}

// Image/audio/video are recognized by extension; anything else must be valid
// UTF-8 text (checked by the callers).
export function mediaType(file) {
    const ext = path.extname(file).toLowerCase();
    if (IMAGE_EXT.has(ext)) return 'image';
    if (AUDIO_EXT.has(ext)) return 'audio';
    if (VIDEO_EXT.has(ext)) return 'video';
    return null;
}

export function fileType(file) {
    return mediaType(file) ?? 'text';
}

export function formatBytes(bytes) { return `${(bytes / (1024 * 1024)).toLocaleString('en-US', { maximumFractionDigits: 2 })} MB`; }

export async function dirSize(dir) {
    let total = 0;
    async function walk(p) {
        let entries;
        try { entries = await fs.readdir(p, { withFileTypes: true }); } catch { return; }
        for (const e of entries) {
            const full = path.join(p, e.name);
            if (e.isDirectory()) await walk(full);
            else if (e.isFile()) { try { total += (await fs.stat(full)).size; } catch { } }
        }
    }
    await walk(dir);
    return total;
}

export async function pathExists(p) { try { await fs.access(p); return true; } catch { return false; } }

export function normalizePath(p) { return path.resolve(p); }