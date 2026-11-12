import fs from 'node:fs/promises';
import path from 'node:path';
import { readConfig, assertRunning, normalizePath, sha256File, sha256Text, stableId, readableUtf8, mediaType, fileType, verbose, log, dirSize, formatBytes, DEFAULT_CHUNK, DEFAULT_OVERLAP, DEFAULT_BATCH, ROOT } from './util.js';
import { getCollection, getByPath, dbSize } from './db.js';
import { embedTexts, embedImage, embedMediaChunked } from './embed.js';

const HELP = 'Usage: butler add . [-r] [-q] [-e="name1,name2"] [--chunk=N] [--overlap=N] [--batch=N] | butler add <paths...> [-q] [--chunk=N] [--overlap=N] [--batch=N] | butler add "free text"';

function chunkText(t, chunkSize, overlap) {
    if (t.length <= chunkSize) return [{ text: t, lineStart: 1, lineEnd: t.split('\n').length }];
    const out = [];
    let lineStart = 1;
    for (let i = 0; i < t.length; i += chunkSize - overlap) {
        const chunk = t.slice(i, i + chunkSize);
        const linesBefore = t.slice(0, i).split('\n').length;
        const linesInChunk = chunk.split('\n').length;
        out.push({ text: chunk, lineStart: linesBefore + 1, lineEnd: linesBefore + linesInChunk });
        if (i + chunkSize >= t.length) break;
    }
    return out;
}

// The server truncates oversize inputs silently (no error is raised), so
// text is pre-windowed below EmbeddingGemma 2's 8192-token context before
// it is sent. ~2 chars/token is the conservative floor — code and some
// languages run denser — and remains far above the default chunk size.
const TEXT_CAP_CHARS = 16000;

// Splits one chunkText part into <=TEXT_CAP_CHARS windows, keeping the
// 1-based line mapping consistent with chunkText's convention.
function windowChunk(part, overlap) {
    if (part.text.length <= TEXT_CAP_CHARS) return [part];
    log(`  ⚠ Chunk exceeds the ${TEXT_CAP_CHARS}-char context ceiling, windowing...`);
    const out = [];
    for (let i = 0; i < part.text.length; i += TEXT_CAP_CHARS - overlap) {
        const text = part.text.slice(i, i + TEXT_CAP_CHARS);
        const linesBefore = part.text.slice(0, i).split('\n').length;
        const lineStart = part.lineStart + linesBefore - 1;
        out.push({ text, lineStart, lineEnd: lineStart + text.split('\n').length - 1 });
        if (i + TEXT_CAP_CHARS >= part.text.length) break;
    }
    return out;
}

async function gatherDirectory(dir, recursive, excludes = new Set()) {
    const result = [];
    const entries = await fs.readdir(dir, { withFileTypes: true });
    for (const ent of entries) {
        const full = path.resolve(dir, ent.name);
        if (ent.isDirectory()) {
            if (excludes.has(ent.name)) { log(`Excluding directory: ${full}`); continue; }
            // Directories are not indexed; recursion gathers their files only.
            if (recursive) result.push(...await gatherDirectory(full, true, excludes));
        } else if (ent.isFile()) {
            const media = mediaType(full);
            if (media) result.push({ path: full, type: media });
            else {
                const text = await readableUtf8(full);
                if (text !== null) result.push({ path: full, type: 'text' });
            }
        }
    }
    return result;
}

function metadataFor(file, checksum, type) {
    return {
        path: file.path,
        parent: path.dirname(file.path),
        name: path.basename(file.path),
        checksum,
        type
    };
}

// Text/metadata document for a candidate. Media files embed as a
// single record whose only text is a short descriptor.
async function fileContent(file) {
    if (file.type === 'image') return `Image file: ${path.basename(file.path)}. Path: ${file.path}`;
    if (file.type === 'audio') return `Audio file: ${path.basename(file.path)}. Path: ${file.path}`;
    if (file.type === 'video') return `Video file: ${path.basename(file.path)}. Path: ${file.path}`;
    const text = await readableUtf8(file.path);
    if (text === null) throw new Error('Not a supported UTF-8 text file.');
    return text;
}

async function addFiles(paths, recursive, isVerbose, excludes = new Set(), options = {}) {
    const config = await readConfig();
    await assertRunning(config);
    const collection = await getCollection(config);

    const chunkSize = options.chunk ?? config.chunk ?? DEFAULT_CHUNK;
    const overlap = options.overlap ?? config.overlap ?? DEFAULT_OVERLAP;
    const batchSize = options.batch ?? config.batch ?? DEFAULT_BATCH;

    const candidates = [];
    for (const p of paths) {
        const abs = normalizePath(p);
        let st;
        try { st = await fs.stat(abs); } catch { log(`Skipping missing path: ${abs}`); continue; }
        if (st.isDirectory()) {
            // Directories are not indexed on their own; their files come via
            // gatherDirectory (recursion required). A directory's description,
            // if any, lives only in free-text notes: butler add "…".
            candidates.push(...await gatherDirectory(abs, recursive, excludes));
        } else if (st.isFile()) {
            const media = mediaType(abs);
            if (media) candidates.push({ path: abs, type: media });
            else if (await readableUtf8(abs) !== null) candidates.push({ path: abs, type: 'text' });
            else log(`Skipping unsupported file: ${abs}`);
        } else log(`Skipping unsupported path: ${abs}`);
    }

    const unique = [...new Map(candidates.map(f => [f.path, f])).values()];

    // path -> { ids: [all chunk ids], metadata }; notes excluded
    const existing = new Map();
    let inDb = 0, notInDb = 0;
    if (isVerbose) log(`Scanning ${unique.length} files...`);
    for (const f of unique) {
        const result = await getByPath(collection, f.path);
        if (result.ids?.length) {
            result.ids.forEach((id, i) => {
                const m = result.metadatas?.[i];
                if (!m || m.source === 'note') return;
                const e = existing.get(m.path) || { ids: [], metadata: m };
                e.ids.push(id);
                existing.set(m.path, e);
            });
        }
        if (existing.has(f.path)) inDb++; else notInDb++;
    }
    if (isVerbose) {
        log(`  ✓ ${inDb} already in DB`);
        log(`  ○ ${notInDb} not in DB`);
    }

    const pending = [];
    let upToDate = 0, changed = 0, newFiles = 0;
    if (isVerbose) log(`Checking ${unique.length} files...`);
    for (const f of unique) {
        const checksum = await sha256File(f.path);
        const old = existing.get(f.path);
        if (old?.metadata?.checksum === checksum && old.metadata.type === f.type) {
            upToDate++;
            continue;
        }
        if (old) changed++; else newFiles++;
        pending.push({ ...f, checksum, old });
    }
    if (isVerbose) {
        if (upToDate) log(`  ✓ ${upToDate} up to date`);
        if (changed) log(`  ↻ ${changed} changed`);
        if (newFiles) log(`  + ${newFiles} new`);
    }

    const records = [];
    const staleIds = [];
    let savedFiles = 0;
    if (isVerbose && pending.length) log(`Vectorizing ${pending.length} files...`);
    for (let idx = 0; idx < pending.length; idx++) {
        const f = pending[idx];
        try {
            const start = performance.now();
            const doc = await fileContent(f);
            let pieces = [];
            if (mediaType(f.path)) {
                if (f.type !== 'image') {
                    // Audio/video are embedded in time windows so recall can
                    // point at a moment, not just the whole file.
                    const chunkSeconds = f.type === 'video'
                        ? config.videoChunk ?? 10
                        : config.audioChunk ?? 30;
                    const parts = await embedMediaChunked(f.path, config, f.type, chunkSeconds);
                    pieces.push(...parts.map(p => ({
                        text: doc,
                        vector: p.vector,
                        lineStart: undefined,
                        lineEnd: undefined,
                        tStart: p.tStart,
                        tEnd: p.tEnd
                    })));
                } else {
                    pieces = [{ text: doc, vector: (await embedImage(f.path, config)).vector, lineStart: undefined, lineEnd: undefined }];
                }
            } else {
                // Chunk, window anything over the context ceiling, then send
                // every piece of the file as one batched request.
                const windows = [];
                for (const part of chunkText(doc, chunkSize, overlap)) {
                    windows.push(...windowChunk(part, overlap));
                }
                const vectors = await embedTexts(windows.map(w => w.text), config);
                pieces.push(...windows.map((w, i) => ({
                    text: w.text,
                    vector: vectors[i],
                    lineStart: w.lineStart,
                    lineEnd: w.lineEnd
                })));
            }
            const elapsed = (performance.now() - start).toFixed(1);
            if (isVerbose) {
                const pct = Math.round((idx + 1) / pending.length * 10);
                const bar = '█'.repeat(pct) + '░'.repeat(10 - pct);
                const name = path.basename(f.path).padEnd(20).slice(0, 20);
                const typeChar = { text: 'T', image: 'I', audio: 'A', video: 'V' }[f.type] ?? '?';
                log(`  [${bar}] ${idx + 1}/${pending.length}  ${name}  ${typeChar}  ${pieces.length} chunks  ${elapsed}ms`);
            }

            const ids = pieces.map((_, i) => stableId(`${f.path}#${i}`));
            pieces.forEach((p, i) => {
                const meta = { ...metadataFor(f, f.checksum, f.type), chunk: i, chunks: pieces.length };
                if (p.lineStart !== undefined) { meta.lineStart = p.lineStart; meta.lineEnd = p.lineEnd; }
                if (p.tStart !== null && p.tStart !== undefined) { meta.tStart = p.tStart; meta.tEnd = p.tEnd; }
                records.push({ id: ids[i], doc: p.text, vector: p.vector, metadata: meta });
            });
            if (f.old) staleIds.push(...f.old.ids.filter(id => !ids.includes(id)));
            savedFiles++;
        } catch (e) {
            log(`Failed ${f.path}: ${e.message}${e.cause ? ` (${e.cause.code || e.cause.message})` : ''}`);
        }
    }

    if (records.length) {
        for (let i = 0; i < records.length; i += batchSize) {
            const b = records.slice(i, i + batchSize);
            await collection.upsert({
                ids: b.map(r => r.id),
                embeddings: b.map(r => r.vector),
                documents: b.map(r => r.doc),
                metadatas: b.map(r => r.metadata)
            });
        }
        if (staleIds.length) await collection.delete({ ids: staleIds });
        log(`Saved ${savedFiles} vectorized files (${records.length} entries) in ChromaDB.`);
    } else log('Nothing to save.');
    const chromaPath = path.join(ROOT, 'databases');
    log(`Current DB size: ${formatBytes(await dirSize(chromaPath))} (${await dbSize(collection)} entries)`);
}

async function addText(text) {
    const config = await readConfig();
    await assertRunning(config);
    const collection = await getCollection(config);
    const cwd = process.cwd();
    const checksum = await sha256Text(text);
    const windows = windowChunk(
        { text, lineStart: 1, lineEnd: text.split('\n').length },
        config.overlap ?? DEFAULT_OVERLAP
    );
    const vectors = await embedTexts(windows.map(w => w.text), config);
    await collection.upsert({
        ids: windows.map((_, i) => stableId(`note:${cwd}:${checksum}:${i}`)),
        embeddings: vectors,
        documents: windows.map(w => w.text),
        metadatas: windows.map((w, i) => ({
            path: cwd,
            parent: path.dirname(cwd),
            name: text.slice(0, 80),
            checksum,
            type: 'text',
            source: 'note',
            chunk: i,
            chunks: windows.length,
            ...(w.lineStart !== undefined ? { lineStart: w.lineStart, lineEnd: w.lineEnd } : {})
        }))
    });
    log('Saved text for current path.');
}

async function exists(p) { try { await fs.access(p); return true; } catch { return false; } }

function parseNumberFlag(args, flag, min, max, defaultVal) {
    const arg = args.find(a => a.startsWith(`--${flag}=`));
    if (!arg) return defaultVal;
    const val = parseInt(arg.split('=')[1], 10);
    if (!Number.isInteger(val) || val < min || val > max) {
        throw new Error(`--${flag} must be an integer between ${min} and ${max}`);
    }
    return val;
}

export async function addCommand(args) {
    if (!args.length) throw new Error(HELP);
    const quiet = args.includes('-q');
    const verboseFlag = !quiet;
    const recursive = args.includes('-r');
    const excludeArg = args.find(a => a.startsWith('-e='));
    const excludes = excludeArg
        ? new Set(excludeArg.slice(3).replace(/^"|"$/g, '').split(',').map(s => s.trim()).filter(Boolean))
        : new Set();

    // Parse --chunk, --overlap, --batch flags
    const chunk = parseNumberFlag(args, 'chunk', 100, 100000, undefined);
    const overlap = parseNumberFlag(args, 'overlap', 0, 10000, undefined);
    const batch = parseNumberFlag(args, 'batch', 1, 10000, undefined);

    // Validate overlap < chunk (considering config defaults if not provided via flags)
    const config = await readConfig();
    const effectiveChunk = chunk ?? config.chunk ?? DEFAULT_CHUNK;
    const effectiveOverlap = overlap ?? config.overlap ?? DEFAULT_OVERLAP;
    if (effectiveOverlap >= effectiveChunk) {
        throw new Error('--overlap must be less than --chunk (and less than config.chunk if not overridden)');
    }

    const rest = args.filter(a => 
        !['-v', '-r', '-q'].includes(a) && 
        !a.startsWith('-e=') &&
        !a.startsWith('--chunk=') &&
        !a.startsWith('--overlap=') &&
        !a.startsWith('--batch=')
    );
    if (rest.length === 1 && !['.', '..'].includes(rest[0]) && !await exists(rest[0])) {
        await addText(rest[0]);
        return;
    }
    if (rest.some(a => a.startsWith('-'))) throw new Error(HELP);
    await addFiles(rest, recursive, verboseFlag, excludes, { chunk, overlap, batch });
}