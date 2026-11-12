import fs from 'node:fs';
import path from 'node:path';
import fsp from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { ROOT, CONFIG, DATABASES, VENVS, VENV, LOGS, PIDS, MODEL_DIR, HF_REPO, SERVER_PY, readConfig, confirm, log, pathExists, DEFAULT_CHUNK, DEFAULT_OVERLAP, DEFAULT_BATCH } from './util.js';

const DEFAULTS = {
    host: '127.0.0.1',
    chromaPort: 8000,
    embedPort: 8765,
    imageTokens: 280, // vision budget per image: 70/140/280/560/1120; 70 ≈ 5x faster, small quality tax
    videoTokens: 70, // vision budget per video frame; 140 is the processor's quality default
    videoChunk: 10, // video embedding window in seconds (also becomes the recall timestamp unit)
    audioChunk: 30, // audio embedding window in seconds; hard max is 30 (feature-extractor cap)
    modelPath: MODEL_DIR,
    chunk: DEFAULT_CHUNK,
    overlap: DEFAULT_OVERLAP,
    batch: DEFAULT_BATCH,
    device: 'cpu'
};

// Top-level layout of ~/.butler:
//   databases/  ChromaDB storage
//   models/     downloaded models (models/embeddinggemma-2)
//   venvs/      Python virtual environments (venvs/embeddinggemma2-venv)
//   pids/       pid files of running services
//   logs/       service logs
//   config.json
const CHROMA_PATH = DATABASES;
const SERVER_PID = path.join(PIDS, 'embedding-server.pid');
const SERVER_LOG = path.join(LOGS, 'embedding-server.log');
const CHROMA_PID = path.join(PIDS, 'chroma.pid');

// Pinned Python dependency stack for the embedding server. Torch/torchcodec
// must come from the CPU index; the rest are regular PyPI installs.
const DEPS_CPU = [
    'torch==2.15.0',
    'torchvision==0.29.2',
    'torchcodec'
];
const DEPS_PURE = [
    'sentence-transformers==6.2.0',
    'transformers==5.22.2',
    'pillow',
    'librosa',
    'fastapi',
    'uvicorn',
    'python-multipart' // required by FastAPI's UploadFile endpoints
];
const CPU_INDEX = 'https://download.pytorch.org/whl/cpu';
const DEPS_MARKER_FILE = path.join(VENV, '.butler-deps');
const DEPS_MARKER = '1';

function commandExists(cmd) {
    return new Promise(resolve => {
        const p = spawn(cmd, ['--version'], {
            stdio: 'ignore',
            env: process.env
        });
        p.on('error', () => resolve(false));
        p.on('close', code => resolve(code === 0));
    });
}

// The embedding server reports "loading" over HTTP while the model loads on
// first start, so wait until /health actually says "ready".
async function waitForServerReady(url, name, timeoutMs = 180000) {
    const end = Date.now() + timeoutMs;
    while (Date.now() < end) {
        try {
            const r = await fetch(url, { signal: AbortSignal.timeout(2000) });
            if (r.ok) {
                const body = await r.json().catch(() => null);
                if (body?.status === 'ready') return;
            }
        } catch { }
        await new Promise(r => setTimeout(r, 500));
    }
    throw new Error(`${name} did not become ready within ${Math.round(timeoutMs / 1000)}s. Check its log: ${SERVER_LOG}`);
}

export function spawnDetached(cmd, args, pidFile, logFile, env = process.env) {
    const out = fs.openSync(logFile, 'a');
    let child;
    try {
        child = spawn(cmd, args, { detached: true, stdio: ['ignore', out, out], env });
    } finally {
        fs.closeSync(out);
    }
    child.on('error', e => { console.error(`${cmd}: ${e.message}`); process.exit(1); });
    if (!child.pid) throw new Error(`Failed to spawn ${cmd}`);
    fs.writeFileSync(pidFile, String(child.pid));
    child.unref();
}

async function pidAlive(file) {
    try {
        const pid = Number((await fsp.readFile(file, 'utf8')).trim());
        if (!Number.isInteger(pid) || pid < 2) return false;
        process.kill(pid, 0);
        return true;
    } catch { return false; }
}

function run(cmd, args, env, { capture = false } = {}) {
    return new Promise((resolve, reject) => {
        const p = spawn(cmd, args, { env, stdio: capture ? ['ignore', 'pipe', 'ignore'] : 'inherit' });
        let out = '';
        if (capture) p.stdout.on('data', d => out += d);
        p.on('error', reject);
        p.on('close', code => code === 0 ? resolve(out) : reject(new Error(`${cmd} ${args[0]} failed (${code})`)));
    });
}

function venvPython() {
    return path.join(VENV, 'bin', 'python');
}

async function ensureVenv() {
    if (await pathExists(venvPython())) return;
    const python = process.env.PYTHON || 'python3.13';
    if (!(await commandExists(python))) {
        throw new Error(`Python 3.13 is required but \`${python}\` was not found in PATH.\nSet PYTHON=/path/to/python to override.`);
    }
    log(`Creating virtual environment at ${VENV}...`);
    await fsp.mkdir(path.dirname(VENV), { recursive: true });
    try {
        await run(python, ['-m', 'venv', VENV]);
    } catch (e) {
        throw new Error(`Failed to create the virtual environment: ${e.message}`);
    }
}

async function ensureDeps() {
    if (await pathExists(DEPS_MARKER_FILE)) {
        const marker = (await fsp.readFile(DEPS_MARKER_FILE, 'utf8')).trim();
        if (marker === DEPS_MARKER) return;
    }
    if (!(await pathExists(venvPython()))) {
        throw new Error(`Virtual environment not found at ${VENV}. Re-run \`butler init\`.`);
    }
    log('Installing Python dependencies (downloading torch is a big one, be patient)...');
    try {
        // CPU-only torch stack first, then the pure-PyPI packages.
        await run(venvPython(), ['-m', 'pip', 'install', '--no-cache-dir', ...DEPS_CPU, '--index-url', CPU_INDEX]);
        await run(venvPython(), ['-m', 'pip', 'install', '--no-cache-dir', ...DEPS_PURE]);
    } catch (e) {
        throw new Error(`Failed to install Python dependencies: ${e.message}`);
    }
    await fsp.writeFile(DEPS_MARKER_FILE, `${DEPS_MARKER}\n`);
}

async function ensureModel(modelPath) {
    const ok =
        await pathExists(path.join(modelPath, 'config.json')) &&
        await pathExists(path.join(modelPath, 'model.safetensors'));

    if (ok) return;

    if (modelPath !== MODEL_DIR) {
        throw new Error(
            `Model not found at ${modelPath} (missing model.safetensors/config.json).\n` +
            `Run \`butler init\` without arguments to download it to ${MODEL_DIR}.`
        );
    }

    log(`Downloading model from HuggingFace (${HF_REPO}) to ${MODEL_DIR}...`);
    try {
        await run(venvPython(), ['-c',
            `from huggingface_hub import snapshot_download\n` +
            `snapshot_download(repo_id="${HF_REPO}", local_dir="${modelPath}")\n`
        ]);
    } catch (e) {
        throw new Error(`Failed to download the model: ${e.message}`);
    }
}

export async function init(modelPathArg, flags = {}) {
    // 1. Virtual environment (skippable via --skip-venv, e.g. for reused/pre-made venvs)
    if (flags.skipVenv) {
        if (!(await pathExists(venvPython()))) {
            throw new Error(`--skip-venv set but no virtual environment found at ${VENV}.\n` +
                `Create or copy one there first.`);
        }
        log(`Skipping venv setup (${VENV} exists).`);
    } else {
        await ensureVenv();
        await ensureDeps();
    }

    // 2. Resolve and prepare the model
    const modelPath = modelPathArg ? path.resolve(modelPathArg) : DEFAULTS.modelPath;
    await ensureModel(modelPath);

    // 3. Create databases directory
    try {
        await fsp.mkdir(CHROMA_PATH, { recursive: true });
    } catch (e) {
        throw new Error(`Failed to create databases directory at ${CHROMA_PATH}: ${e.message}`);
    }

    // 4. Check chroma is installed
    if (!(await commandExists('chroma'))) {
        throw new Error('Chroma CLI (chroma) was not found in PATH.\n' +
            'Install the Chroma server CLI (`pipx install chromadb`), then run `butler init` again.');
    }

    // 5. Write config
    try {
        const config = { ...DEFAULTS, modelPath };
        if (flags.imageTokens !== undefined) config.imageTokens = flags.imageTokens;
        if (flags.videoTokens !== undefined) config.videoTokens = flags.videoTokens;
        await writeConfig(config);
    } catch (e) {
        throw new Error(`Failed to write config file at ${CONFIG}: ${e.message}`);
    }

    log(`Initialized ${ROOT}`);
    log(`Config: ${CONFIG}`);
    log(`Model: ${modelPath}`);
}

async function writeConfig(config) {
    await fsp.writeFile(CONFIG, JSON.stringify(config, null, 2) + '\n', { mode: 0o600 });
}

// Budgets reported by an already-running embedding server, or nulls.
async function runningServerState(c) {
    try {
        const r = await fetch(
            `http://${c.host || '127.0.0.1'}:${c.embedPort || 8765}/health`,
            { signal: AbortSignal.timeout(1500) }
        );
        if (r.ok) {
            const body = await r.json();
            return {
                imageTokens: typeof body?.image_tokens === 'number' ? body.image_tokens : null,
                videoTokens: typeof body?.video_tokens === 'number' ? body.video_tokens : null
            };
        }
    } catch { }
    return { imageTokens: null, videoTokens: null };
}

export async function start(flags = {}) {
    const c = await readConfig();
    const wantedImageTokens = flags.imageTokens !== undefined ? flags.imageTokens : (c.imageTokens || DEFAULTS.imageTokens);
    const wantedVideoTokens = flags.videoTokens !== undefined ? flags.videoTokens : (c.videoTokens ?? DEFAULTS.videoTokens);

    // Persist the flags so later plain `butler start` runs keep them
    if (flags.imageTokens !== undefined && flags.imageTokens !== c.imageTokens) {
        await writeConfig({ ...c, imageTokens: flags.imageTokens });
        log(`Set imageTokens=${flags.imageTokens} in config.`);
    }
    if (flags.videoTokens !== undefined && flags.videoTokens !== c.videoTokens) {
        const cc = flags.imageTokens !== undefined ? { ...c, imageTokens: flags.imageTokens } : c;
        await writeConfig({ ...cc, videoTokens: flags.videoTokens });
        log(`Set videoTokens=${flags.videoTokens} in config.`);
    }

    await fsp.mkdir(ROOT, { recursive: true });
    // pids/ and logs/ must exist before any spawn opens its log file
    await Promise.all([
        fsp.mkdir(PIDS, { recursive: true }),
        fsp.mkdir(LOGS, { recursive: true })
    ]);

    let serveFirst = !(await pidAlive(SERVER_PID));
    if (!serveFirst) {
        // A running server may have been started with different vision
        // budgets; /health reports them, so hot-restart when they differ.
        const running = await runningServerState(c);
        if (
            (running.imageTokens !== null && running.imageTokens !== wantedImageTokens) ||
            (running.videoTokens !== null && running.videoTokens !== wantedVideoTokens)
        ) {
            log(
                `Embedding server is running with imageTokens=${running.imageTokens}, ` +
                `videoTokens=${running.videoTokens}; restarting with ` +
                `${wantedImageTokens}/${wantedVideoTokens}...`
            );
            await stopOne('Embedding server', SERVER_PID);
            serveFirst = true;
        }
    }
    if (!serveFirst) {
        log('Embedding server already running.');
    } else {
        // Redirect Python bytecode caches into ~/.butler so nothing ever
        // litters the repo (server.py would otherwise run without caching,
        // but imported sibling modules would write __pycache__ next to it).
        const env = { ...process.env, PYTHONPYCACHEPREFIX: path.join(ROOT, 'pycache') };
        spawnDetached(venvPython(), [
            SERVER_PY,
            '--model', c.modelPath || MODEL_DIR,
            '--host', c.host || '127.0.0.1',
            '--port', String(c.embedPort || 8765),
            '--image-tokens', String(wantedImageTokens),
            '--video-tokens', String(wantedVideoTokens)
        ], SERVER_PID, SERVER_LOG, env);
        log('Started embedding server. Waiting for model to load...');
        await waitForServerReady(
            `http://${c.host || '127.0.0.1'}:${c.embedPort || 8765}/health`,
            'Embedding server'
        );
        log('Embedding server ready.');
    }

    if (await pidAlive(CHROMA_PID)) {
        log('ChromaDB already running.');
    } else {
        await fsp.mkdir(CHROMA_PATH, { recursive: true });
        spawnDetached('chroma', ['run', '--path', CHROMA_PATH, '--host', c.host, '--port', String(c.chromaPort)],
            CHROMA_PID, path.join(LOGS, 'chroma.log'));
        log('Started ChromaDB.');
    }
}

async function stopOne(name, pidFile) {
    let pid;
    try { pid = parseInt(await fsp.readFile(pidFile, 'utf8'), 10); } catch { log(`${name} not running.`); return; }
    if (!Number.isInteger(pid) || pid < 2 || !(await pidAlive(pidFile))) {
        await fsp.rm(pidFile, { force: true });
        log(`${name} not running.`);
        return;
    }
    const kill = sig => { try { process.kill(-pid, sig); } catch { try { process.kill(pid, sig); } catch { } } };
    const alive = () => { try { process.kill(pid, 0); return true; } catch { return false; } };

    kill('SIGTERM');
    for (let i = 0; i < 50 && alive(); i++) await new Promise(r => setTimeout(r, 200)); // 10s
    if (alive()) { kill('SIGKILL'); log(`${name} force-killed.`); }
    await fsp.rm(pidFile, { force: true });
    log(`Stopped ${name} (PID ${pid}).`);
}

export async function stop() {
    await stopOne('ChromaDB', CHROMA_PID);
    await stopOne('Embedding server', SERVER_PID);
}

// reset: clears everything butler generated (databases, logs, pid files,
// config) but keeps the venvs and downloads so nothing needs reinstalling.
export async function reset() {
    if (!(await pathExists(ROOT))) { log('Butler is not initialized.'); return; }
    if (!(await confirm(`Reset Butler? This deletes the database, config, logs and pid files (venvs and models are kept).`))) {
        log('Cancelled.'); return;
    }
    await stop();

    // Remove everything in ROOT except venvs/ and models/
    const entries = await fsp.readdir(ROOT);
    for (const entry of entries) {
        if (entry === 'venvs' || entry === 'models') continue;
        await fsp.rm(path.join(ROOT, entry), { recursive: true, force: true });
    }

    // Recreate fresh defaults
    await Promise.all([
        fsp.mkdir(CHROMA_PATH, { recursive: true }),
        fsp.mkdir(LOGS, { recursive: true }),
        fsp.mkdir(PIDS, { recursive: true })
    ]);
    await fsp.writeFile(CONFIG, JSON.stringify(DEFAULTS, null, 2) + '\n', { mode: 0o600 });

    log('Butler reset. Config restored to defaults, database cleared.');
}

export async function purge() {
    if (!(await pathExists(ROOT))) { log('Butler is not initialized.'); return; }
    if (!(await confirm(`Stop services and permanently delete ${ROOT} and everything inside it?`))) {
        log('Cancelled.'); return;
    }
    await stop();
    await fsp.rm(ROOT, { recursive: true, force: true });
    log('Butler data removed.');
}
