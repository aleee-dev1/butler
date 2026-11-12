#!/usr/bin/env node
import { init, start, stop, purge, reset } from './runtime.js';
import { addCommand } from './add.js';
import { recallCommand } from './recall.js';
import { deleteCommand, flushCommand } from './maintenance.js';
import { error } from './util.js';

main();

const SUPPORTED_TOKENS = [70, 140, 280, 560, 1120];

function parseTokensFlag(args, name) {
    const arg = args.find(a => a.startsWith(`--${name}=`));
    if (!arg) return undefined;
    const v = parseInt(arg.split('=')[1], 10);
    if (!SUPPORTED_TOKENS.includes(v)) {
        throw new Error(`--${name} must be one of ${SUPPORTED_TOKENS.join(',')}.`);
    }
    return v;
}

async function main() {

    const [command, ...args] = process.argv.slice(2);

    try {
        switch (command) {
            case 'init': {
                const imageTokens = parseTokensFlag(args, 'image-tokens');
                const videoTokens = parseTokensFlag(args, 'video-tokens');
                const excluded = ['--skip-venv', '--image-tokens=', '--video-tokens='];
                const positional = args.filter(a => !excluded.some(x => a.startsWith(x)))[0] || undefined;
                await init(positional, { skipVenv: args.includes('--skip-venv'), imageTokens, videoTokens });
                break;
            }
            case 'start':
                await start({
                    imageTokens: parseTokensFlag(args, 'image-tokens'),
                    videoTokens: parseTokensFlag(args, 'video-tokens')
                });
                break;
            case 'stop':
                await stop();
                break;
            case 'reset':
                await reset();
                break;
            case 'purge':
                await purge();
                break;
            case 'add':
                await addCommand(args);
                break;
            case 'recall':
                await recallCommand(args);
                break;
            case 'delete':
                await deleteCommand(args);
                break;
            case 'flush':
                await flushCommand(args);
                break;
            case '--help':
            case '-h':
            case undefined:
                console.log(`Butler — local semantic search

Usage:
  butler init [--skip-venv] [--image-tokens=N] [--video-tokens=N] [/path/to/model-dir]
  butler start [--image-tokens=N] [--video-tokens=N] | stop | reset
  butler add . [-r] [-q] [-e="name1,name2"] [--chunk=N] [--overlap=N] [--batch=N]
  butler add <file...> [-q] [--chunk=N] [--overlap=N] [--batch=N]
  butler add "free text describing the current path"
  butler recall <query> [-i|-a|-v|-d|-t] [-n count] [--json|--toml]
  butler delete .
  butler delete <file>
  butler flush [-q]
  butler purge

File types: text, image, audio, video. Directories are never indexed on their
own — use: butler add "description…" to attach a note to the current path.

Config (in ~/.butler/config.json):
  chunk: text chunk size (default 6000)
  overlap: chunk overlap (default 300)
  batch: upsert batch size (default 500)
  embedPort: embedding server port (default 8765)
  imageTokens: vision budget per image 70/140/280/560/1120 (default 280;
    70 is about 5x faster for bulk photo indexing, small quality tax)
  videoTokens: vision budget per video frame (default 70 for CPU-friendly
    indexing; 140 is the processor's quality default)
  videoChunk: video embedding window in seconds (default 10); recall shows
    matching segments as mm:ss-mm:ss like line numbers for text
  audioChunk: audio embedding window in seconds (default 30; hard max 30)
  --image-tokens=N / --video-tokens=N on start/init set and persist config;
    start hot-restarts the server if it is running with other budgets.

init sets up ~/.butler/venvs/embeddinggemma2-venv, installs the Python stack
and downloads the model to ~/.butler/models. --skip-venv reuses the venv you
copied there. Logs go to ~/.butler/logs, pid files to ~/.butler/pids.`);
                break;
            default:
                error(`Unknown command: ${command}`);
                process.exitCode = 2;
        }
    } catch (e) {
        error(e?.message || String(e));
        process.exitCode = 1;
    }

    process.exit(process.exitCode ?? 0);

}