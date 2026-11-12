import fs from 'node:fs/promises';
import path from 'node:path';
import { readConfig, assertRunning, confirm, log, normalizePath, verbose } from './util.js';
import { getCollection, getByPath, iterateAll, dbSize } from './db.js';

export async function deleteCommand(args) {
    if (args.length !== 1) throw new Error('Usage: butler delete . | butler delete <filepath>');
    const config = await readConfig();
    await assertRunning(config);
    const target = normalizePath(args[0]);
    const collection = await getCollection(config);

    let matches = [];

    if (args[0] === '.') {
        // Delete all entries under current directory - iterate and filter
        const cwd = process.cwd();
        for await (const page of iterateAll(collection)) {
            for (let i = 0; i < (page.ids || []).length; i++) {
                const p = page.metadatas?.[i]?.path;
                if (p === cwd || p?.startsWith(cwd + path.sep)) {
                    matches.push(page.ids[i]);
                }
            }
        }
    } else {
        // Try exact match first (for files)
        const exact = await getByPath(collection, target);
        matches = exact.ids || [];

        if (!matches.length) {
            // No exact match - try prefix match for directories (iterate all)
            for await (const page of iterateAll(collection)) {
                for (let i = 0; i < (page.ids || []).length; i++) {
                    const p = page.metadatas?.[i]?.path;
                    if (p === target || p?.startsWith(target + path.sep)) {
                        matches.push(page.ids[i]);
                    }
                }
            }
        }
    }

    if (!matches.length) { log('No matching entries found.'); return; }
    if (!(await confirm(`Delete ${matches.length} DB entr${matches.length === 1 ? 'y' : 'ies'}?`))) { log('Cancelled.'); return; }
    await collection.delete({ ids: matches });
    log(`Deleted ${matches.length} entries.`);
}

export async function flushCommand(args) {
    const quiet = args.includes('-q');
    const isVerbose = !quiet;
    const config = await readConfig();
    await assertRunning(config);
    const collection = await getCollection(config);
    const remove = [];
    for await (const page of iterateAll(collection)) {
        for (let i = 0; i < (page.ids || []).length; i++) {
            const meta = page.metadatas?.[i];
            if (!meta?.path) { remove.push(page.ids[i]); continue; }
            try {
                await fs.access(meta.path);
                if (meta.type === 'dir') {
                    const st = await fs.stat(meta.path);
                    if (!st.isDirectory()) remove.push(page.ids[i]);
                } else {
                    const st = await fs.stat(meta.path);
                    if (!st.isFile()) remove.push(page.ids[i]);
                }
            } catch { remove.push(page.ids[i]); }
            if (isVerbose && remove.includes(page.ids[i])) log(`Removing ${meta.name || path.basename(meta.path)} — ${meta.path}`);
        }
    }
    if (remove.length) await collection.delete({ ids: remove });
    log(`Removed ${remove.length} stale entries.`);
}
