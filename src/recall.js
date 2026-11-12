import { readConfig, assertRunning, error } from './util.js';
import { getCollection } from './db.js';
import { embedText } from './embed.js';

export async function recallCommand(args) {
    const flags = new Set(args.filter(a => a.startsWith('-')));
    const nIndex = args.indexOf('-n');
    // exclude only the value that belongs to -n, so queries containing
    // numbers (e.g. "bug 404") keep them in the query text
    const nValueIndex = nIndex >= 0 ? nIndex + 1 : -1;
    const query = args
        .filter((a, i) => !a.startsWith('-') && i !== nValueIndex)
        .join(' ')
        .trim();
    const n = nIndex >= 0 ? Number(args[nValueIndex]) : 10;
    if (!query) throw new Error('Usage: butler recall "query" [-i|-d|-t] [-n count] [--json|--toml]');
    if (!Number.isInteger(n) || n < 1 || n > 1000) throw new Error('-n must be an integer between 1 and 1000.');
    const types = [];
    if (flags.has('-i')) types.push('image');
    if (flags.has('-a')) types.push('audio');
    if (flags.has('-v')) types.push('video');
    if (flags.has('-d')) types.push('dir');
    if (flags.has('-t')) types.push('text');
    if (types.length > 1) throw new Error('Choose only one type filter: -i, -a, -v, -d, or -t.');
    const config = await readConfig();
    await assertRunning(config);
    const collection = await getCollection(config);
    const vector = await embedText(query, config, 'query');
    const result = await collection.query({
        queryEmbeddings: [vector],
        nResults: n,
        ...(types.length ? { where: { type: types[0] } } : {}),
        include: ['metadatas', 'documents', 'distances']
    });
    const rows = (result.ids?.[0] || []).map((id, i) => {
        const meta = result.metadatas?.[0]?.[i] ?? {};
        return {
            score: Math.max(0, Math.min(1, 1 - (result.distances?.[0]?.[i] ?? 1))),
            type: meta.type ?? 'text',
            path: meta.path ?? '',
            name: meta.name ?? '',
            document: result.documents?.[0]?.[i] ?? '',
            lineStart: meta.lineStart,
            lineEnd: meta.lineEnd,
            tStart: meta.tStart,
            tEnd: meta.tEnd
        };
    });
    if (flags.has('--json')) {
        console.log(JSON.stringify(rows, null, 2));
    } else if (flags.has('--toml')) {
        console.log(rows.map((r, i) => `[result_${i + 1}]\nscore = ${r.score.toFixed(5)}\ntype = ${JSON.stringify(r.type)}\npath = ${JSON.stringify(r.path)}\nname = ${JSON.stringify(r.name)}\ndocument = ${JSON.stringify(r.document)}${r.lineStart !== undefined ? `\nlineStart = ${r.lineStart}\nlineEnd = ${r.lineEnd}` : ''}${r.tStart !== undefined && r.tStart !== null ? `\ntStart = ${r.tStart}\ntEnd = ${r.tEnd}` : ''}`).join('\n\n'));
    } else {
        // mm:ss helper for audio/video segment hits (text keeps L ranges).
        const mmss = s => {
            const t = Math.floor(Number(s) || 0);
            return `${String(Math.floor(t / 60)).padStart(2, '0')}:${String(t % 60).padStart(2, '0')}`;
        };
        const timeInfo = r => (r.tStart !== undefined && r.tStart !== null)
            ? `  ${mmss(r.tStart)}-${mmss(r.tEnd ?? r.tStart)}`
            : (r.lineStart !== undefined ? `  L${r.lineStart}-${r.lineEnd}` : '');
        for (const r of rows) {
            console.log(`${r.score.toFixed(4)}  ${r.type.padEnd(5)}  ${timeInfo(r).padEnd(12)}  ${r.path}`);
        }
        if (!rows.length) console.log('No results.');
    }
}
