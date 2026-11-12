import { ChromaClient } from 'chromadb';
import { chromaUrl, COLLECTION } from './util.js';

const PAGE_SIZE = 1000;

export async function getCollection(config) {
    const url = new URL(chromaUrl(config));
    const client = new ChromaClient({ host: url.hostname, port: Number(url.port), ssl: false });

    const _warn = console.warn;
    console.warn = (...args) => {
        if (typeof args[0] === 'string' && args[0].includes('DefaultEmbeddingFunction')) return;
        _warn.apply(console, args);
    };
    try {
        return await client.getOrCreateCollection({
            name: COLLECTION,
            embeddingFunction: null,
            metadata: { 'hnsw:space': 'cosine' }
        });
    } finally {
        console.warn = _warn;
    }
}

/**
 * Iterate over ALL records in batches (pagination).
 * Yields { ids, metadatas, documents } for each page.
 */
export async function* iterateAll(collection, options = {}) {
    const { include = ['metadatas', 'documents'], pageSize = PAGE_SIZE } = options;
    let offset = 0;
    while (true) {
        const page = await collection.get({ include, limit: pageSize, offset });
        const ids = page.ids || [];
        if (!ids.length) break;
        yield page;
        offset += ids.length;
    }
}

/**
 * Get records for exact path match (server-side filter).
 */
export async function getByPath(collection, path, options = {}) {
    const { include = ['metadatas', 'documents'] } = options;
    return collection.get({
        where: { path },
        include
    });
}

export async function dbSize(collection) {
    return collection.count();
}