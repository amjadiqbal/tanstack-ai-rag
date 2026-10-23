import { PGlite } from '@electric-sql/pglite';
import { vector } from '@electric-sql/pglite-pgvector';
import { describe, expect, it } from 'vitest';
import { chunkText, cosineSimilarity, createRag, memoryVectorStore, pgvectorStore } from '../src';
import { BagOfWordsEmbedding, DOCS } from './helpers';

describe('documented edge cases', () => {
  it('default overlap is min(64, floor(maxTokens / 4))', () => {
    const text = Array.from({ length: 200 }, (_, i) => `word${i}`).join(' ');
    // maxTokens 20 -> overlap 5: with overlap 5 the default and the explicit option give the same chunks
    expect(chunkText(text, { maxTokens: 20 })).toEqual(chunkText(text, { maxTokens: 20, overlap: 5 }));
    expect(chunkText(text, { maxTokens: 400 })).toEqual(chunkText(text, { maxTokens: 400, overlap: 64 }));
  });

  it('ingest([]) does nothing and returns zeros', async () => {
    const embedding = new BagOfWordsEmbedding();
    const rag = createRag({ embedding, store: memoryVectorStore() });
    expect(await rag.ingest([])).toEqual({ documents: 0, chunks: 0 });
    expect(embedding.calls).toEqual([]);
  });

  it('a document with blank content gives no chunks and removes its old chunks', async () => {
    const store = memoryVectorStore();
    const rag = createRag({ embedding: new BagOfWordsEmbedding(), store });
    await rag.ingest({ id: 'a', content: DOCS.pricing });
    expect(await rag.ingest({ id: 'a', content: '   ' })).toEqual({ documents: 1, chunks: 0 });
    expect(store.size).toBe(0);
  });

  it('throws when the adapter returns a different number of vectors than inputs', async () => {
    const embedding = new BagOfWordsEmbedding();
    embedding.createEmbeddings = async () => ({ id: 'x', model: 'bow', embeddings: [] });
    const rag = createRag({ embedding, store: memoryVectorStore() });
    await expect(rag.ingest({ id: 'a', content: 'hello' })).rejects.toThrow(
      'embedding adapter returned 0 vectors for 1 inputs',
    );
  });

  it('memory store: a query vector of another length throws RangeError', async () => {
    const store = memoryVectorStore();
    await store.upsert([{ id: '1', documentId: 'd', content: 'c', metadata: {}, vector: [1, 0] }]);
    await expect(store.query({ vector: [1, 0, 0], topK: 1 })).rejects.toThrow(RangeError);
    expect(() => cosineSimilarity([1], [1, 2])).toThrow('vector length mismatch: 1 vs 2');
    expect(cosineSimilarity([0, 0], [1, 0])).toBe(0);
  });

  it('pgvector: a vector of the wrong size is rejected by Postgres', async () => {
    const db = new PGlite({ extensions: { vector } });
    const store = pgvectorStore({ db, dimensions: 3 });
    await store.migrate();
    await expect(
      store.upsert([{ id: '1', documentId: 'd', content: 'c', metadata: {}, vector: [1, 0] }]),
    ).rejects.toThrow(/expected 3 dimensions, not 2/);
    await db.close();
  });

  it('batchSize below 1 is treated as 1', async () => {
    const embedding = new BagOfWordsEmbedding();
    const rag = createRag({ embedding, store: memoryVectorStore(), batchSize: 0 });
    await rag.ingest({ id: 'a', content: DOCS.install, format: 'markdown' });
    expect(embedding.calls.map((c) => c.length)).toEqual([1, 1]);
  });

  it('title, url, headings and chunkIndex overwrite metadata keys of the same name', async () => {
    const store = memoryVectorStore();
    const rag = createRag({ embedding: new BagOfWordsEmbedding(), store });
    await rag.ingest({ id: 'a', title: 'T', content: 'x', metadata: { title: 'mine', chunkIndex: 99, keep: 1 } });
    const [hit] = await store.query({ vector: new Array(256).fill(1), topK: 1 });
    expect(hit?.metadata).toEqual({ title: 'T', chunkIndex: 0, keep: 1 });
  });

  it('topK 0 returns no results', async () => {
    const rag = createRag({ embedding: new BagOfWordsEmbedding(), store: memoryVectorStore() });
    await rag.ingest({ id: 'a', content: 'hello' });
    expect(await rag.retrieve('hello', { topK: 0 })).toEqual([]);
  });

  it('the tool treats a missing query argument like an empty one and rounds scores to 3 decimals', async () => {
    const rag = createRag({ embedding: new BagOfWordsEmbedding(), store: memoryVectorStore() });
    await rag.ingest({ id: 'a', content: DOCS.refunds });
    expect(await rag.tool().execute?.({} as never)).toEqual({ query: '', results: [] });
    const out = (await rag.tool().execute?.({ query: 'refund days' })) as { results: { score: number }[] };
    const score = out.results[0]?.score ?? 0;
    expect(Math.round(score * 1000) / 1000).toBe(score);
  });
});
