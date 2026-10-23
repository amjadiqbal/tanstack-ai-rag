import { PGlite } from '@electric-sql/pglite';
import { vector } from '@electric-sql/pglite-pgvector';
import { drizzle } from 'drizzle-orm/node-postgres';
import pg from 'pg';
import postgres from 'postgres';
import { afterAll, describe, expect, it } from 'vitest';
import { cosineSimilarity, memoryVectorStore, pgvectorStore, type VectorRecord, type VectorStore } from '../src';

/** Set by scripts/test-pgvector.sh; the Docker suites are skipped without it. */
const PG_URL = process.env.PGVECTOR_URL;

const rec = (id: string, documentId: string, vec: number[], metadata: Record<string, unknown> = {}): VectorRecord => ({
  id,
  documentId,
  content: `content of ${id}`,
  metadata,
  vector: vec,
});

/** The same behaviour checks for every store. */
function conformance(name: string, make: () => Promise<VectorStore & { cleanup?: () => Promise<void> }>) {
  describe(name, () => {
    let store: VectorStore & { cleanup?: () => Promise<void> };
    afterAll(async () => store?.cleanup?.());

    it('returns the nearest records by cosine similarity, best first', async () => {
      store = await make();
      await store.upsert([
        rec('a#0', 'a', [1, 0, 0], { lang: 'en', title: 'A' }),
        rec('a#1', 'a', [0.9, 0.1, 0], { lang: 'en' }),
        rec('b#0', 'b', [0, 1, 0], { lang: 'de' }),
        rec('c#0', 'c', [0, 0, 1], { lang: 'en', nested: { deep: [1, 2] } }),
      ]);
      const hits = await store.query({ vector: [1, 0.05, 0], topK: 2 });
      expect(hits.map((h) => h.id)).toEqual(['a#0', 'a#1']);
      expect(hits[0]?.score).toBeCloseTo(cosineSimilarity([1, 0.05, 0], [1, 0, 0]), 5);
      expect(hits[0]).toMatchObject({
        documentId: 'a',
        content: 'content of a#0',
        metadata: { lang: 'en', title: 'A' },
      });
      expect(hits[0]).not.toHaveProperty('vector');
    });

    it('filters on metadata equality', async () => {
      const hits = await store.query({ vector: [0, 1, 0], topK: 10, filter: { lang: 'en' } });
      expect(hits.map((h) => h.documentId).sort()).toEqual(['a', 'a', 'c']);
      const none = await store.query({ vector: [0, 1, 0], topK: 10, filter: { lang: 'fr' } });
      expect(none).toEqual([]);
    });

    it('round-trips nested JSON metadata', async () => {
      const [hit] = await store.query({ vector: [0, 0, 1], topK: 1 });
      expect(hit?.metadata).toEqual({ lang: 'en', nested: { deep: [1, 2] } });
    });

    it('replaces a record on upsert with the same id', async () => {
      await store.upsert([{ ...rec('b#0', 'b', [0, 0, 1], { lang: 'de' }), content: 'updated' }]);
      const [hit] = await store.query({ vector: [0, 0, 1], topK: 1, filter: { lang: 'de' } });
      expect(hit).toMatchObject({ id: 'b#0', content: 'updated' });
      expect(hit?.score).toBeCloseTo(1, 5);
    });

    it('deletes all chunks of a document', async () => {
      await store.deleteDocuments(['a']);
      await store.deleteDocuments([]);
      const hits = await store.query({ vector: [1, 0, 0], topK: 10 });
      expect(hits.map((h) => h.documentId).sort()).toEqual(['b', 'c']);
    });

    it('stores large batches', async () => {
      const many = Array.from({ length: 1200 }, (_, i) => rec(`m#${i}`, 'm', [Math.cos(i), Math.sin(i), 0.5]));
      await store.upsert(many);
      const hits = await store.query({ vector: [Math.cos(7), Math.sin(7), 0.5], topK: 1 });
      expect(hits[0]?.id).toBe('m#7');
    });
  });
}

conformance('memoryVectorStore', async () => memoryVectorStore());

conformance('pgvectorStore on PGlite', async () => {
  const db = new PGlite({ extensions: { vector } });
  const store = pgvectorStore({ db, dimensions: 3 });
  await store.migrate();
  await store.migrate(); // idempotent
  return Object.assign(store, { cleanup: () => db.close() });
});

describe.runIf(PG_URL)('pgvector in Docker', () => {
  conformance('pgvectorStore on pg Pool', async () => {
    const pool = new pg.Pool({ connectionString: PG_URL });
    await pool.query('DROP TABLE IF EXISTS rag_pg');
    const store = pgvectorStore({ db: pool, table: 'rag_pg', dimensions: 3 });
    await store.migrate();
    return Object.assign(store, { cleanup: () => pool.end() });
  });

  conformance('pgvectorStore on postgres.js', async () => {
    const sql = postgres(PG_URL as string, { onnotice: () => {} });
    await sql`DROP TABLE IF EXISTS rag_pgjs`;
    const store = pgvectorStore({ db: sql, table: 'rag_pgjs', dimensions: 3 });
    await store.migrate();
    return Object.assign(store, { cleanup: () => sql.end() });
  });

  conformance('pgvectorStore on Drizzle (node-postgres)', async () => {
    const pool = new pg.Pool({ connectionString: PG_URL });
    await pool.query('CREATE SCHEMA IF NOT EXISTS kb; DROP TABLE IF EXISTS kb.chunks');
    const db = drizzle(pool);
    const store = pgvectorStore({ db, table: 'kb.chunks', dimensions: 3, index: 'none' });
    await store.migrate();
    return Object.assign(store, { cleanup: () => pool.end() });
  });

  it('migrate() creates the HNSW cosine index', async () => {
    const pool = new pg.Pool({ connectionString: PG_URL });
    const { rows } = await pool.query(
      "SELECT indexdef FROM pg_indexes WHERE tablename = 'rag_pg' AND indexname = 'rag_pg_embedding_idx'",
    );
    await pool.end();
    expect(rows[0]?.indexdef).toMatch(/USING hnsw \(embedding vector_cosine_ops\)/);
  });
});

describe('pgvectorStore input checks', () => {
  it('rejects unsafe table names and unknown clients', () => {
    expect(() => pgvectorStore({ db: { query: async () => ({ rows: [] }) }, table: 'x; drop table y' })).toThrow(
      /invalid table name/,
    );
    expect(() => pgvectorStore({ db: {} as never })).toThrow(/pg Pool/);
  });

  it('requires dimensions for migrate()', async () => {
    const store = pgvectorStore({ db: { query: async () => ({ rows: [] }) } });
    await expect(store.migrate()).rejects.toThrow(/dimensions/);
  });

  it('rejects non-finite vector values', async () => {
    const store = pgvectorStore({ db: { query: async () => ({ rows: [] }) } });
    await expect(store.query({ vector: [Number.NaN], topK: 1 })).rejects.toThrow(/finite/);
  });
});
