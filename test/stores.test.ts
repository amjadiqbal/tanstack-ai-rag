import { PGlite } from '@electric-sql/pglite';
import { vector } from '@electric-sql/pglite-pgvector';
import { drizzle } from 'drizzle-orm/node-postgres';
import pg from 'pg';
import postgres from 'postgres';
import { afterAll, describe, expect, it } from 'vitest';
import {
  cosineSimilarity,
  type MetadataFilter,
  memoryVectorStore,
  pgvectorStore,
  type VectorRecord,
  type VectorStore,
} from '../src';

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

    it('a null filter matches only keys stored as null, not missing keys', async () => {
      await store.upsert([rec('n#0', 'n', [0, 1, 1], { gone: null })]);
      expect((await store.query({ vector: [0, 1, 1], topK: 50, filter: { gone: null } })).map((h) => h.id)).toEqual([
        'n#0',
      ]);
      expect(await store.query({ vector: [0, 1, 1], topK: 50, filter: { missing: null } })).toEqual([]);
    });

    it('filter values must have the same type: 1 does not match "1"', async () => {
      await store.upsert([rec('v#0', 'v', [0, 1, 1], { n: 1 })]);
      expect((await store.query({ vector: [0, 1, 1], topK: 50, filter: { n: 1 } })).map((h) => h.id)).toEqual(['v#0']);
      expect(await store.query({ vector: [0, 1, 1], topK: 50, filter: { n: '1' } })).toEqual([]);
    });

    it('replaceDocuments swaps a document in one step', async () => {
      await store.replaceDocuments?.(['m'], [rec('m#new', 'm', [1, 1, 1])]);
      const hits = await store.query({ vector: [1, 1, 1], topK: 5000 });
      expect(hits.filter((h) => h.documentId === 'm').map((h) => h.id)).toEqual(['m#new']);
    });

    it('replaceDocuments keeps the old chunks when the new ones fail, even in a later batch', async () => {
      const next = Array.from({ length: 600 }, (_, i) => rec(`m#r${i}`, 'm', [1, 0, 1]));
      next[599] = badRecord();
      await expect(store.replaceDocuments?.(['m'], next)).rejects.toThrow();
      const hits = await store.query({ vector: [1, 1, 1], topK: 5000 });
      expect(hits.filter((h) => h.documentId === 'm').map((h) => h.id)).toEqual(['m#new']);
    });
  });
}

/** A record each store rejects: wrong vector size for Postgres, un-cloneable metadata for memory. */
let badRecord: () => VectorRecord = () => rec('bad', 'm', [1, 0]);

conformance('memoryVectorStore', async () => {
  badRecord = () => rec('bad', 'm', [1, 0, 1], { fn: () => 1 });
  return memoryVectorStore();
});

conformance('pgvectorStore on PGlite', async () => {
  badRecord = () => rec('bad', 'm', [1, 0]);
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

describe('filters on arrays and objects (where the stores differ)', () => {
  const tagged = rec('t#0', 't', [1, 0, 0], { tags: ['a', 'b'], owner: { id: 1, name: 'x' } });
  const arrayFilter = { tags: ['a'] } as unknown as MetadataFilter;
  const objectFilter = { owner: { id: 1 } } as unknown as MetadataFilter;

  it('memory: arrays and objects never match (=== comparison)', async () => {
    const store = memoryVectorStore();
    await store.upsert([tagged]);
    expect(await store.query({ vector: [1, 0, 0], topK: 5, filter: arrayFilter })).toEqual([]);
    expect(await store.query({ vector: [1, 0, 0], topK: 5, filter: objectFilter })).toEqual([]);
  });

  it('pgvector: jsonb containment, so a subset of an array or object matches', async () => {
    const db = new PGlite({ extensions: { vector } });
    const store = pgvectorStore({ db, dimensions: 3 });
    await store.migrate();
    await store.upsert([tagged]);
    expect((await store.query({ vector: [1, 0, 0], topK: 5, filter: arrayFilter })).map((h) => h.id)).toEqual(['t#0']);
    expect((await store.query({ vector: [1, 0, 0], topK: 5, filter: objectFilter })).map((h) => h.id)).toEqual(['t#0']);
    await db.close();
  });

  it('pgvector: filter keys and values are bound parameters, not SQL', async () => {
    const db = new PGlite({ extensions: { vector } });
    const store = pgvectorStore({ db, dimensions: 3 });
    await store.migrate();
    await store.upsert([tagged]);
    const evil = { "x' OR 1=1; DROP TABLE rag_chunks; --": "' OR '1'='1" };
    expect(await store.query({ vector: [1, 0, 0], topK: 5, filter: evil })).toEqual([]);
    expect(await store.query({ vector: [1, 0, 0], topK: 5 })).toHaveLength(1);
    await db.close();
  });
});
