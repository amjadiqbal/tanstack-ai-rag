import type { VectorMatch, VectorRecord, VectorStore } from './store';

/** `pg` Pool or Client, PGlite, and anything else with `query(text, params)` that resolves to `{ rows }`. */
export interface QueryClient {
  query(text: string, params?: unknown[]): Promise<{ rows: unknown[] }>;
}

/** postgres.js: the `sql` function, used through `sql.unsafe(text, params)`. */
export interface UnsafeClient {
  unsafe(text: string, params?: never[]): PromiseLike<unknown>;
}

/** A Drizzle database; its underlying driver client is read from `db.$client`. */
export interface DrizzleLike {
  $client: QueryClient | UnsafeClient;
}

export type PgDatabase = QueryClient | UnsafeClient | DrizzleLike;

export interface PgvectorStoreOptions {
  /** A `pg` Pool or Client, a postgres.js `sql`, a PGlite instance, or a Drizzle database built on one of them. */
  db: PgDatabase;
  /** Table name, optionally schema-qualified (`rag.chunks`). Default `rag_chunks`. */
  table?: string;
  /** Embedding size. Needed only by `migrate()`, to create the `vector(n)` column. */
  dimensions?: number;
  /**
   * Vector index `migrate()` creates. `hnsw` (default) needs pgvector 0.5+ and at most 2,000 dimensions;
   * pick `none` for larger embeddings or tiny tables (exact search).
   */
  index?: 'hnsw' | 'none';
}

export interface PgvectorStore extends VectorStore {
  /** Deletes the documents' chunks and inserts `records` in one transaction (used by `ingest()`). */
  replaceDocuments(documentIds: string[], records: VectorRecord[]): Promise<void>;
  /** Creates the `vector` extension, the table and its indexes if they don't exist. Safe to run on every start. */
  migrate(): Promise<void>;
  /** The quoted table name used in SQL. */
  readonly table: string;
}

type Exec = (text: string, params?: unknown[]) => Promise<Record<string, unknown>[]>;
/** Runs `fn` inside BEGIN ... COMMIT on one connection, rolling back if it throws. */
type Transact = (fn: (exec: Exec) => Promise<void>) => Promise<void>;

// Driver shapes used only for transactions; kept loose because each driver types them differently.
type AnyClient = {
  query?: (text: string, params?: unknown[]) => Promise<{ rows: unknown[] }>;
  unsafe?: (text: string, params?: never[]) => PromiseLike<unknown>;
  begin?: (fn: (tx: AnyClient) => Promise<unknown>) => PromiseLike<unknown>;
  transaction?: (fn: (tx: AnyClient) => Promise<unknown>) => Promise<unknown>;
  connect?: () => Promise<AnyClient & { release(): void }>;
  idleCount?: number;
};

function clientOf(db: PgDatabase): AnyClient {
  return ('$client' in db && db.$client ? db.$client : db) as AnyClient;
}

function execOf(client: AnyClient): Exec | null {
  const { unsafe, query } = client;
  if (typeof unsafe === 'function') {
    return async (text, params) => Array.from((await unsafe.call(client, text, params as never[])) as Iterable<never>);
  }
  if (typeof query === 'function') {
    return async (text, params) => (await query.call(client, text, params)).rows as Record<string, unknown>[];
  }
  return null;
}

function toExec(db: PgDatabase): Exec {
  const exec = execOf(clientOf(db));
  if (!exec) {
    throw new TypeError('pgvectorStore: db must be a pg Pool/Client, postgres.js sql, PGlite or a Drizzle database');
  }
  return exec;
}

function toTransact(db: PgDatabase): Transact {
  const client = clientOf(db);
  // postgres.js: sql.begin() pins one connection.
  if (typeof client.begin === 'function') {
    return async (fn) => {
      await (client.begin as NonNullable<AnyClient['begin']>)((tx) => fn(execOf(tx) as Exec));
    };
  }
  // PGlite: db.transaction().
  if (typeof client.transaction === 'function') {
    return async (fn) => {
      await (client.transaction as NonNullable<AnyClient['transaction']>)((tx) => fn(execOf(tx) as Exec));
    };
  }
  // pg Pool: check out one client for the whole transaction.
  if (typeof client.connect === 'function' && typeof client.idleCount === 'number') {
    return async (fn) => {
      const conn = await (client.connect as NonNullable<AnyClient['connect']>)();
      const exec = execOf(conn) as Exec;
      try {
        await exec('BEGIN');
        await fn(exec);
        await exec('COMMIT');
      } catch (error) {
        await exec('ROLLBACK').catch(() => {});
        throw error;
      } finally {
        conn.release();
      }
    };
  }
  // A single connection (pg Client): BEGIN/COMMIT on it directly.
  const exec = toExec(db);
  return async (fn) => {
    await exec('BEGIN');
    try {
      await fn(exec);
      await exec('COMMIT');
    } catch (error) {
      await exec('ROLLBACK').catch(() => {});
      throw error;
    }
  };
}

const IDENT = /^[A-Za-z_][A-Za-z0-9_]{0,62}$/;

function quoteTable(name: string): string {
  const parts = name.split('.');
  if (parts.length > 2 || !parts.every((p) => IDENT.test(p))) {
    throw new TypeError(`pgvectorStore: invalid table name "${name}" (letters, digits and _ only, optional schema.)`);
  }
  return parts.map((p) => `"${p}"`).join('.');
}

const toVector = (v: readonly number[]) => {
  for (const x of v) if (!Number.isFinite(x)) throw new TypeError('pgvectorStore: vectors must contain finite numbers');
  return `[${v.join(',')}]`;
};

/** Rows per INSERT; 5 parameters each keeps us far below Postgres' 65,535-parameter limit. */
const BATCH = 500;

/**
 * A Postgres + pgvector store using cosine distance. Works through `pg`, postgres.js, PGlite (with its vector
 * extension) or Drizzle on any of them; this package imports none of them.
 */
export function pgvectorStore(options: PgvectorStoreOptions): PgvectorStore {
  const exec = toExec(options.db);
  const transact = toTransact(options.db);
  const table = quoteTable(options.table ?? 'rag_chunks');
  const bare = (options.table ?? 'rag_chunks').replace('.', '_');

  async function upsertWith(run: Exec, records: VectorRecord[]) {
    for (let i = 0; i < records.length; i += BATCH) {
      const batch = records.slice(i, i + BATCH);
      const params: unknown[] = [];
      const rows = batch.map((r, j) => {
        params.push(r.id, r.documentId, r.content, JSON.stringify(r.metadata ?? {}), toVector(r.vector));
        const b = j * 5;
        return `($${b + 1}, $${b + 2}, $${b + 3}, $${b + 4}::text::jsonb, $${b + 5}::vector)`;
      });
      await run(
        `INSERT INTO ${table} (id, document_id, content, metadata, embedding) VALUES ${rows.join(', ')}
         ON CONFLICT (id) DO UPDATE SET document_id = EXCLUDED.document_id, content = EXCLUDED.content,
           metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding`,
        params,
      );
    }
  }

  async function deleteWith(run: Exec, documentIds: string[]) {
    if (!documentIds.length) return;
    await run(`DELETE FROM ${table} WHERE document_id = ANY($1::text[])`, [documentIds]);
  }

  return {
    table,
    async migrate() {
      const dims = options.dimensions;
      if (!Number.isInteger(dims) || (dims as number) < 1) {
        throw new TypeError('pgvectorStore: set `dimensions` to your embedding size before calling migrate()');
      }
      await exec('CREATE EXTENSION IF NOT EXISTS vector');
      await exec(
        `CREATE TABLE IF NOT EXISTS ${table} (
          id text PRIMARY KEY,
          document_id text NOT NULL,
          content text NOT NULL,
          metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
          embedding vector(${dims}) NOT NULL
        )`,
      );
      await exec(`CREATE INDEX IF NOT EXISTS "${bare}_document_id_idx" ON ${table} (document_id)`);
      await exec(`CREATE INDEX IF NOT EXISTS "${bare}_metadata_idx" ON ${table} USING gin (metadata jsonb_path_ops)`);
      if ((options.index ?? 'hnsw') === 'hnsw') {
        await exec(
          `CREATE INDEX IF NOT EXISTS "${bare}_embedding_idx" ON ${table} USING hnsw (embedding vector_cosine_ops)`,
        );
      }
    },
    upsert: (records) => upsertWith(exec, records),
    async query({ vector, topK, filter }) {
      const params: unknown[] = [toVector(vector), Math.max(0, Math.floor(topK))];
      let where = '';
      if (filter && Object.keys(filter).length) {
        params.push(JSON.stringify(filter));
        where = 'WHERE metadata @> $3::text::jsonb';
      }
      const rows = await exec(
        `SELECT id, document_id, content, metadata, 1 - (embedding <=> $1::vector) AS score
         FROM ${table} ${where} ORDER BY embedding <=> $1::vector LIMIT $2`,
        params,
      );
      return rows.map(
        (row): VectorMatch => ({
          id: String(row.id),
          documentId: String(row.document_id),
          content: String(row.content),
          metadata: (typeof row.metadata === 'string' ? JSON.parse(row.metadata) : row.metadata) ?? {},
          score: Number(row.score),
        }),
      );
    },
    deleteDocuments: (documentIds) => deleteWith(exec, documentIds),
    replaceDocuments: (documentIds, records) =>
      transact(async (tx) => {
        await deleteWith(tx, documentIds);
        await upsertWith(tx, records);
      }),
  };
}
