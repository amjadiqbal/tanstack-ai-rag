/** Scalar metadata values a filter can match on. */
export type MetadataValue = string | number | boolean | null;

/** Exact-match filter on chunk metadata: every key must equal its value. */
export type MetadataFilter = Record<string, MetadataValue>;

/** One stored chunk. */
export interface VectorRecord {
  /** Unique chunk id. `createRag` uses `<documentId>#<chunkIndex>`. */
  id: string;
  /** The source document the chunk came from. Deleting a document deletes all its chunks. */
  documentId: string;
  /** The chunk text returned to the model. */
  content: string;
  /** JSON-serialisable metadata. */
  metadata: Record<string, unknown>;
  vector: number[];
}

export interface VectorQuery {
  vector: number[];
  topK: number;
  filter?: MetadataFilter;
}

/** A stored chunk plus its cosine similarity to the query (1 = same direction, 0 = unrelated). */
export interface VectorMatch extends Omit<VectorRecord, 'vector'> {
  score: number;
}

/** The small contract every vector store implements. Write your own for any database in about 50 lines. */
export interface VectorStore {
  /** Inserts the records, replacing any with the same id. */
  upsert(records: VectorRecord[]): Promise<void>;
  /** Returns up to `topK` records by cosine similarity, best first. */
  query(query: VectorQuery): Promise<VectorMatch[]>;
  /** Deletes every chunk of the given documents. */
  deleteDocuments(documentIds: string[]): Promise<void>;
}

export function cosineSimilarity(a: readonly number[], b: readonly number[]): number {
  if (a.length !== b.length) throw new RangeError(`vector length mismatch: ${a.length} vs ${b.length}`);
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i] as number;
    const y = b[i] as number;
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  return na === 0 || nb === 0 ? 0 : dot / Math.sqrt(na * nb);
}

export function matchesFilter(metadata: Record<string, unknown>, filter?: MetadataFilter): boolean {
  if (!filter) return true;
  return Object.entries(filter).every(([key, value]) => metadata[key] === value);
}

export interface MemoryVectorStore extends VectorStore {
  /** Number of stored chunks. */
  readonly size: number;
  /** Removes everything. */
  clear(): void;
}

/**
 * An in-process vector store with exact cosine search. Good for tests, demos, serverless functions that rebuild
 * their index on start, and corpora up to a few thousand chunks. Nothing is persisted.
 */
export function memoryVectorStore(): MemoryVectorStore {
  const records = new Map<string, VectorRecord>();
  return {
    get size() {
      return records.size;
    },
    clear() {
      records.clear();
    },
    async upsert(items) {
      for (const r of items) records.set(r.id, { ...r, vector: [...r.vector], metadata: structuredClone(r.metadata) });
    },
    async query({ vector, topK, filter }) {
      const scored: VectorMatch[] = [];
      for (const { vector: v, ...rest } of records.values()) {
        if (!matchesFilter(rest.metadata, filter)) continue;
        scored.push({ ...rest, metadata: structuredClone(rest.metadata), score: cosineSimilarity(vector, v) });
      }
      return scored.sort((a, b) => b.score - a.score).slice(0, topK);
    },
    async deleteDocuments(documentIds) {
      const ids = new Set(documentIds);
      for (const [id, r] of records) if (ids.has(r.documentId)) records.delete(id);
    },
  };
}
