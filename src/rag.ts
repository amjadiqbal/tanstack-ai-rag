import {
  type AnyEmbeddingAdapter,
  type AnyRerankAdapter,
  embed,
  type JSONSchema,
  rerank,
  toolDefinition,
} from '@tanstack/ai';
import { type Chunk, type ChunkOptions, chunkMarkdown, chunkText } from './chunk';
import type { MetadataFilter, VectorMatch, VectorRecord, VectorStore } from './store';

/** A document to index. */
export interface RagDocument {
  /** Stable id (a path, URL or database key). Ingesting the same id again replaces its chunks. */
  id: string;
  content: string;
  /** How to chunk `content`. Default `text`. */
  format?: 'text' | 'markdown';
  /** Shown as the citation's source; falls back to `url`, then `id`. */
  title?: string;
  url?: string;
  /** Extra JSON-serialisable metadata stored on every chunk; scalar keys can be used in `filter`. */
  metadata?: Record<string, unknown>;
}

export interface RagRerankOptions {
  /** Any TanStack AI rerank adapter, e.g. `cohereRerank('rerank-v3.5')`. */
  adapter: AnyRerankAdapter;
  /** How many vector matches to send to the reranker. Default `topK * 4`. */
  candidates?: number;
  modelOptions?: Record<string, unknown>;
}

export interface RagOptions {
  /** Any TanStack AI embedding adapter, e.g. `openaiEmbedding('text-embedding-3-small')`. */
  embedding: AnyEmbeddingAdapter;
  store: VectorStore;
  /** Chunk size, overlap and token counter. */
  chunking?: ChunkOptions;
  /** Requested embedding size, for models that support it. Use the same value for ingest and queries. */
  dimensions?: number;
  /** `modelOptions` for embedding documents (e.g. Cohere `{ inputType: 'search_document' }`). */
  documentModelOptions?: Record<string, unknown>;
  /** `modelOptions` for embedding queries (e.g. Cohere `{ inputType: 'search_query' }`). */
  queryModelOptions?: Record<string, unknown>;
  /** Chunks per `embed()` call. Default 64. */
  batchSize?: number;
  /** Results per query. Default 5. */
  topK?: number;
  /** Drop vector matches below this cosine similarity. Default none. */
  minScore?: number;
  /** Optional second-stage rerank with TanStack AI's `rerank()`. */
  rerank?: RagRerankOptions;
}

export interface RetrieveOptions {
  topK?: number;
  minScore?: number;
  filter?: MetadataFilter;
  /** Forwarded to the rerank request. */
  abortSignal?: AbortSignal;
}

/** A retrieved chunk, numbered for citation. */
export interface RagResult extends VectorMatch {
  /** 1-based number the model cites as `[n]`. */
  citation: number;
  /** Display name of the source: the document's title, url or id. */
  source: string;
  url?: string;
  headings?: string[];
  /** Relevance from the reranker, when one is configured. */
  rerankScore?: number;
}

/** What the retrieval tool returns to the model (and what your UI receives as the tool call's output). */
export interface RagToolOutput {
  query: string;
  results: Array<{
    citation: number;
    source: string;
    url?: string;
    headings?: string[];
    content: string;
    score: number;
  }>;
}

export interface RagToolOptions {
  /** Tool name the model calls. Default `search_knowledge_base`. */
  name?: string;
  /** Tells the model when to use the tool. Say what the knowledge base holds. */
  description?: string;
  topK?: number;
  minScore?: number;
  /** Fixed filter applied to every search the model makes (e.g. a tenant or product id). */
  filter?: MetadataFilter;
}

export interface IngestResult {
  documents: number;
  chunks: number;
}

/** Add this to your system prompt so the model cites retrieved sources the way the tool numbers them. */
export const citationInstructions =
  'Answer from the knowledge base search results. After each claim, cite its source by number, like [1] or [2]. ' +
  'If the results do not contain the answer, say you could not find it instead of guessing.';

const inputSchema: JSONSchema = {
  type: 'object',
  properties: {
    query: { type: 'string', description: 'A focused search query, rephrased from the question if needed.' },
  },
  required: ['query'],
};

function chunkDocument(doc: RagDocument, options?: ChunkOptions): Chunk[] {
  return doc.format === 'markdown' ? chunkMarkdown(doc.content, options) : chunkText(doc.content, options);
}

/** Text that gets embedded for a chunk: its title and heading path give short chunks the context they lack. */
function embeddingText(doc: RagDocument, chunk: Chunk): string {
  const path = [doc.title, ...(chunk.headings ?? [])].filter(Boolean).join(' > ');
  return path ? `${path}\n\n${chunk.content}` : chunk.content;
}

/**
 * Creates a RAG pipeline: `ingest()` chunks, embeds and stores documents, `retrieve()` finds the best chunks for a
 * question, and `tool()` hands retrieval to `chat()` as a server tool whose results carry citation numbers.
 */
export function createRag(options: RagOptions) {
  const { store } = options;
  const batchSize = Math.max(1, options.batchSize ?? 64);

  async function embedTexts(texts: string[], modelOptions?: Record<string, unknown>): Promise<number[][]> {
    const vectors: number[][] = [];
    for (let i = 0; i < texts.length; i += batchSize) {
      const result = await embed({
        adapter: options.embedding,
        input: texts.slice(i, i + batchSize),
        ...(options.dimensions ? { dimensions: options.dimensions } : {}),
        ...(modelOptions ? { modelOptions } : {}),
      } as Parameters<typeof embed>[0]);
      for (const e of result.embeddings) vectors.push(e.vector);
    }
    if (vectors.length !== texts.length) {
      throw new Error(`embedding adapter returned ${vectors.length} vectors for ${texts.length} inputs`);
    }
    return vectors;
  }

  async function ingest(documents: RagDocument | RagDocument[]): Promise<IngestResult> {
    const docs = Array.isArray(documents) ? documents : [documents];
    const seen = new Set<string>();
    for (const d of docs) {
      if (!d.id) throw new TypeError('every document needs an id');
      if (seen.has(d.id)) throw new TypeError(`duplicate document id "${d.id}"`);
      seen.add(d.id);
    }
    const pending: { doc: RagDocument; chunk: Chunk }[] = [];
    for (const doc of docs) for (const chunk of chunkDocument(doc, options.chunking)) pending.push({ doc, chunk });

    const vectors = await embedTexts(
      pending.map((p) => embeddingText(p.doc, p.chunk)),
      options.documentModelOptions,
    );
    const records: VectorRecord[] = pending.map(({ doc, chunk }, i) => ({
      id: `${doc.id}#${chunk.index}`,
      documentId: doc.id,
      content: chunk.content,
      vector: vectors[i] as number[],
      metadata: {
        ...doc.metadata,
        ...(doc.title ? { title: doc.title } : {}),
        ...(doc.url ? { url: doc.url } : {}),
        ...(chunk.headings ? { headings: chunk.headings } : {}),
        chunkIndex: chunk.index,
      },
    }));
    // Embed first, then replace: a failed embedding call leaves the old chunks in place.
    await store.deleteDocuments(docs.map((d) => d.id));
    await store.upsert(records);
    return { documents: docs.length, chunks: records.length };
  }

  async function retrieve(query: string, opts: RetrieveOptions = {}): Promise<RagResult[]> {
    const topK = opts.topK ?? options.topK ?? 5;
    const minScore = opts.minScore ?? options.minScore;
    const [vector] = await embedTexts([query], options.queryModelOptions);
    const candidates = options.rerank ? Math.max(topK, options.rerank.candidates ?? topK * 4) : topK;
    let matches: (VectorMatch & { rerankScore?: number })[] = await store.query({
      vector: vector as number[],
      topK: candidates,
      ...(opts.filter ? { filter: opts.filter } : {}),
    });
    if (minScore !== undefined) matches = matches.filter((m) => m.score >= minScore);

    if (options.rerank && matches.length > 0) {
      const ranked = await rerank({
        adapter: options.rerank.adapter,
        query,
        documents: matches.map((m) => m.content),
        topN: topK,
        ...(options.rerank.modelOptions ? { modelOptions: options.rerank.modelOptions } : {}),
        ...(opts.abortSignal ? { abortSignal: opts.abortSignal } : {}),
      });
      matches = ranked.ranking.map((r) => ({ ...(matches[r.index] as VectorMatch), rerankScore: r.score }));
    }

    return matches.slice(0, topK).map((m, i) => {
      const meta = m.metadata;
      const url = typeof meta.url === 'string' ? meta.url : undefined;
      const title = typeof meta.title === 'string' ? meta.title : undefined;
      const headings = Array.isArray(meta.headings) ? (meta.headings as string[]) : undefined;
      return {
        ...m,
        citation: i + 1,
        source: title ?? url ?? m.documentId,
        ...(url ? { url } : {}),
        ...(headings ? { headings } : {}),
      };
    });
  }

  /** Formats results as a numbered context block for a system prompt, when you retrieve before calling `chat()`. */
  function formatContext(results: RagResult[]): string {
    return results
      .map((r) => {
        const where = r.headings?.length ? ` (${r.headings.join(' > ')})` : '';
        return `[${r.citation}] ${r.source}${where}\n${r.content}`;
      })
      .join('\n\n');
  }

  /** A server tool for `chat({ tools })`: the model searches the knowledge base and cites results as `[n]`. */
  function tool(toolOptions: RagToolOptions = {}) {
    return toolDefinition({
      name: toolOptions.name ?? 'search_knowledge_base',
      description:
        toolOptions.description ??
        'Search the knowledge base for passages relevant to the question. Results are numbered; cite them as [n].',
      inputSchema,
    }).server(async (args, context): Promise<RagToolOutput> => {
      const query = String((args as { query?: unknown } | undefined)?.query ?? '').trim();
      if (!query) return { query, results: [] };
      const results = await retrieve(query, {
        ...(toolOptions.topK !== undefined ? { topK: toolOptions.topK } : {}),
        ...(toolOptions.minScore !== undefined ? { minScore: toolOptions.minScore } : {}),
        ...(toolOptions.filter ? { filter: toolOptions.filter } : {}),
        ...(context?.abortSignal ? { abortSignal: context.abortSignal } : {}),
      });
      return {
        query,
        results: results.map((r) => ({
          citation: r.citation,
          source: r.source,
          ...(r.url ? { url: r.url } : {}),
          ...(r.headings ? { headings: r.headings } : {}),
          content: r.content,
          score: Math.round((r.rerankScore ?? r.score) * 1000) / 1000,
        })),
      };
    });
  }

  return {
    ingest,
    retrieve,
    formatContext,
    tool,
    /** Deletes every chunk of the given documents from the store. */
    remove: (documentIds: string | string[]) =>
      store.deleteDocuments(Array.isArray(documentIds) ? documentIds : [documentIds]),
    store,
  };
}

export type Rag = ReturnType<typeof createRag>;
