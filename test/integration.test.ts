/**
 * Real TanStack AI adapters over HTTP against local servers: `@tanstack/ai-openai` embeddings against a local
 * OpenAI-compatible embeddings endpoint, `@tanstack/ai-openai` chat and `@tanstack/ai-cohere` rerank against aimock.
 * No live provider API is called and no API key is used.
 */

import { LLMock } from '@copilotkit/aimock';
import { PGlite } from '@electric-sql/pglite';
import { vector } from '@electric-sql/pglite-pgvector';
import { chat, type StreamChunk } from '@tanstack/ai';
import { createCohereRerank } from '@tanstack/ai-cohere';
import { createOpenaiChat, createOpenaiChatCompletions, createOpenaiEmbedding } from '@tanstack/ai-openai';
import pg from 'pg';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { citationInstructions, createRag, pgvectorStore, type RagToolOutput, type VectorStore } from '../src';
import { DOCS, startEmbeddingServer } from './helpers';

const PG_URL = process.env.PGVECTOR_URL;
const key = 'mock-key';
const mock = new LLMock({ port: 0 });
let mockUrl = '';
let embeddings: Awaited<ReturnType<typeof startEmbeddingServer>>;

const docs = [
  { id: 'docs/install.md', title: 'Install guide', content: DOCS.install, format: 'markdown' as const },
  { id: 'docs/pricing.md', title: 'Pricing', content: DOCS.pricing, format: 'markdown' as const },
  { id: 'docs/refunds.md', title: 'Refund policy', content: DOCS.refunds, format: 'markdown' as const },
];

/** Answers from whatever the retrieval tool returned, citing the first result, like a model following the prompt. */
function answerFromToolResult(req: { messages: { role: string; content: unknown }[] }) {
  const last = req.messages.at(-1);
  const text = typeof last?.content === 'string' ? last.content : JSON.stringify(last?.content);
  const out = JSON.parse(text) as RagToolOutput;
  const first = out.results[0];
  return {
    content: first ? `From ${first.source}: ${first.content.split('\n').at(-1)} [${first.citation}]` : 'Not found.',
  };
}

beforeAll(async () => {
  embeddings = await startEmbeddingServer();
  mock.addFixture({
    match: { userMessage: /refund/i, hasToolResult: false },
    response: {
      toolCalls: [{ name: 'search_knowledge_base', arguments: JSON.stringify({ query: 'refund policy days' }) }],
    },
  });
  mock.addFixture({ match: { userMessage: /refund/i, hasToolResult: true }, response: answerFromToolResult });
  // Only answers when the retrieved context reached the system prompt.
  mock.addFixture({
    match: { userMessage: /pricing question/, systemMessage: ['[1] Pricing (Pricing)', '49 dollars'] },
    response: { content: 'The team plan is 49 dollars a month [1].' },
  });
  mock.onRerank(/refund/, [
    { index: 2, relevance_score: 0.97 },
    { index: 0, relevance_score: 0.41 },
  ]);
  mockUrl = await mock.start();
});

afterAll(async () => {
  await mock.stop();
  await embeddings.close();
});

const embeddingAdapter = () => createOpenaiEmbedding('text-embedding-3-small', key, { baseURL: embeddings.url });

async function collect(stream: AsyncIterable<StreamChunk>) {
  const chunks: StreamChunk[] = [];
  for await (const c of stream) chunks.push(c);
  return chunks;
}

const stores: [string, () => Promise<{ store: VectorStore; close: () => Promise<void> }>][] = [
  [
    'PGlite + pgvector',
    async () => {
      const db = new PGlite({ extensions: { vector } });
      const store = pgvectorStore({ db, dimensions: 256 });
      await store.migrate();
      return { store, close: () => db.close() };
    },
  ],
];
if (PG_URL) {
  stores.push([
    'Postgres + pgvector in Docker (pg Pool)',
    async () => {
      const pool = new pg.Pool({ connectionString: PG_URL });
      await pool.query('DROP TABLE IF EXISTS rag_it');
      const store = pgvectorStore({ db: pool, table: 'rag_it', dimensions: 256 });
      await store.migrate();
      return { store, close: () => pool.end() };
    },
  ]);
}

describe.each(stores)('%s with @tanstack/ai-openai embeddings', (_label, makeStore) => {
  let store: VectorStore;
  let close: () => Promise<void>;
  beforeAll(async () => {
    ({ store, close } = await makeStore());
  });
  afterAll(() => close());

  it('ingests through the OpenAI embeddings endpoint in batches', async () => {
    const before = embeddings.requests.length;
    const rag = createRag({ embedding: embeddingAdapter(), store, batchSize: 2 });
    expect(await rag.ingest(docs)).toEqual({ documents: 3, chunks: 4 });
    expect(embeddings.requests.length - before).toBe(2);
    expect(embeddings.requests.at(-1)).toMatchObject({ model: 'text-embedding-3-small' });
  });

  it('retrieves the right chunk for a question', async () => {
    const rag = createRag({ embedding: embeddingAdapter(), store });
    const [top] = await rag.retrieve('How many days do I have to ask for a refund?');
    expect(top).toMatchObject({ documentId: 'docs/refunds.md', source: 'Refund policy', citation: 1 });
  });

  it('runs the retrieval tool inside chat() and the answer cites it', async () => {
    const rag = createRag({ embedding: embeddingAdapter(), store, topK: 3 });
    const chunks = await collect(
      chat({
        threadId: 't1',
        adapter: createOpenaiChat('gpt-4o-mini', key, { baseURL: `${mockUrl}/v1` }),
        systemPrompts: [citationInstructions],
        messages: [{ role: 'user', content: 'What is the refund policy?' }],
        tools: [rag.tool()],
      }),
    );
    expect(chunks.find((c) => c.type === 'RUN_ERROR')).toBeUndefined();
    const result = chunks.find((c) => c.type === 'TOOL_CALL_RESULT') as { content: string } | undefined;
    const output = JSON.parse(result?.content ?? '{}') as RagToolOutput;
    expect(output.query).toBe('refund policy days');
    expect(output.results[0]).toMatchObject({ citation: 1, source: 'Refund policy' });
    const text = chunks
      .filter((c) => c.type === 'TEXT_MESSAGE_CONTENT')
      .map((c) => (c as { delta: string }).delta)
      .join('');
    expect(text).toBe(
      'From Refund policy: Refunds are available within 30 days of purchase. Email billing to request a refund. [1]',
    );
  });
});

describe('retrieve first, then chat with the context in the system prompt', () => {
  it('sends formatContext() output to the model', async () => {
    const db = new PGlite({ extensions: { vector } });
    const store = pgvectorStore({ db, dimensions: 256 });
    await store.migrate();
    const rag = createRag({ embedding: embeddingAdapter(), store, topK: 1 });
    await rag.ingest(docs);
    const question = 'A pricing question: what does the team plan cost per month?';
    const results = await rag.retrieve(question);
    const chunks = await collect(
      chat({
        threadId: 't3',
        adapter: createOpenaiChat('gpt-4o-mini', key, { baseURL: `${mockUrl}/v1` }),
        systemPrompts: [citationInstructions, rag.formatContext(results)],
        messages: [{ role: 'user', content: question }],
      }),
    );
    const text = chunks
      .filter((c) => c.type === 'TEXT_MESSAGE_CONTENT')
      .map((c) => (c as { delta: string }).delta)
      .join('');
    expect(text).toBe('The team plan is 49 dollars a month [1].');
    await db.close();
  });
});

describe('rerank with @tanstack/ai-cohere', () => {
  it('reorders vector matches by the reranker and records rerankScore', async () => {
    const db = new PGlite({ extensions: { vector } });
    const store = pgvectorStore({ db, dimensions: 256 });
    await store.migrate();
    const rag = createRag({
      embedding: embeddingAdapter(),
      store,
      topK: 2,
      rerank: { adapter: createCohereRerank('rerank-v3.5', key, { baseURL: mockUrl }), candidates: 4 },
    });
    await rag.ingest(docs);
    const plain = await createRag({ embedding: embeddingAdapter(), store, topK: 4 }).retrieve('refund');
    const reranked = await rag.retrieve('refund');
    expect(reranked).toHaveLength(2);
    expect(reranked[0]?.id).toBe(plain[2]?.id);
    expect(reranked[1]?.id).toBe(plain[0]?.id);
    expect(reranked.map((r) => [r.citation, r.rerankScore])).toEqual([
      [1, 0.97],
      [2, 0.41],
    ]);
    await db.close();
  });
});

describe('chat completions adapter', () => {
  it('also drives the tool through openaiChatCompletions', async () => {
    const db = new PGlite({ extensions: { vector } });
    const store = pgvectorStore({ db, dimensions: 256 });
    await store.migrate();
    const rag = createRag({ embedding: embeddingAdapter(), store });
    await rag.ingest(docs);
    const chunks = await collect(
      chat({
        threadId: 't2',
        adapter: createOpenaiChatCompletions('gpt-4o-mini', key, { baseURL: `${mockUrl}/v1` }),
        messages: [{ role: 'user', content: 'Tell me about refunds' }],
        tools: [rag.tool()],
      }),
    );
    expect(chunks.find((c) => c.type === 'RUN_ERROR')).toBeUndefined();
    const text = chunks
      .filter((c) => c.type === 'TEXT_MESSAGE_CONTENT')
      .map((c) => (c as { delta: string }).delta)
      .join('');
    expect(text).toMatch(/30 days .*\[1\]$/);
    await db.close();
  });
});
