import { BaseRerankAdapter, type RerankAdapterResult, type RerankOptions } from '@tanstack/ai';
import { describe, expect, it } from 'vitest';
import { citationInstructions, createRag, memoryVectorStore, type RagToolOutput } from '../src';
import { BagOfWordsEmbedding, DOCS } from './helpers';

const docs = [
  {
    id: 'docs/install.md',
    title: 'Install guide',
    url: 'https://example.com/install',
    content: DOCS.install,
    format: 'markdown' as const,
    metadata: { product: 'widgets' },
  },
  {
    id: 'docs/pricing.md',
    title: 'Pricing',
    content: DOCS.pricing,
    format: 'markdown' as const,
    metadata: { product: 'widgets' },
  },
  { id: 'docs/refunds.md', content: DOCS.refunds, format: 'markdown' as const, metadata: { product: 'billing' } },
];

function setup(options: Partial<Parameters<typeof createRag>[0]> = {}) {
  const embedding = new BagOfWordsEmbedding();
  const store = memoryVectorStore();
  const rag = createRag({ embedding, store, ...options });
  return { rag, store, embedding };
}

describe('createRag', () => {
  it('ingests, then retrieves the most relevant chunk first with citation data', async () => {
    const { rag } = setup();
    expect(await rag.ingest(docs)).toEqual({ documents: 3, chunks: 4 });

    const results = await rag.retrieve('How much does the team plan cost per month?');
    expect(results[0]).toMatchObject({
      documentId: 'docs/pricing.md',
      citation: 1,
      source: 'Pricing',
      headings: ['Pricing'],
    });
    expect(results.map((r) => r.citation)).toEqual(results.map((_, i) => i + 1));

    const [install] = await rag.retrieve('which Node version do I need to install it', { topK: 1 });
    expect(install).toMatchObject({
      id: 'docs/install.md#0',
      source: 'Install guide',
      url: 'https://example.com/install',
      headings: ['Install'],
      metadata: { product: 'widgets', title: 'Install guide', url: 'https://example.com/install', chunkIndex: 0 },
    });
    expect(install?.content).toContain('npm install acme-widgets');
  });

  it('falls back to the document id as the citation source', async () => {
    const { rag } = setup();
    await rag.ingest(docs);
    const [hit] = await rag.retrieve('refund within 30 days', { topK: 1 });
    expect(hit).toMatchObject({ documentId: 'docs/refunds.md', source: 'docs/refunds.md' });
  });

  it('embeds the title and heading path with each chunk, but stores only the chunk text', async () => {
    const { rag, embedding, store } = setup();
    await rag.ingest(docs[0] as (typeof docs)[0]);
    expect(embedding.calls[0]?.[1]).toBe(
      'Install guide > Install > Configure\n\n## Configure\n\nCreate `acme.config.ts` and export your API region. The default region is eu-west.',
    );
    const [hit] = await store.query({ vector: new Array(256).fill(1), topK: 10, filter: { chunkIndex: 1 } });
    expect(hit?.content.startsWith('## Configure')).toBe(true);
  });

  it('replaces a document on re-ingest and removes it on remove()', async () => {
    const { rag, store } = setup();
    await rag.ingest(docs);
    expect(store.size).toBe(4);
    await rag.ingest({ id: 'docs/install.md', content: 'Install with pnpm add acme-widgets.' });
    expect(store.size).toBe(3);
    const [hit] = await rag.retrieve('pnpm', { topK: 1 });
    expect(hit?.content).toBe('Install with pnpm add acme-widgets.');
    await rag.remove(['docs/install.md', 'docs/pricing.md']);
    expect(store.size).toBe(1);
  });

  it('keeps the old chunks when embedding fails', async () => {
    const { rag, store, embedding } = setup();
    await rag.ingest(docs);
    embedding.createEmbeddings = async () => {
      throw new Error('provider down');
    };
    await expect(rag.ingest({ id: 'docs/pricing.md', content: 'new text' })).rejects.toThrow('provider down');
    expect(store.size).toBe(4);
  });

  it('filters by metadata and drops matches below minScore', async () => {
    const { rag } = setup();
    await rag.ingest(docs);
    const billing = await rag.retrieve('install plan cost refund', { filter: { product: 'billing' } });
    expect(billing.map((r) => r.documentId)).toEqual(['docs/refunds.md']);
    const strict = await rag.retrieve('refund within 30 days', { minScore: 0.3 });
    expect(strict.map((r) => r.documentId)).toEqual(['docs/refunds.md']);
  });

  it('embeds in batches of batchSize and forwards dimensions and model options', async () => {
    const { rag, embedding } = setup({
      batchSize: 2,
      dimensions: 256,
      documentModelOptions: { inputType: 'search_document' },
      queryModelOptions: { inputType: 'search_query' },
    });
    const seen: unknown[] = [];
    const original = embedding.createEmbeddings.bind(embedding);
    embedding.createEmbeddings = async (o) => {
      seen.push({ n: o.input.length, dimensions: o.dimensions, modelOptions: o.modelOptions });
      return original(o);
    };
    await rag.ingest(docs);
    await rag.retrieve('refund');
    expect(seen).toEqual([
      { n: 2, dimensions: 256, modelOptions: { inputType: 'search_document' } },
      { n: 2, dimensions: 256, modelOptions: { inputType: 'search_document' } },
      { n: 1, dimensions: 256, modelOptions: { inputType: 'search_query' } },
    ]);
  });

  it('rejects documents without an id or with duplicate ids', async () => {
    const { rag } = setup();
    await expect(rag.ingest({ id: '', content: 'x' })).rejects.toThrow(/id/);
    await expect(
      rag.ingest([
        { id: 'a', content: 'x' },
        { id: 'a', content: 'y' },
      ]),
    ).rejects.toThrow(/duplicate/);
  });

  it('chunks plain text by default and markdown on request', async () => {
    const { rag, store } = setup({ chunking: { maxTokens: 20, overlap: 0 } });
    await rag.ingest({ id: 'plain', content: DOCS.install });
    const plain = await store.query({ vector: new Array(256).fill(1), topK: 50 });
    expect(plain.every((m) => m.metadata.headings === undefined)).toBe(true);
    await rag.ingest({ id: 'plain', content: DOCS.install, format: 'markdown' });
    const md = await store.query({ vector: new Array(256).fill(1), topK: 50 });
    expect(md.every((m) => Array.isArray(m.metadata.headings))).toBe(true);
  });

  it('formats a numbered context block', async () => {
    const { rag } = setup();
    await rag.ingest(docs);
    const results = await rag.retrieve('team plan cost', { topK: 2 });
    const context = rag.formatContext(results);
    expect(context.startsWith('[1] Pricing (Pricing)\n# Pricing')).toBe(true);
    expect(context).toContain('\n\n[2] ');
  });

  it('exports citation instructions for the system prompt', () => {
    expect(citationInstructions).toMatch(/\[1\]/);
  });
});

describe('rag.tool()', () => {
  it('is a server tool whose output lists numbered, cited results', async () => {
    const { rag } = setup();
    await rag.ingest(docs);
    const tool = rag.tool({ topK: 2 });
    expect(tool).toMatchObject({ name: 'search_knowledge_base', __toolSide: 'server' });
    expect(tool.inputSchema).toMatchObject({ required: ['query'] });
    const out = await tool.execute?.({ query: 'refund policy days' });
    expect(out).toMatchObject({
      query: 'refund policy days',
      results: [{ citation: 1, source: 'docs/refunds.md' }, { citation: 2 }],
    });
  });

  it('applies the fixed filter, custom name and description', async () => {
    const { rag } = setup();
    await rag.ingest(docs);
    const tool = rag.tool({ name: 'search_billing', description: 'Billing docs', filter: { product: 'billing' } });
    expect(tool).toMatchObject({ name: 'search_billing', description: 'Billing docs' });
    const out = (await tool.execute?.({ query: 'install plan cost' })) as { results: { source: string }[] };
    expect(out.results.map((r) => r.source)).toEqual(['docs/refunds.md']);
  });

  it('returns no results for an empty query without calling the embedder', async () => {
    const { rag, embedding } = setup();
    const tool = rag.tool();
    expect(await tool.execute?.({ query: '  ' })).toEqual({ query: '', results: [] });
    expect(embedding.calls).toEqual([]);
  });
});

class ReverseRerank extends BaseRerankAdapter<'reverse'> {
  readonly name = 'reverse';
  seen: RerankOptions[] = [];
  constructor() {
    super({}, 'reverse');
  }
  async rerank(options: RerankOptions): Promise<RerankAdapterResult> {
    this.seen.push(options);
    const n = options.documents.length;
    return {
      id: 'rr',
      ranking: options.documents.map((_, i) => ({ index: n - 1 - i, score: (n - i) / 10 })).slice(0, options.topN ?? n),
      usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
    };
  }
}

describe('rerank', () => {
  it('reranks topK * 4 candidates, keeps topK in the reranker order and reports rerankScore', async () => {
    const reranker = new ReverseRerank();
    const { rag } = setup({ topK: 2, rerank: { adapter: reranker } });
    await rag.ingest(docs);
    const plain = await createRag({ embedding: new BagOfWordsEmbedding(), store: setupStore(rag), topK: 4 }).retrieve(
      'refund policy',
    );
    const results = await rag.retrieve('refund policy');
    expect(reranker.seen[0]?.documents).toHaveLength(4);
    expect(reranker.seen[0]?.topN).toBe(2);
    expect(results.map((r) => r.id)).toEqual([plain[3]?.id, plain[2]?.id]);
    expect(results.map((r) => [r.citation, r.rerankScore])).toEqual([
      [1, 0.4],
      [2, 0.3],
    ]);
  });

  it('applies minScore before reranking and the tool reports rerankScore as score', async () => {
    const reranker = new ReverseRerank();
    const { rag } = setup({ topK: 3, minScore: 0.2, rerank: { adapter: reranker, candidates: 10 } });
    await rag.ingest(docs);
    const out = (await rag.tool().execute?.({ query: 'refund within 30 days' })) as RagToolOutput;
    expect(reranker.seen[0]?.documents).toHaveLength(1);
    expect(out.results).toEqual([expect.objectContaining({ citation: 1, source: 'docs/refunds.md', score: 0.1 })]);
  });

  it('passes the tool call abort signal to the reranker', async () => {
    const reranker = new ReverseRerank();
    const { rag } = setup({ rerank: { adapter: reranker } });
    await rag.ingest(docs);
    const controller = new AbortController();
    await rag.tool().execute?.({ query: 'refund' }, { abortSignal: controller.signal } as never);
    expect(reranker.seen[0]?.abortSignal).toBe(controller.signal);
  });

  it('skips the reranker when nothing matched', async () => {
    const reranker = new ReverseRerank();
    const { rag } = setup({ rerank: { adapter: reranker } });
    expect(await rag.retrieve('anything')).toEqual([]);
    expect(reranker.seen).toEqual([]);
  });
});

const setupStore = (rag: ReturnType<typeof createRag>) => rag.store;

describe('citation numbering in rag.tool()', () => {
  const call = async (tool: ReturnType<ReturnType<typeof createRag>['tool']>, query: string, signal?: AbortSignal) =>
    (await tool.execute?.({ query }, (signal ? { abortSignal: signal } : {}) as never)) as RagToolOutput;

  it('continues numbers across calls in one run and keeps a chunk number stable', async () => {
    const { rag } = setup({ topK: 2 });
    await rag.ingest(docs);
    const tool = rag.tool();
    const run = new AbortController().signal;
    const first = await call(tool, 'refund within 30 days', run);
    const second = await call(tool, 'team plan cost per month', run);
    const again = await call(tool, 'refund within 30 days', run);
    expect(first.results.map((r) => r.citation)).toEqual([1, 2]);
    const byId = new Map([...first.results, ...second.results].map((r) => [r.id, r.citation]));
    expect(new Set(byId.values()).size).toBe(byId.size); // no number used twice
    for (const r of second.results) {
      const seen = first.results.find((f) => f.id === r.id);
      expect(r.citation).toEqual(seen ? seen.citation : expect.any(Number));
    }
    expect(second.results.some((r) => r.citation > 2)).toBe(true);
    expect(again.results.map((r) => r.citation)).toEqual(first.results.map((r) => r.citation));
  });

  it('starts again at 1 in a new run, with citations: "call", and without a signal', async () => {
    const { rag } = setup({ topK: 2 });
    await rag.ingest(docs);
    const tool = rag.tool();
    await call(tool, 'refund within 30 days', new AbortController().signal);
    const other = await call(tool, 'team plan cost per month', new AbortController().signal);
    expect(other.results.map((r) => r.citation)).toEqual([1, 2]);
    const perCall = rag.tool({ citations: 'call' });
    const run = new AbortController().signal;
    await call(perCall, 'refund within 30 days', run);
    expect((await call(perCall, 'team plan cost per month', run)).results.map((r) => r.citation)).toEqual([1, 2]);
    expect((await call(tool, 'team plan cost per month')).results.map((r) => r.citation)).toEqual([1, 2]);
  });

  it('returns the chunk id with each result', async () => {
    const { rag } = setup({ topK: 1 });
    await rag.ingest(docs);
    expect((await call(rag.tool(), 'refund within 30 days')).results[0]?.id).toBe('docs/refunds.md#0');
  });
});
