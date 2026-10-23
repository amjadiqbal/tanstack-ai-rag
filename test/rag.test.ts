import { describe, expect, it } from 'vitest';
import { citationInstructions, createRag, memoryVectorStore } from '../src';
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
