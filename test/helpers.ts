import { createServer, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import type { EmbeddingOptions, EmbeddingResult } from '@tanstack/ai';
import { BaseEmbeddingAdapter } from '@tanstack/ai/adapters';

const STOP = new Set(
  'a an and are as at be by do for from how i in is it of on or the to with what which you your'.split(' '),
);

/** Bag-of-words vector: each non-stop word (crudely stemmed) hashed into one of `dims` buckets, then L2-normalised. */
export function bagOfWords(text: string, dims = 256): number[] {
  const v = new Array<number>(dims).fill(0);
  for (const raw of text.toLowerCase().match(/[a-z0-9]+/g) ?? []) {
    if (STOP.has(raw)) continue;
    const word = raw.replace(/(ing|ed|es|s)$/, '');
    let h = 2166136261;
    for (let i = 0; i < word.length; i++) h = Math.imul(h ^ word.charCodeAt(i), 16777619);
    v[(h >>> 0) % dims]! += 1;
  }
  const n = Math.hypot(...v) || 1;
  return v.map((x) => x / n);
}

/** In-process embedding adapter built on `BaseEmbeddingAdapter`, for tests that need meaningful similarity. */
export class BagOfWordsEmbedding extends BaseEmbeddingAdapter<'bow'> {
  readonly name = 'bow';
  calls: string[][] = [];
  constructor(private dims = 256) {
    super('bow');
  }
  async createEmbeddings(options: EmbeddingOptions): Promise<EmbeddingResult> {
    const texts = options.input.map((i) => (typeof i === 'string' ? i : JSON.stringify(i)));
    this.calls.push(texts);
    return {
      id: `emb-${this.calls.length}`,
      model: 'bow',
      embeddings: texts.map((t, index) => ({ vector: bagOfWords(t, this.dims), index })),
    };
  }
}

/**
 * A local stand-in for OpenAI's `POST /v1/embeddings` that returns a bag-of-words vector per input, so the real
 * `@tanstack/ai-openai` adapter (and the OpenAI SDK under it) can be exercised with meaningful similarity offline.
 */
export async function startEmbeddingServer(
  dims = 256,
): Promise<{ url: string; requests: unknown[]; close(): Promise<void> }> {
  const requests: unknown[] = [];
  const server: Server = createServer((req, res) => {
    let body = '';
    req.on('data', (c) => {
      body += c;
    });
    req.on('end', () => {
      if (req.method !== 'POST' || !req.url?.endsWith('/embeddings')) {
        res.writeHead(404).end();
        return;
      }
      const parsed = JSON.parse(body) as { input: string | string[]; model: string; dimensions?: number };
      requests.push(parsed);
      const inputs = Array.isArray(parsed.input) ? parsed.input : [parsed.input];
      const size = parsed.dimensions ?? dims;
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(
        JSON.stringify({
          object: 'list',
          model: parsed.model,
          data: inputs.map((t, index) => ({ object: 'embedding', index, embedding: bagOfWords(t, size) })),
          usage: { prompt_tokens: inputs.length, total_tokens: inputs.length },
        }),
      );
    });
  });
  await new Promise<void>((r) => server.listen(0, '127.0.0.1', r));
  const { port } = server.address() as AddressInfo;
  return {
    url: `http://127.0.0.1:${port}/v1`,
    requests,
    close: () => new Promise((r) => server.close(() => r())),
  };
}

export const DOCS = {
  install: `# Install

Run \`npm install acme-widgets\` in your project. The package needs Node 20 or newer.

## Configure

Create \`acme.config.ts\` and export your API region. The default region is eu-west.
`,
  pricing: `# Pricing

The free plan includes 1,000 widget renders per month. The team plan costs 49 dollars per month and adds SSO.
`,
  refunds: `# Refunds

Refunds are available within 30 days of purchase. Email billing to request a refund.
`,
};
