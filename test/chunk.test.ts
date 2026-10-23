import { getEncoding } from 'js-tiktoken';
import { describe, expect, it } from 'vitest';
import { chunkMarkdown, chunkText, estimateTokens } from '../src';

const sentence = (i: number) => `Sentence number ${i} talks about topic ${i % 7} in some detail.`;
const prose = Array.from({ length: 12 }, (_, p) =>
  Array.from({ length: 6 }, (_, s) => sentence(p * 6 + s)).join(' '),
).join('\n\n');

describe('chunkText', () => {
  it('keeps every chunk within maxTokens and points offsets at the source', () => {
    const chunks = chunkText(prose, { maxTokens: 60, overlap: 10 });
    expect(chunks.length).toBeGreaterThan(5);
    for (const c of chunks) {
      expect(c.tokens).toBeLessThanOrEqual(60);
      expect(prose.slice(c.start, c.end)).toBe(c.content);
    }
    expect(chunks.map((c) => c.index)).toEqual(chunks.map((_, i) => i));
  });

  it('covers the whole text, in order', () => {
    const chunks = chunkText(prose, { maxTokens: 60, overlap: 0 });
    const words = prose.split(/\s+/);
    expect(
      chunks
        .map((c) => c.content)
        .join(' ')
        .split(/\s+/),
    ).toEqual(words);
  });

  it('repeats trailing text of one chunk at the start of the next when overlap is set', () => {
    const chunks = chunkText(prose, { maxTokens: 60, overlap: 20 });
    let overlapping = 0;
    for (let i = 1; i < chunks.length; i++) {
      if ((chunks[i] as { start: number }).start < (chunks[i - 1] as { end: number }).end) overlapping++;
    }
    expect(overlapping).toBeGreaterThan(0);
  });

  it('prefers paragraph breaks over sentence breaks', () => {
    const text = `${'alpha '.repeat(30).trim()}.\n\n${'beta '.repeat(30).trim()}.`;
    const chunks = chunkText(text, { maxTokens: 50, overlap: 0 });
    expect(chunks.map((c) => c.content[0])).toEqual(['a', 'b']);
  });

  it('hard-splits a single word longer than maxTokens', () => {
    const chunks = chunkText('x'.repeat(100), { maxTokens: 10, overlap: 0 });
    expect(chunks.map((c) => c.content.length)).toEqual([40, 40, 20]);
  });

  it('measures with a real tokenizer when one is passed', () => {
    const enc = getEncoding('cl100k_base');
    const countTokens = (t: string) => enc.encode(t).length;
    const chunks = chunkText(prose, { maxTokens: 50, overlap: 8, countTokens });
    for (const c of chunks) expect(countTokens(c.content)).toBeLessThanOrEqual(50);
    expect(chunks[0]?.tokens).toBe(countTokens(chunks[0]?.content ?? ''));
  });

  it('returns nothing for empty or blank text', () => {
    expect(chunkText('')).toEqual([]);
    expect(chunkText('  \n\n  ')).toEqual([]);
  });

  it('rejects overlap >= maxTokens', () => {
    expect(() => chunkText('a', { maxTokens: 10, overlap: 10 })).toThrow(RangeError);
    expect(() => chunkText('a', { maxTokens: 0 })).toThrow(RangeError);
  });

  it('estimates about four characters per token', () => {
    expect(estimateTokens('abcdefgh')).toBe(2);
  });
});

describe('chunkMarkdown', () => {
  const md = `Intro paragraph before any heading.

# Guide

Welcome to the guide.

## Install

Run the installer.

\`\`\`bash
npm install acme

# not a heading, inside a fence
npm run setup
\`\`\`

## Usage
### Basics

Call \`acme()\` with no arguments.

# FAQ

Is it free? Yes.
`;

  it('records the heading path of every chunk and never crosses headings', () => {
    const chunks = chunkMarkdown(md, { maxTokens: 200, overlap: 0 });
    expect(chunks.map((c) => c.headings)).toEqual([
      undefined,
      ['Guide'],
      ['Guide', 'Install'],
      ['Guide', 'Usage', 'Basics'],
      ['FAQ'],
    ]);
    expect(chunks[2]?.content).toContain('# not a heading, inside a fence');
    expect(chunks[2]?.content.startsWith('## Install')).toBe(true);
  });

  it('skips a heading with no body of its own (## Usage directly followed by ### Basics)', () => {
    const chunks = chunkMarkdown(md, { maxTokens: 200, overlap: 0 });
    expect(chunks.some((c) => c.content.trim() === '## Usage')).toBe(false);
  });

  it('keeps a fenced code block whole when it fits', () => {
    const chunks = chunkMarkdown(md, { maxTokens: 30, overlap: 0 });
    const fenced = chunks.find((c) => c.content.includes('npm install acme'));
    expect(fenced?.content).toContain('npm run setup');
    expect(fenced?.content.trim().endsWith('```')).toBe(true);
  });

  it('splits a long section and keeps its heading path on each piece', () => {
    const long = `# Big\n\n${prose}`;
    const chunks = chunkMarkdown(long, { maxTokens: 60, overlap: 10 });
    expect(chunks.length).toBeGreaterThan(5);
    for (const c of chunks) {
      expect(c.headings).toEqual(['Big']);
      expect(c.tokens).toBeLessThanOrEqual(60);
      expect(long.slice(c.start, c.end)).toBe(c.content);
    }
  });

  it('handles CRLF line endings and ~~~ fences', () => {
    const crlf = '# A\r\n\r\ntext a\r\n\r\n~~~\r\n# inside\r\n~~~\r\n\r\n# B\r\n\r\ntext b\r\n';
    const chunks = chunkMarkdown(crlf, { maxTokens: 100, overlap: 0 });
    expect(chunks.map((c) => c.headings)).toEqual([['A'], ['B']]);
    expect(chunks[0]?.content).toContain('# inside');
  });
});

describe('overlap with long paragraphs', () => {
  // Ten ~150-token paragraphs: no whole paragraph fits in a 64-token overlap.
  const para = (n: number) =>
    Array.from({ length: 12 }, (_, s) => `Paragraph ${n} sentence ${s} explains part ${s} of the topic.`).join(' ');
  const text = Array.from({ length: 10 }, (_, n) => para(n)).join('\n\n');

  it('still overlaps every neighbouring pair, by at most `overlap` tokens', () => {
    expect(estimateTokens(para(0))).toBeGreaterThan(140);
    const chunks = chunkText(text, { maxTokens: 512, overlap: 64 });
    expect(chunks.length).toBeGreaterThan(3);
    for (let i = 1; i < chunks.length; i++) {
      const prev = chunks[i - 1] as { start: number; end: number };
      const cur = chunks[i] as { start: number; end: number; tokens: number };
      expect(cur.start).toBeLessThan(prev.end);
      const shared = estimateTokens(text.slice(cur.start, prev.end));
      expect(shared).toBeLessThanOrEqual(64);
      expect(shared).toBeGreaterThan(48);
      expect(cur.tokens).toBeLessThanOrEqual(512);
      expect(/\s/.test(text[cur.start - 1] as string)).toBe(true); // starts at a word boundary
    }
  });

  it('does the same with a real tokenizer', () => {
    const enc = getEncoding('cl100k_base');
    const countTokens = (t: string) => enc.encode(t).length;
    const chunks = chunkText(text, { maxTokens: 400, overlap: 50, countTokens });
    for (let i = 1; i < chunks.length; i++) {
      const prev = chunks[i - 1] as { end: number };
      const cur = chunks[i] as { start: number; content: string };
      expect(cur.start).toBeLessThan(prev.end);
      expect(countTokens(text.slice(cur.start, prev.end))).toBeLessThanOrEqual(50);
      expect(countTokens(cur.content)).toBeLessThanOrEqual(400);
    }
  });

  it('falls back to characters when no word boundary fits', () => {
    const src = `${'a'.repeat(36)}\n\n${'b'.repeat(8)}`;
    const chunks = chunkText(src, { maxTokens: 10, overlap: 2 });
    expect(chunks.map((c) => [c.start, c.end])).toEqual([
      [0, 36],
      [28, 46],
    ]);
  });

  it('gives less overlap when the next piece leaves no room', () => {
    const chunks = chunkText('x'.repeat(100), { maxTokens: 10, overlap: 2 });
    expect(chunks.map((c) => [c.start, c.end])).toEqual([
      [0, 40],
      [40, 80],
      [72, 100],
    ]);
  });

  it('overlap 0 gives none', () => {
    const chunks = chunkText(text, { maxTokens: 512, overlap: 0 });
    for (let i = 1; i < chunks.length; i++) {
      expect((chunks[i] as { start: number }).start).toBeGreaterThanOrEqual((chunks[i - 1] as { end: number }).end);
    }
  });

  it('markdown sections get the same overlap inside a section, never across headings', () => {
    const md = `# A\n\n${text}\n\n# B\n\nshort`;
    const chunks = chunkMarkdown(md, { maxTokens: 512, overlap: 64 });
    const a = chunks.filter((c) => c.headings?.[0] === 'A');
    for (let i = 1; i < a.length; i++)
      expect((a[i] as { start: number }).start).toBeLessThan((a[i - 1] as { end: number }).end);
    const b = chunks.find((c) => c.headings?.[0] === 'B');
    expect(b?.content).toBe('# B\n\nshort');
  });
});
