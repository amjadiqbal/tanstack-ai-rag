/** Counts tokens in a piece of text. Pass your model's tokenizer for exact sizes. */
export type CountTokens = (text: string) => number;

export interface ChunkOptions {
  /** Largest chunk, in tokens as measured by `countTokens`. Default 512. */
  maxTokens?: number;
  /** Tokens repeated from the end of one chunk at the start of the next. Default 64. Must be below `maxTokens`. */
  overlap?: number;
  /**
   * Token counter. The default, {@link estimateTokens}, is a fast approximation (about 4 characters per token).
   * Pass a real tokenizer (for example `js-tiktoken`) when chunks must fit a hard model limit.
   */
  countTokens?: CountTokens;
}

export interface Chunk {
  /** The chunk text, trimmed. */
  content: string;
  /** Position of the chunk in the document, from 0. */
  index: number;
  /** Size of `content` according to `countTokens`. */
  tokens: number;
  /** Character offset of `content` in the source text. */
  start: number;
  /** Character offset just past `content` in the source text. */
  end: number;
  /** Markdown only: the heading path the chunk sits under, outermost first. */
  headings?: string[];
}

/** Rough token count: about 4 characters per token for English prose and code. */
export const estimateTokens: CountTokens = (text) => Math.ceil(text.length / 4);

interface Span {
  start: number;
  end: number;
}

interface Resolved {
  maxTokens: number;
  overlap: number;
  count: CountTokens;
}

function resolve(options: ChunkOptions = {}): Resolved {
  const maxTokens = options.maxTokens ?? 512;
  const overlap = options.overlap ?? Math.min(64, Math.floor(maxTokens / 4));
  if (!Number.isInteger(maxTokens) || maxTokens < 1) throw new RangeError('maxTokens must be a positive integer');
  if (!Number.isInteger(overlap) || overlap < 0 || overlap >= maxTokens) {
    throw new RangeError('overlap must be an integer from 0 to maxTokens - 1');
  }
  return { maxTokens, overlap, count: options.countTokens ?? estimateTokens };
}

const SEPARATORS: RegExp[] = [/\n[ \t]*\n/g, /\n/g, /(?<=[.!?])\s+/g, /\s+/g];

/** Splits a span at every match of `sep`, keeping the separator on the left piece so the pieces tile the span. */
function splitSpan(text: string, span: Span, sep: RegExp): Span[] {
  const out: Span[] = [];
  const slice = text.slice(span.start, span.end);
  let from = 0;
  for (const m of slice.matchAll(sep)) {
    const cut = (m.index ?? 0) + m[0].length;
    if (cut > from && cut < slice.length) {
      out.push({ start: span.start + from, end: span.start + cut });
      from = cut;
    }
  }
  out.push({ start: span.start + from, end: span.end });
  return out;
}

/** Cuts a span with no usable separator into the longest pieces that fit, by binary search on length. */
function hardSplit(text: string, span: Span, r: Resolved): Span[] {
  const out: Span[] = [];
  let start = span.start;
  while (start < span.end) {
    let lo = start + 1;
    let hi = span.end;
    while (lo < hi) {
      const mid = Math.ceil((lo + hi) / 2);
      if (r.count(text.slice(start, mid)) <= r.maxTokens) lo = mid;
      else hi = mid - 1;
    }
    out.push({ start, end: lo });
    start = lo;
  }
  return out;
}

/** Breaks a span into atoms that each fit `maxTokens`, trying paragraph, line, sentence and word breaks in turn. */
function atomize(text: string, span: Span, r: Resolved, level = 0): Span[] {
  if (r.count(text.slice(span.start, span.end)) <= r.maxTokens) return [span];
  const sep = SEPARATORS[level];
  if (!sep) return hardSplit(text, span, r);
  const parts = splitSpan(text, span, sep);
  if (parts.length === 1) return atomize(text, span, r, level + 1);
  return parts.flatMap((p) => atomize(text, p, r, level + 1));
}

function trimSpan(text: string, span: Span): Span | null {
  let { start, end } = span;
  while (start < end && /\s/.test(text[start] as string)) start++;
  while (end > start && /\s/.test(text[end - 1] as string)) end--;
  return start < end ? { start, end } : null;
}

/** Greedily packs atoms into chunks of at most `maxTokens`, carrying up to `overlap` tokens of trailing atoms forward. */
function pack(text: string, atoms: Span[], r: Resolved, headings?: string[]): Omit<Chunk, 'index'>[] {
  const chunks: Omit<Chunk, 'index'>[] = [];
  const sizeOf = (a: Span[]) => r.count(text.slice((a[0] as Span).start, (a[a.length - 1] as Span).end));
  let current: Span[] = [];
  let fresh = 0; // atoms in `current` that are not overlap from the previous chunk

  const emit = () => {
    const first = current[0];
    const last = current[current.length - 1];
    if (!first || !last || fresh === 0) return;
    const span = trimSpan(text, { start: first.start, end: last.end });
    if (!span) return;
    const content = text.slice(span.start, span.end);
    chunks.push({
      content,
      tokens: r.count(content),
      start: span.start,
      end: span.end,
      ...(headings ? { headings } : {}),
    });
  };

  for (const atom of atoms) {
    if (current.length === 0 || sizeOf([...current, atom]) <= r.maxTokens) {
      current.push(atom);
      fresh++;
      continue;
    }
    emit();
    const carried: Span[] = [];
    for (let i = current.length - 1; i >= 0 && r.overlap > 0; i--) {
      const next = [current[i] as Span, ...carried];
      if (sizeOf(next) > r.overlap || sizeOf([...next, atom]) > r.maxTokens) break;
      carried.unshift(current[i] as Span);
    }
    current = [...carried, atom];
    fresh = 1;
  }
  emit();
  return chunks;
}

/**
 * Splits plain text into chunks of at most `maxTokens`, breaking at paragraphs, then lines, then sentences,
 * then words, and repeating about `overlap` tokens between neighbouring chunks.
 */
export function chunkText(text: string, options?: ChunkOptions): Chunk[] {
  const r = resolve(options);
  const atoms = atomize(text, { start: 0, end: text.length }, r);
  return pack(text, atoms, r).map((c, index) => ({ ...c, index }));
}

const FENCE = /^ {0,3}(`{3,}|~{3,})/;
const HEADING = /^ {0,3}(#{1,6})[ \t]+(.+?)[ \t#]*$/;

/**
 * Splits Markdown into chunks that never cross a heading. Each chunk records its heading path, fenced code blocks
 * are kept whole when they fit, and long sections fall back to the same breaks as {@link chunkText}.
 */
export function chunkMarkdown(markdown: string, options?: ChunkOptions): Chunk[] {
  const r = resolve(options);
  const sections: { span: Span; headings: string[]; blocks: Span[] }[] = [];
  const stack: { level: number; title: string }[] = [];
  let section = { span: { start: 0, end: 0 }, headings: [] as string[], blocks: [] as Span[] };
  let block: Span | null = null;
  let fence: string | null = null;

  const closeBlock = (end: number) => {
    if (block) section.blocks.push({ start: block.start, end });
    block = null;
  };

  let offset = 0;
  for (const line of markdown.split(/(?<=\n)/)) {
    const lineEnd = offset + line.length;
    const bare = line.replace(/\r?\n$/, '');
    const fenceMatch = FENCE.exec(bare);
    if (fence) {
      if (fenceMatch && fenceMatch[1]?.[0] === fence[0] && (fenceMatch[1]?.length ?? 0) >= fence.length) {
        fence = null;
        closeBlock(lineEnd);
      }
    } else if (fenceMatch) {
      closeBlock(offset);
      fence = fenceMatch[1] as string;
      block = { start: offset, end: lineEnd };
    } else {
      const heading = HEADING.exec(bare);
      if (heading) {
        closeBlock(offset);
        section.span.end = offset;
        sections.push(section);
        const level = (heading[1] as string).length;
        while (stack.length && (stack[stack.length - 1] as { level: number }).level >= level) stack.pop();
        stack.push({ level, title: (heading[2] as string).trim() });
        section = { span: { start: offset, end: offset }, headings: stack.map((h) => h.title), blocks: [] };
        block = { start: offset, end: lineEnd };
        closeBlock(lineEnd);
      } else if (bare.trim() === '') {
        closeBlock(offset);
      } else if (!block) {
        block = { start: offset, end: lineEnd };
      }
    }
    offset = lineEnd;
  }
  closeBlock(offset);
  section.span.end = offset;
  sections.push(section);

  const out: Chunk[] = [];
  for (const s of sections) {
    // A heading followed straight by a sub-heading has no body of its own; its title lives on in the sub-section's path.
    if (s.headings.length && s.blocks.length <= 1) continue;
    const atoms = s.blocks.flatMap((b) => atomize(markdown, b, r));
    for (const c of pack(markdown, atoms, r, s.headings.length ? s.headings : undefined)) {
      out.push({ ...c, index: out.length });
    }
  }
  return out;
}
