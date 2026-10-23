# Changelog

All notable changes to `@amjadiqbal/tanstack-ai-rag` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.4] - 2026-10-23

### Fixed

- Chunk overlap now works with paragraphs longer than `overlap`. Before, only whole trailing pieces (paragraphs, lines, sentences) were carried over, so prose with paragraphs over about 64 tokens got no overlap at all. Each chunk now starts with the longest end of the previous chunk that fits in `overlap` tokens, from a word boundary (any character when there are no spaces).
- Re-ingesting a document into `pgvectorStore` is atomic: delete and insert run in one transaction, so a failed insert keeps the old chunks. `memoryVectorStore` swaps in one synchronous step.
- `rag.tool()` citation numbers continue across searches within one `chat()` run, so two searches in one answer no longer both produce `[1]`. A chunk returned again keeps its number. `citations: 'call'` restores per-call numbering.

### Added

- `VectorStore.replaceDocuments()` (optional), used by `ingest()` when present.
- `id` (the chunk id) on each result in the tool's output.

### Documentation

- README: how filters behave per store (`===` in memory, jsonb containment in pgvector, bound parameters), the overlap rule, citation numbering, the heading line in chunk text, heading-only sections, re-embedding unchanged content, what the reranker sees, and which requests get the abort signal.

## [0.1.3] - 2026-10-23

Corrects the release history. The 0.1.1 changelog wrongly said 0.1.0 was not published, and the v0.1.2 git tag (never published to npm) wrongly said 0.1.1 was not published. Both 0.1.0 and 0.1.1 are on npm. No code changes.

## [0.1.1] - 2026-10-23

No code changes since 0.1.0.

### Documentation

- README: an "Errors and edge cases" table (every row backed by a test), links to TanStack AI's `embed()`, `rerank()` and server-tools docs, and a note that the package imports only public `@tanstack/ai` APIs.

## [0.1.0] - 2026-10-23

### Added

- `chunkText()` and `chunkMarkdown()`: token-aware chunking with overlap, a pluggable token counter (default: about 4 characters per token), character offsets, Markdown heading paths, and fenced code blocks kept whole when they fit.
- `memoryVectorStore()` and `pgvectorStore()` (Postgres + pgvector, cosine distance) behind a three-method `VectorStore` interface. `pgvectorStore` takes a `pg` Pool/Client, postgres.js, PGlite or a Drizzle database, and `migrate()` creates the extension, table, metadata GIN index and HNSW index.
- `createRag()`: `ingest()` (chunk, embed with TanStack AI `embed()` in batches, replace by document id), `retrieve()` (metadata filter, `minScore`, citation numbers), `remove()`, `formatContext()`, optional rerank through TanStack AI `rerank()`, and `tool()`, a server tool for `chat()` that returns numbered, cited results.
- `citationInstructions`, a system prompt for cited answers.
