# Changelog

All notable changes to `@amjadiqbal/tanstack-ai-rag` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
