# Changelog

All notable changes to `@amjadiqbal/tanstack-ai-rag` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-23

### Added

- `chunkText()` and `chunkMarkdown()`: token-aware chunking with overlap, a pluggable token counter (default: about 4 characters per token), character offsets, Markdown heading paths, and fenced code blocks kept whole when they fit.
- `memoryVectorStore()` and `pgvectorStore()` (Postgres + pgvector, cosine distance) behind a three-method `VectorStore` interface. `pgvectorStore` takes a `pg` Pool/Client, postgres.js, PGlite or a Drizzle database, and `migrate()` creates the extension, table, metadata GIN index and HNSW index.
- `createRag()`: `ingest()` (chunk, embed with TanStack AI `embed()` in batches, replace by document id), `retrieve()` (metadata filter, `minScore`, citation numbers), `remove()`, `formatContext()`, optional rerank through TanStack AI `rerank()`, and `tool()`, a server tool for `chat()` that returns numbered, cited results.
- `citationInstructions`, a system prompt for cited answers.
