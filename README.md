# rust-embeddings

A self-hosted gRPC service for generating text embeddings.

## The Problem

Every time you store or search text in a vector database, you need embeddings. Most solutions require calling external APIs like OpenAI, Cohere, or Voyage—paying per request for something that doesn't need an expensive LLM.

Embeddings are just numerical representations of text. They don't require reasoning, creativity, or the capabilities you're paying for with LLM APIs. Yet many teams spend significant budget on these simple transformations.

## The Solution

Run this service on your local machine or internal server. When your application needs to convert text to embeddings:

1. Send the text to this service via gRPC
2. Get back a 384-dimensional vector
3. Store it in your vector database (Pinecone, Qdrant, Weaviate, etc.)

No external API calls. No per-request costs. No data leaving your network.

## Quick Start

```bash
cargo build --release
cargo run --release
```

The service listens on `[::1]:50051`.

## Usage

```bash
grpcurl -plaintext -d '{"texts": ["your document text here"]}' \
  '[::1]:50051' embeddings.EmbeddingService/Embed
```

Response:

```json
{
  "embeddings": [{ "values": [0.12, -0.34, 0.56, ...] }]
}
```

## When to Use This

- Indexing documents into a vector database
- Building semantic search without API costs
- RAG pipelines where you control the embedding step
- Any workflow where text → vector conversion happens frequently

## License

MIT
