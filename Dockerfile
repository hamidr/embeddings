# Build stage
FROM rust:1.83-bookworm AS builder

# Install protobuf compiler for gRPC
RUN apt-get update && apt-get install -y protobuf-compiler && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy manifests first for better caching
COPY Cargo.toml Cargo.lock ./
COPY build.rs ./
COPY proto ./proto

# Create dummy src to build dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs

# Build dependencies (this layer will be cached)
RUN cargo build --release && rm -rf src

# Copy actual source code
COPY src ./src

# Touch main.rs to ensure rebuild
RUN touch src/main.rs

# Build the application
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the binary from builder
COPY --from=builder /app/target/release/embeddings /app/embeddings

# Create directory for HuggingFace cache
RUN mkdir -p /root/.cache/huggingface

# Expose gRPC port
EXPOSE 50051

# Run the service
CMD ["/app/embeddings"]
