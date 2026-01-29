use anyhow::Result;

pub mod proto {
    tonic::include_proto!("embeddings");
}

use proto::embedding_service_client::EmbeddingServiceClient;
use proto::EmbedRequest;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<()> {
    let mut client = EmbeddingServiceClient::connect("http://[::1]:50051").await?;

    // Test 1: Single text
    println!("=== Test 1: Single text ===");
    let start = Instant::now();
    let req = tonic::Request::new(EmbedRequest {
        texts: vec!["Hello world".to_string()],
    });
    let resp = client.embed(req).await?;
    println!(
        "Latency: {:?}, dim: {}",
        start.elapsed(),
        resp.into_inner().embeddings[0].values.len()
    );

    // Test 2: Batch of 5
    println!("\n=== Test 2: Batch of 5 ===");
    let start = Instant::now();
    let req = tonic::Request::new(EmbedRequest {
        texts: vec![
            "The quick brown fox".to_string(),
            "jumps over the lazy dog".to_string(),
            "Rust is a systems programming language".to_string(),
            "Python is great for data science".to_string(),
            "JavaScript runs in the browser".to_string(),
        ],
    });
    let resp = client.embed(req).await?;
    println!(
        "Latency: {:?}, count: {}",
        start.elapsed(),
        resp.into_inner().embeddings.len()
    );

    // Test 3: Batch of 10
    println!("\n=== Test 3: Batch of 10 ===");
    let start = Instant::now();
    let texts: Vec<String> = (0..10)
        .map(|i| format!("This is test sentence number {}", i))
        .collect();
    let req = tonic::Request::new(EmbedRequest { texts });
    let resp = client.embed(req).await?;
    println!(
        "Latency: {:?}, count: {}",
        start.elapsed(),
        resp.into_inner().embeddings.len()
    );

    // Test 4: Sequential requests (10x single)
    println!("\n=== Test 4: 10 sequential single requests ===");
    let start = Instant::now();
    for i in 0..10 {
        let req = tonic::Request::new(EmbedRequest {
            texts: vec![format!("Sequential request {}", i)],
        });
        client.embed(req).await?;
    }
    println!(
        "Total latency: {:?}, avg: {:?}",
        start.elapsed(),
        start.elapsed() / 10
    );

    // Test 5: Empty input
    println!("\n=== Test 5: Empty input ===");
    let req = tonic::Request::new(EmbedRequest { texts: vec![] });
    let resp = client.embed(req).await?;
    println!("Empty result count: {}", resp.into_inner().embeddings.len());

    // Test 6: Long text
    println!("\n=== Test 6: Long text ===");
    let long_text = "Lorem ipsum dolor sit amet consectetur adipiscing elit. ".repeat(50);
    let start = Instant::now();
    let req = tonic::Request::new(EmbedRequest {
        texts: vec![long_text],
    });
    let resp = client.embed(req).await?;
    println!(
        "Latency: {:?}, dim: {}",
        start.elapsed(),
        resp.into_inner().embeddings[0].values.len()
    );

    println!("\n=== All tests passed! ===");
    Ok(())
}
