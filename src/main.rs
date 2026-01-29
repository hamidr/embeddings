use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::sync::Arc;
use tokenizers::Tokenizer;
use tonic::{transport::Server, Request, Response, Status};
use tonic_reflection::server::Builder as ReflectionBuilder;

pub mod proto {
    tonic::include_proto!("embeddings");
    pub const FILE_DESCRIPTOR_SET: &[u8] =
        tonic::include_file_descriptor_set!("embeddings_descriptor");
}

use proto::embedding_service_server::{EmbeddingService, EmbeddingServiceServer};
use proto::{EmbedRequest, EmbedResponse, Embedding};

struct EmbeddingModel {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl EmbeddingModel {
    fn load() -> Result<Self> {
        let device = Device::Cpu;
        println!("Using device: {:?}", device);

        let model_id = "sentence-transformers/all-MiniLM-L6-v2";
        println!("Loading model: {}", model_id);

        let api = Api::new()?;
        let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

        let config_path = repo
            .get("config.json")
            .context("Failed to get config.json")?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .context("Failed to get tokenizer.json")?;
        let weights_path = repo
            .get("model.safetensors")
            .context("Failed to get model.safetensors")?;

        let config: Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device)? };
        let model = BertModel::load(vb, &config)?;

        println!("Model loaded successfully");

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    fn encode(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

        let tokens = self
            .tokenizer
            .encode_batch(text_refs.clone(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;

        let token_ids: Vec<Vec<u32>> = tokens.iter().map(|t| t.get_ids().to_vec()).collect();
        let attention_mask: Vec<Vec<u32>> = tokens
            .iter()
            .map(|t| t.get_attention_mask().to_vec())
            .collect();
        let token_type_ids: Vec<Vec<u32>> =
            tokens.iter().map(|t| t.get_type_ids().to_vec()).collect();

        let max_len = token_ids.iter().map(|t| t.len()).max().unwrap_or(0);

        let token_ids: Vec<u32> = token_ids
            .iter()
            .flat_map(|t| {
                let mut padded = t.clone();
                padded.resize(max_len, 0);
                padded
            })
            .collect();

        let attention_mask: Vec<u32> = attention_mask
            .iter()
            .flat_map(|t| {
                let mut padded = t.clone();
                padded.resize(max_len, 0);
                padded
            })
            .collect();

        let token_type_ids: Vec<u32> = token_type_ids
            .iter()
            .flat_map(|t| {
                let mut padded = t.clone();
                padded.resize(max_len, 0);
                padded
            })
            .collect();

        let batch_size = texts.len();
        let input_ids = Tensor::from_vec(token_ids, (batch_size, max_len), &self.device)?;
        let attention_mask_tensor =
            Tensor::from_vec(attention_mask.clone(), (batch_size, max_len), &self.device)?;
        let token_type_ids = Tensor::from_vec(token_type_ids, (batch_size, max_len), &self.device)?;

        let embeddings =
            self.model
                .forward(&input_ids, &token_type_ids, Some(&attention_mask_tensor))?;

        // Mean pooling with attention mask
        let attention_mask_expanded = attention_mask_tensor.unsqueeze(2)?.to_dtype(DTYPE)?;

        let sum_embeddings = embeddings.broadcast_mul(&attention_mask_expanded)?.sum(1)?;

        let attention_mask_sum = attention_mask_tensor
            .to_dtype(DTYPE)?
            .sum_keepdim(1)?
            .clamp(1e-9, f64::MAX)?;

        let mean_embeddings = sum_embeddings.broadcast_div(&attention_mask_sum)?;

        // L2 normalize
        let l2_norm = mean_embeddings.sqr()?.sum_keepdim(1)?.sqrt()?;
        let normalized = mean_embeddings.broadcast_div(&l2_norm)?;

        // Convert to Vec<Vec<f32>>
        let result: Vec<Vec<f32>> = normalized.to_vec2()?;

        Ok(result)
    }
}

struct EmbeddingServiceImpl {
    model: Arc<EmbeddingModel>,
}

#[tonic::async_trait]
impl EmbeddingService for EmbeddingServiceImpl {
    async fn embed(
        &self,
        request: Request<EmbedRequest>,
    ) -> Result<Response<EmbedResponse>, Status> {
        let texts = request.into_inner().texts;

        let model = Arc::clone(&self.model);
        let result = tokio::task::spawn_blocking(move || model.encode(&texts))
            .await
            .map_err(|e| Status::internal(format!("Task join error: {}", e)))?
            .map_err(|e| Status::internal(format!("Encoding error: {}", e)))?;

        let embeddings = result
            .into_iter()
            .map(|values| Embedding { values })
            .collect();

        Ok(Response::new(EmbedResponse { embeddings }))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Initializing embedding service...");

    let model = EmbeddingModel::load()?;
    let model = Arc::new(model);

    let addr = "0.0.0.0:50051".parse()?;
    let service = EmbeddingServiceImpl { model };

    let reflection_service = ReflectionBuilder::configure()
        .register_encoded_file_descriptor_set(proto::FILE_DESCRIPTOR_SET)
        .build_v1()?;

    println!("Embedding service listening on {}", addr);

    Server::builder()
        .add_service(reflection_service)
        .add_service(EmbeddingServiceServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
