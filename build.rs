fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .file_descriptor_set_path(
            std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap())
                .join("embeddings_descriptor.bin"),
        )
        .compile_protos(&["proto/embeddings.proto"], &["proto"])?;
    Ok(())
}
