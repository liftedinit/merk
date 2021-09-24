pub use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error(transparent)]
    FailureError(#[from] failure::Error),
    #[error("Index OoB Error: {0}")]
    IndexOutOfBounds(String),
    #[error("Fetch Error: {0}")]
    FetchError(String),
    #[error("Proof Error: {0}")]
    ProofError(String),
    #[error("Key not found: {0}")]
    KeyNotFound(String),
    #[error("Batch Key Error: {0}")]
    BatchKey(String),
    #[error("Key Error: {0}")]
    KeyError(String),
    #[error("Path Error: {}")]
    PathError(String),
    #[error("Chunk Processing Error: {}")]
    ChunkProcessingError(String),
    #[error("Proof did not match expected hash\n\tExpected: {0}\n\tActual: {1}")]
    ProofHashMismatch(String, String),
    #[error("Query Error: {0}")]
    QueryError(String),
    #[error("Unknown Error")]
    Unknown,
}

pub type Result<T> = std::result::Result<T, Error>;
