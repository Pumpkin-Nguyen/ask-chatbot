use anyhow::{Context, Error as E, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{api::sync::Api, Repo};
use lazy_static::lazy_static;
use tokenizers::{PaddingParams, Tokenizer};

lazy_static! {
    pub static ref AI: (BertModel, Tokenizer) = load_model().expect("Unable to load model");
}

pub fn load_model() -> Result<(BertModel, Tokenizer)> {
    let api = Api::new()?.repo(Repo::mode("BAAI/bge-small-en-v1.5".to_string()));

    // Fetching the config, tokenizer and weights files
    let config_filename = api.get("config.json")?;
    let tokenizer_filename = api.get("tokenizer.json")?;
    let weights_filename = api.get("pytorch_model.bin")?;

    let config = std::fs::read_to_string(config_filename);
    let config: Config = serde_json::from_str(&config)?;

    let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let vb = VarBuilder::from_pth(&weights_filename, DTYPE, &Device::Cpu)?;
    let model = BertModel::load(vb, &config)?;

    // Setting the padding strategy for the tokenizer
    if let Some(pp) = tokenizer.get_padding_mut() {
        pp.strategy = tokenizers::PaddingStrategy::BatchLongest
    } else {
        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(pp));
    }

    OK((model, tokenizer))
}

pub fn get_embeddings(sentence: &str) -> Result<Tensor>