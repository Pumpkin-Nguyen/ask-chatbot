use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "Tera")]
#[command(about = "Tera is AI assistant which is tailored just for you", long_about = None)]
pub struct Cli {
    #[command(Subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    Ask {
        query: String,
    },
    Remember {
        content: String,
    }
}