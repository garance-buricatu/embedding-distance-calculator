use std::env;

use clap::{Parser, ValueEnum};
use itertools::Itertools;
use rig::embeddings::EmbeddingModel;
use semanticsimilarity_rs::{
    cosine_similarity, dot_product_distance, euclidean_distance, manhattan_distance,
};

#[derive(Debug, Clone, ValueEnum)]
enum DistanceMetric {
    Cosine,
    L2,
    Dot,
    Manhattan,
}

impl ToString for DistanceMetric {
    fn to_string(&self) -> String {
        match self {
            DistanceMetric::Cosine => "cosine".to_string(),
            DistanceMetric::L2 => "l2".to_string(),
            DistanceMetric::Dot => "dot".to_string(),
            DistanceMetric::Manhattan => "manhattan".to_string(),
        }
    }
}

#[derive(Debug, Clone, ValueEnum)]
enum Provider {
    Openai,
    Cohere,
}

impl ToString for Provider {
    fn to_string(&self) -> String {
        match self {
            Provider::Openai => "openai".to_string(),
            Provider::Cohere => "cohere".to_string(),
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "Distance Calculator")]
#[command(version = "1.0")]
#[command(about = "Calculates distance between multiple vectors", long_about = None)]
struct Args {
    #[arg(short, long, value_delimiter = ',')]
    strings: Vec<String>,
    #[arg(short, long, default_value_t = Provider::Openai)]
    provider: Provider,
    #[arg(short, long)]
    embedding_model: String,
    #[arg(short, long, default_value_t = DistanceMetric::Cosine)]
    distance_metric: DistanceMetric,
}

/// Usage:
/// cargo build --release
/// ./target/release/embedding-distance-calculator -s 'i love bananas,good morning!,muffins,bananas,bananas' -p openai -e text-embedding-ada-002 -d l2
#[tokio::main]
async fn main() {
    // Parse command-line arguments
    let args = Args::parse();

    let documents = match args.provider {
        Provider::Openai => {
            let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
            let openai_client = rig::providers::openai::Client::new(&openai_api_key);

            let model = openai_client.embedding_model(&args.embedding_model);
            model.embed_documents(args.strings.into_iter()).await
        }
        Provider::Cohere => {
            let cohere_api_key = env::var("COHERE_API_HERE").expect("COHERE_API_HERE not set");
            let cohere_client = rig::providers::cohere::Client::new(&cohere_api_key);

            let model = cohere_client.embedding_model(&args.embedding_model, "search_document");
            model.embed_documents(args.strings.into_iter()).await
        }
    }
    .unwrap();

    let mut distances = documents
        .into_iter()
        .combinations(2)
        .map(|pair| {
            let first = pair.first().unwrap();
            let second = pair.last().unwrap();

            let distance = match args.distance_metric {
                DistanceMetric::Cosine => cosine_similarity(&first.vec, &second.vec, false),
                DistanceMetric::L2 => euclidean_distance(&first.vec, &second.vec),
                DistanceMetric::Dot => dot_product_distance(&first.vec, &second.vec),
                DistanceMetric::Manhattan => manhattan_distance(&first.vec, &second.vec),
            };

            (distance, first.document.clone(), second.document.clone())
        })
        .collect::<Vec<_>>();

    match args.distance_metric {
        DistanceMetric::Cosine | DistanceMetric::Dot => {
            // Larger distance = higher similarity
            distances.sort_by(|a, b| b.partial_cmp(a).unwrap());
        }
        DistanceMetric::L2 | DistanceMetric::Manhattan => {
            // Smaller distance = higher similarity
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        }
    };

    distances.iter().for_each(|(d, first, second)| {
        println!("{d}: \n* {first}\n* {second}");
    })
}
