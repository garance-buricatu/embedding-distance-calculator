use std::{env, fmt::Display, fs::File, io::BufReader};

use clap::{Parser, ValueEnum};
use itertools::Itertools;
use pretty_table::print_table;
use rig::embeddings::EmbeddingModel;
use semanticsimilarity_rs::{
    cosine_similarity, dot_product_distance, euclidean_distance, manhattan_distance,
};

const EMPTY: &str = "-";

#[derive(Debug, Clone, ValueEnum)]
enum DistanceMetric {
    Cosine,
    L2,
    Dot,
    Manhattan,
}

impl Display for DistanceMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DistanceMetric::Cosine => write!(f, "cosine"),
            DistanceMetric::L2 => write!(f, "l2"),
            DistanceMetric::Dot => write!(f, "dot"),
            DistanceMetric::Manhattan => write!(f, "manhattan"),
        }
    }
}

#[derive(Debug, Clone, ValueEnum)]
enum Provider {
    Openai,
    Cohere,
}

impl Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Provider::Openai => write!(f, "openai"),
            Provider::Cohere => write!(f, "cohere"),
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "Distance Calculator")]
#[command(version = "1.0")]
#[command(about = "Calculates distance between multiple vectors", long_about = None)]
struct Args {
    #[arg(short)]
    input_file: String,
    #[arg(short, long, default_value_t = Provider::Openai)]
    provider: Provider,
    #[arg(short, long)]
    embedding_model: String,
    #[arg(short, long, default_value_t = DistanceMetric::Cosine)]
    distance_metric: DistanceMetric,
}

impl Args {
    fn input_strings(&self) -> Vec<String> {
        let file = File::open(self.input_file.clone()).unwrap();
        let reader = BufReader::new(file);

        serde_json::from_reader(reader).unwrap()
    }
}

#[tokio::main]
async fn main() {
    // Parse command-line arguments
    let args = Args::parse();

    let input_strings = args.input_strings();

    let documents = match args.provider {
        Provider::Openai => {
            let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
            let openai_client = rig::providers::openai::Client::new(&openai_api_key);

            let model = openai_client.embedding_model(&args.embedding_model);
            model
                .embed_documents(input_strings.clone().into_iter())
                .await
        }
        Provider::Cohere => {
            let cohere_api_key = env::var("COHERE_API_HERE").expect("COHERE_API_HERE not set");
            let cohere_client = rig::providers::cohere::Client::new(&cohere_api_key);

            let model = cohere_client.embedding_model(&args.embedding_model, "search_document");
            model
                .embed_documents(input_strings.clone().into_iter())
                .await
        }
    }
    .unwrap();

    let mut dataframe = DataFrame::set_headers(input_strings.clone());

    documents
        .into_iter()
        .enumerate()
        .combinations_with_replacement(2)
        .map(|mut combination| {
            combination.sort_by(|(i, _), (j, _)| Ord::cmp(i, j)); // Sort the pair to make (1, 2) and (2, 1) identical
            combination
        })
        .unique_by(|pair| {
            let (i, _) = pair.first().unwrap();
            let (j, _) = pair.last().unwrap();

            (*i, *j)
        })
        .for_each(|pair| {
            let (i, first) = pair.first().unwrap();
            let (j, second) = pair.last().unwrap();

            let distance = match args.distance_metric {
                DistanceMetric::Cosine => cosine_similarity(&first.vec, &second.vec, false),
                DistanceMetric::L2 => euclidean_distance(&first.vec, &second.vec),
                DistanceMetric::Dot => dot_product_distance(&first.vec, &second.vec),
                DistanceMetric::Manhattan => manhattan_distance(&first.vec, &second.vec),
            };

            dataframe.add_row_header(i, &second.document);
            dataframe.add_row_distances(i, j, distance);
        });

    print_table!(dataframe.as_dataframe());
}

struct DataFrame {
    headers: Vec<String>,
    data: Vec<Vec<String>>,
}

impl DataFrame {
    fn set_headers(input_strings: Vec<String>) -> Self {
        let mut headers = vec!["".to_string()];
        headers.extend(input_strings.clone());

        DataFrame {
            headers: headers
                .iter()
                .enumerate()
                .map(|(i, string)| {
                    if i == 0 {
                        "".to_string()
                    } else {
                        format_header(i-1, string)
                    }
                })
                .collect::<Vec<_>>(),
            data: vec![vec![]; input_strings.len()],
        }
    }

    fn get_row(&mut self, i: &usize) -> &mut Vec<String> {
        self.data.get_mut(*i).unwrap()
    }

    fn add_row_header(&mut self, i: &usize, string: &str) {
        let row_i = self.get_row(i);

        if row_i.is_empty() {
            row_i.push(format_header(*i, string));
        }
    }

    fn add_row_distances(&mut self, i: &usize, j: &usize, distance: f64) {
        let row_i = self.get_row(i);

        if row_i.len() == *j + 1 {
            row_i.push(distance.to_string());
        } else {
            while row_i.len() < *j + 1 {
                row_i.push(EMPTY.to_string());
            }
            row_i.push(distance.to_string());
        }
    }

    fn as_dataframe(&self) -> Vec<Vec<String>> {
        let mut data = vec![self.headers.clone()];
        data.extend(self.data.clone());
        data
    }
}

fn format_header(i: usize, string: &str) -> String {
    format!(
        "{}: {}",
        i,
        match string.len() {
            0..=10 => string.to_string(),
            _ => if let Some(index) = string.find('.') {
                format!("{}...", &string[..index])
            } else {
                format!("{}...", &string[..10])
            }
        }
    )
}
