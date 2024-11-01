# Embedding Distance Calculator
Rust script that calculates distances between the embeddings of strings.

### Input:
1. List of strings (string of comma-separated values)
2. Provider name (`openai` or `cohere`)
3. Embedding model name
4. Distance function (`l2`, `cosine`, `dot`, `manhattan`)   

### Output:
Distances between embeddings (created by defined provider/model) of each pair of strings based on the provided distance function. Pairs are sorted in order from closest to farthest.

## Usage
```bash
export OPENAI_API_KEY=""
cargo build --release
./target/release/distance-calculator -s 'i love bananas,good morning!,muffins,bananas,bananas' -p openai -e text-embedding-ada-002 -d l2
```