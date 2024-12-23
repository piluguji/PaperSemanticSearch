from transformers import AutoTokenizer
import pandas as pd
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from model import TripletBERTModel

# Load the tokenizer used during training
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

# Function to generate the embedding for a given text
def generate_embedding(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        embedding = model(input_ids, attention_mask)
    return embedding

# Function to find the most similar paper
def find_most_similar_paper(query, model, tokenizer, paper_embeddings, device):
    query_embedding = generate_embedding(query, model, tokenizer, device)
    similarities = F.cosine_similarity(query_embedding, paper_embeddings, dim=1)
    best_match_idx = torch.argmax(similarities).item()
    best_match_similarity = similarities[best_match_idx].item()
    return best_match_idx, best_match_similarity

# Function to get the top-k similar papers (excluding the query paper itself)
def get_similar_papers(paper_embedding, embeddings, top_k=3):
    similarities = F.cosine_similarity(paper_embedding, embeddings)
    top_indices = torch.topk(similarities, top_k + 1).indices.tolist()
    return [idx for idx in top_indices if similarities[idx] < 0.999][:top_k]  # Exclude the identical paper

# Load the model architecture and weights
model = TripletBERTModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load the saved weights
model.load_state_dict(torch.load('Models/semantic_search.pt', map_location=device))
model.eval()

# Load precomputed embeddings and metadata
paper_embeddings = torch.stack(torch.load('Data/embeddings.pt')).to(device).squeeze(1)
papers = pd.read_json('Papers/all_papers.json', orient='records', lines=True)

# Open a file to write the output
output_file = "query_results.txt"

# Continuous querying loop
while True:
    # Get the query from the user
    query = input("Enter your query (or type 'exit' to quit): ").strip()
    if query.lower() == 'exit':
        print("Exiting the script. Goodbye!")
        break
    
    # Find the most similar paper
    idx, best_match_similarity = find_most_similar_paper(query, model, tokenizer, paper_embeddings, device)
    
    # Prepare the output for the query
    output = []
    output.append(f"Query: {query}\n")
    output.append(f"Best match: {papers.iloc[idx]['summary']}\n")
    output.append(f"Similarity: {best_match_similarity:.4f}\n")
    
    # Find and display similar papers
    similar_indices = get_similar_papers(paper_embeddings[idx], paper_embeddings)
    output.append("Similar papers:\n")
    for i in similar_indices:
        output.append(f"- {papers.iloc[i]['title']}\n")
    
    output.append("-" * 50 + "\n")
    
    # Write to file
    with open(output_file, "a") as f:
        f.writelines(output)
    
    # Print to console as well (optional)
    print("".join(output))
