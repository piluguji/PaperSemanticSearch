from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer
import pandas as pd
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from model import TripletBERTModel

# Load the tokenizer used during training
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

import ollama

# Define the system prompt that sets the behavior
system_prompt = """
You are an AI assistant that answers questions with general knowledge but also with supplemented knowledge from given research papers. Your primary role is to analyze the provided abstracts/summaries of research papers and form a thoughtful, detailed response to the user's query. Your response must integrate information from the papers when relevant and cite them explicitly.

### Guidelines for Your Response:

1. Always use information from the provided summaries to support your response. Avoid making unsupported generalizations.
2. If a specific statement or idea is derived from a paper, cite it by its number (e.g., Paper 1). 
3. When combining insights or evidence from multiple papers, cite all relevant papers (e.g., Paper 1, 2).
4. If the provided papers do not directly answer the query, state this explicitly but attempt to provide a general answer based on your knowledge.

### Format for the User's Prompt:

Prompt: 
Query: <question from user>
1: <Paper 1 Summary>
2: <Paper 2 Summary>
3: <Paper 3 Summary>

### Format for Your Response:

Response: 
Begin with a concise summary addressing the query. Use citations (e.g., Paper 1) to indicate the source of specific information. Clearly differentiate between information sourced from papers and general knowledge. Where multiple papers are relevant, combine insights with proper citations (e.g., Paper 1, 2).
"""

def generate_questions_for_abstract(query, summaries):
    # Number the summaries and format them
    numbered_summaries = "\n".join(
        f"{i + 1}: {summary}" for i, summary in enumerate(summaries)
    )
    
    # Construct the user content with the query and numbered summaries
    user_content = f"Query: {query}\n{numbered_summaries}"

    # Send the formatted prompt to the model
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    full_response = response["message"]["content"]
    clean_response = full_response.split("References:")[0].strip()
    
    # Remove the "Response:" label if present
    if clean_response.lower().startswith("response:"):
        clean_response = clean_response[len("Response:"):].strip()
    
    return clean_response 

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

app = Flask(__name__)




@app.route("/")
def chat():
    return render_template("chat.html")

@app.route("/send_message", methods=["POST"])
def send_message():
    query = request.json.get("message")


    # Find the most similar paper
    idx, best_match_similarity = find_most_similar_paper(query, model, tokenizer, paper_embeddings, device)
    
    # Prepare the output for the query
    output = []
    summaries = [papers.iloc[idx]['summary']]
    
    # Find and display similar papers
    similar_indices = get_similar_papers(paper_embeddings[idx], paper_embeddings)
    for i in similar_indices:
        summaries.append(papers.iloc[i]['summary'])
    
    llm_response = generate_questions_for_abstract(query, summaries)
    output.append(f"{llm_response}\n")

    output.append(f"Best match: {papers.iloc[idx]['title']}\n")
    output.append("Similar Papers:\n")
    for i in similar_indices:
        output.append(f"- {papers.iloc[i]['title']}\n")

    bot_response = ''.join(output)

    return jsonify({"bot_response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
