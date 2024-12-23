import torch
import json
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

class TripletDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=512):
        """
        Args:
            json_path (str): Path to JSON file containing triplets
            tokenizer: HuggingFace tokenizer
            max_length (int): Maximum sequence length for tokenization
        """
        # Load JSON data
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get single triplet example
        item = self.data[idx]
        
        # Tokenize the query
        query_encoding = self.tokenizer(
            item['question'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Convert embeddings to tensors
        # Assuming embeddings in JSON are stored as lists or arrays
        positive_emb = torch.tensor(item['positive'], dtype=torch.float)
        negative_emb = torch.tensor(item['negative'], dtype=torch.float)
        
        return {
            'query_ids': query_encoding['input_ids'].squeeze(),
            'query_mask': query_encoding['attention_mask'].squeeze(),
            'positive_embedding': positive_emb,
            'negative_embedding': negative_emb
        }
    
class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.cosine = nn.CosineSimilarity(dim=1)
        
    def forward(self, anchor, positive, negative):
        # Compute cosine similarities
        similarity_positive = self.cosine(anchor, positive)  # Higher means more similar
        similarity_negative = self.cosine(anchor, negative)  # Higher means more similar
        
        # Loss = max(0, margin - (pos_sim - neg_sim))
        # We want positive similarity to be higher than negative similarity by at least the margin
        losses = torch.relu(self.margin - (similarity_positive - similarity_negative))
        
        return losses.mean()
    
class TripletBERTModel(nn.Module):
    def __init__(self, model_name='allenai/scibert_scivocab_uncased', freeze_layers=True):
        super(TripletBERTModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        
        if freeze_layers:
            # Freeze all layers except the last transformer layer
            for name, param in self.bert.named_parameters():
                if "encoder.layer.11" not in name:  # 11 is the last layer (0-11)
                    param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]