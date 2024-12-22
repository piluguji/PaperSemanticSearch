import requests
import xmltodict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import time
from tqdm import tqdm

def get_pubmed_papers(query, api_key):
    # Define the search query (e.g., "DNA sequencing")
    max_results = 40  # Number of results to fetch

    # Define the URL for Esearch API (search PubMed)
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={query}&retmax={max_results}&retmode=xml&api_key={api_key}"

    # Send request to PubMed
    response = requests.get(url)

    if response.status_code != 200:
        print(response.content)
        print("Could not get PMIDS")
        return []

    # Parse the XML response
    search_results = xmltodict.parse(response.content)

    # Extract list of PubMed IDs (PMIDs) from the response
    pmids = search_results['eSearchResult']['IdList']['Id']
    return pmids

# Define a function to fetch details of papers from PubMed

def parse_abstract(abstract):
    if type(abstract) == str: 
        return abstract
    elif type(abstract) == dict:
        return abstract['#text']
    elif type(abstract) == list: 
        abs = ""
        for sec in abstract: 
            abs += sec['#text']
        return abs
    else:
        return "Error"

def fetch_paper_details(pmids, api_key, query):
    pmid_str = ",".join(pmids)  # Join PMIDs into a comma-separated string
    
    # Define the URL for Efetch API (fetch details by PMIDs)
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid_str}&retmode=xml&api_key={api_key}"
    
    # Send request to PubMed
    response = requests.get(url)

    if response.status_code != 200:
        return pd.DataFrame()
    
    # Parse the XML response
    papers = xmltodict.parse(response.content)['PubmedArticleSet']['PubmedArticle']
    
    # Extract relevant information (title, abstract, authors, etc.)
    paper_details = []
    for idx, paper in enumerate(papers):
        try:
            # Extract title
            title = parse_abstract(paper['MedlineCitation']['Article'].get('ArticleTitle', 'No Title'))
            
            # Extract abstract
            abstract = paper['MedlineCitation']['Article'].get('Abstract', {}).get('AbstractText', '')
            parsed_abstract = parse_abstract(abstract)
            
            # Extract publication date
            pub_date = paper['MedlineCitation']['Article']['Journal']['JournalIssue'].get('PubDate', {})
            year = pub_date.get('Year', '')
            month = pub_date.get('Month', '')
            published = f"{year}-{month}".strip('-')
            
            # Extract PMID as ID
            pmid = paper['MedlineCitation']['PMID']['#text']
            
            # Standardize output to match arXiv function
            paper_details.append({
                "id": pmid,  # Matches arXiv's "id" column
                "title": title.strip(),
                "summary": parsed_abstract.strip(),
                "published": published,
                "sourced_from": "PubMed",  # Added to match arXiv's structure
                "query": query
            })
        except Exception as e:
            print(f"Error processing paper: {e} at {idx}")
    
    return pd.DataFrame(paper_details)

def fetch_arxiv_papers(query="all", max_results=50):
    """
    Fetches papers from arXiv based on a query.
    
    Args:
        query (str): Search term or category.
        max_results (int): Number of papers to retrieve.
        
    Returns:
        DataFrame: Contains titles, abstracts, and arXiv IDs.
    """
    base_url = "http://export.arxiv.org/api/query?"
    url = f"{base_url}search_query={query}&start=0&max_results={max_results}"

    response = requests.get(url)
    if response.status_code != 200:
        return pd.DataFrame()

    data = xmltodict.parse(response.content)
    if 'entry' not in data['feed']:
        return pd.DataFrame()
    entries = data['feed']['entry']
    
    # Parse entries into a DataFrame
    papers = []
    required_keys = ["id", "title", "summary", "published"]
    for entry in entries:
        if all(key in entry for key in required_keys):
            papers.append({
                "id": entry["id"],
                "title": entry["title"].strip(),
                "summary": entry["summary"].strip(),
                "published": entry["published"],
                "sourced_from": "arXiv",
                "query": query
            })

    return pd.DataFrame(papers)

load_dotenv()
api_key = os.getenv("pubmed_key")


# PubMed Searching
# keywords = [
#     "genome editing", "cancer immunotherapy", "telomere biology", "neurodegenerative diseases",
#     "CRISPR technology", "metabolic syndromes", "cardiovascular health", "microbiome analysis",
#     "epigenetic regulation", "stem cell therapy", "drug resistance mechanisms",
#     "infectious disease outbreaks", "mental health disorders", "vaccine development",
#     "precision medicine", "biomarker discovery", "gene therapy", "molecular diagnostics",
#     "diabetes management", "autoimmune diseases", "clinical trials", "oncogenic pathways",
#     "antibiotic resistance", "protein-protein interactions", "psychiatric genomics",
#     "regenerative medicine", "bioinformatics algorithms", "pharmacokinetics",
#     "aging research", "antiviral therapies", "public health epidemiology", "nutritional science",
#     "reproductive health", "immunological pathways", "sleep disorders", "nanomedicine",
#     "occupational health", "neurosurgical techniques", "biostatistics methods"
# ]
# df = pd.DataFrame()
# for kw in tqdm(keywords, desc="PubMed"): 
#     pmids = get_pubmed_papers(kw, api_key)
#     if len(pmids) == 0:
#         break
#     papers_data = fetch_paper_details(pmids, api_key, kw)
#     df = pd.concat([df, papers_data], ignore_index=True)
#     print(f'finished {kw}')
#     time.sleep(1)

# Arxiv Searching
keywords = [
    # AI - Core and General Topics
    "neural networks", "deep learning", "reinforcement learning", 
    "transformers", "few-shot learning", "zero-shot learning", 
    "multimodal learning", "contrastive learning", "causal inference in AI",
    "generative AI", "self-supervised learning", "meta-learning", 
    "unsupervised learning", "active learning", "representation learning", "genetic algorithms"

    # Natural Language Processing (NLP)
    "natural language processing", "transformer models", "language models",
    "large language models", "pretrained language models", "question answering", 
    "text summarization", "sentiment analysis", "semantic search", 
    "text generation", "entity recognition", "knowledge graphs", 
    "language model fine-tuning", "dialog systems", "prompt engineering",
    "retrieval-augmented generation", "low-resource NLP", 

    # Generative AI
    "generative adversarial networks", "diffusion models", 
    "text-to-image generation", "text-to-video generation", 
    "image synthesis", "video synthesis", "latent diffusion models", 
    "image-to-text models", "AI art generation", "generative AI ethics", 

    # Computer Vision (CV)
    "computer vision", "image segmentation", "object detection", 
    "semantic segmentation", "instance segmentation", "pose estimation", 
    "optical flow", "vision transformers", "multi-object tracking", 
    "video understanding", "image captioning", "3D object detection", 
    "3D reconstruction", "visual SLAM", "augmented reality", 
    "depth estimation", "scene understanding", "self-supervised vision", 

    # Diffusion Models and Generative Techniques
    "denoising diffusion models", "latent diffusion models", 
    "score-based generative models", "video diffusion models", 
    "conditional diffusion", "stochastic differential equations in AI", 

    # AI in Real-World Applications
    "AI for healthcare", "AI in robotics", "AI for education", 
    "AI for code generation", "AI for cybersecurity", "AI for accessibility", 
    "AI for personalized recommendations", "AI for autonomous vehicles", 
    "AI ethics", "fairness in AI", "bias in machine learning", 

    # Computer Science - General
    "theoretical computer science", "computational complexity", 
    "distributed systems", "parallel computing", "edge computing", 
    "neuromorphic computing", "quantum-inspired computing", 
    "automated theorem proving", "cloud computing", "cyber-physical systems",
    "virtualization technologies", "software engineering", 

    # Physics and Astrophysics
    "quantum computing", "quantum entanglement", "dark matter detection", 
    "gravitational waves", "exoplanet detection", "neutrino physics", 
    "quantum cryptography", "cosmic microwave background", 
    "galactic evolution", "theoretical cosmology", "dark energy", 
    "quantum mechanics foundations", "astroparticle physics", 
    "holography in physics",

    # Robotics and Engineering
    "autonomous robotics", "robotic process automation", "swarm robotics", 
    "robot perception", "human-robot interaction", "robotic manipulation", 
    "robotic exoskeletons", "distributed robotics", "underwater robotics", 
    "drone navigation", "autonomous vehicle systems", 
    "cyber-physical systems", "control theory", "robotics in agriculture",

    # Climate and Energy
    "renewable energy", "nuclear fusion plasma", "wind energy optimization", 
    "carbon capture", "solar energy harvesting", "geothermal energy", 
    "electric vehicle batteries", "smart grid technologies", 
    "climate modeling", "urban heat islands", "climate adaptation strategies", 
    "sustainable infrastructure",

    # Materials Science
    "solid-state batteries", "2D materials", "quantum materials", 
    "perovskite solar cells", "superconductors", "nanoelectronics", 
    "self-healing materials", "biodegradable plastics", "smart materials", 
    "artificial photosynthesis",

    # Cryptography and Blockchain
    "cryptographic protocols", "post-quantum cryptography", 
    "blockchain technology", "decentralized finance", 
    "secure multiparty computation", "privacy-preserving machine learning", 

    # Interdisciplinary and Emerging Fields
    "synthetic biology", "astrobiology", "cognitive neuroscience", 
    "functional connectomics", "computational archaeology", 
    "computational social science", "edge computing", "neuromorphic computing", 
    "digital twin technology", "quantum-inspired algorithms", 
    "human-computer interaction", "theoretical computer science", 
    "swarm intelligence", "computational chemistry",

    # Earth and Environmental Sciences
    "geospatial analysis", "oceanographic modeling", "forest carbon dynamics", 
    "permafrost thawing", "sustainable agriculture", "urban infrastructure resilience", 
    "volcanic eruption prediction"
]
df = pd.DataFrame()
for query in tqdm(keywords, desc="ArXiv Keywords"): 
    papers_df = fetch_arxiv_papers(query=query, max_results=40)
    if len(papers_df) == 0:
        print(f'ended on {query}')
        break
    df = pd.concat([df, papers_df], ignore_index=True)
    time.sleep(0.5)

df.to_json('Papers/papers_arxiv.json', orient='records', lines=True)