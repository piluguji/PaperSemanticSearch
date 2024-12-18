import requests
import xmltodict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import time

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

def fetch_paper_details(pmids, api_key):
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
                "query": "telomeres" 
            })
        except Exception as e:
            print(f"Error processing paper: {e} at {idx}")
    
    return pd.DataFrame(paper_details)


load_dotenv()
api_key = os.getenv("pubmed_key")
keywords = [
    "genome editing", "cancer immunotherapy", "telomere biology", "neurodegenerative diseases",
    "CRISPR technology", "metabolic syndromes", "cardiovascular health", "microbiome analysis",
    "epigenetic regulation", "stem cell therapy", "drug resistance mechanisms",
    "infectious disease outbreaks", "mental health disorders", "vaccine development",
    "precision medicine", "biomarker discovery", "gene therapy", "molecular diagnostics",
    "diabetes management", "autoimmune diseases", "clinical trials", "oncogenic pathways",
    "antibiotic resistance", "protein-protein interactions", "psychiatric genomics",
    "regenerative medicine", "bioinformatics algorithms", "pharmacokinetics",
    "aging research", "antiviral therapies", "public health epidemiology", "nutritional science",
    "reproductive health", "immunological pathways", "sleep disorders", "nanomedicine",
    "occupational health", "neurosurgical techniques", "biostatistics methods"
]

df = pd.DataFrame()
for kw in keywords: 
    pmids = get_pubmed_papers(kw, api_key)
    if len(pmids) == 0:
        break
    papers_data = fetch_paper_details(pmids, api_key)
    df = pd.concat([df, papers_data], ignore_index=True)
    print(f'finished {kw}')
    time.sleep(1)

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
    for entry in entries:
        papers.append({
            "id": entry["id"],
            "title": entry["title"].strip(),
            "summary": entry["summary"].strip(),
            "published": entry["published"],
            "sourced_from": "arXiv",
            "query": query
        })

    return pd.DataFrame(papers)

# Example Usage

keywords = [
    "neural networks", "reinforcement learning", "quantum computing", "galactic formation",
    "genome sequencing", "protein folding", "machine vision", "particle accelerators",
    "dark matter detection", "economic modeling", "climate change impacts", "renewable energy",
    "natural language processing", "cryptographic protocols", "functional genomics",
    "telomere dynamics", "gravitational waves", "solar energy harvesting", "autonomous robotics",
    "blockchain technology", "synthetic biology", "exoplanet detection", "fusion reactors",
    "semantic segmentation", "theoretical computer science", "cognitive neuroscience",
    "statistical mechanics", "bioinformatics pipelines", "deep reinforcement learning",
    "AI ethics", "hydropower modeling", "nuclear fusion plasma", "social network analysis",
    "human-computer interaction", "computational chemistry", "astrobiology", "drug discovery"
]
df = pd.DataFrame()

for query in keywords: 
    papers_df = fetch_arxiv_papers(query=query, max_results=40)
    if len(papers_df) == 0:
        print(f'ended on {query}')
        break
    df = pd.concat([df, papers_df], ignore_index=True)
    time.sleep(1)

df.to_json('papers.json', orient='records', lines=True)