from idlelib import query

from ollama import Client
import requests
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import torch
import numpy as np

URL_login = # url login if you need it

API_URL = # API url


token = # your token

headers = {
    "Authorization": f"Bearer {token}"
}

response = requests.get(API_URL, headers=headers)

#print(response.status_code)
#print(response.text)
data = response.json()
#print(response.json())
#print(data.keys())
print(data['features'][53])

documents = []

for feature in data['features']:
    properties = feature['properties']
    city = properties.get('lib_zone', 'Unknown')
    date = properties.get('date_dif', 'Unknown')
    no2 = properties.get('code_no2', 'N/A')
    o3 = properties.get('code_o3', 'N/A')
    pm10 = properties.get('code_pm10', 'N/A')
    pm25 = properties.get('code_pm25', 'N/A')
    quality = properties.get('code_qual', 'N/A')
    so2 = properties.get('code_so2', 'N/A')
    #quality = properties.get('lib_qual', 'Unknown')

    text = f"On {date} in {city}, the air quality was : PM2.5 = {pm25}, PM10 = {pm10}, NO2 = {no2}, O3 = {o3} and So2 = {so2}."
    documents.append(text)

print(documents[0])

# Generate embeddings with all-MiniLM-L6-v2

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = embed_model.encode(documents)
embeddings = np.array(embeddings).astype("float32")
print(embeddings.shape)
# Initialize Faiss Index

dimension = embeddings.shape[1]  # Dimension of the embedding vector
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)  # Add embeddings to the index


def search_(query, top_k=1):
    query_embedding = embed_model.encode([query])[0]
    query_embedding = np.array(query_embedding).astype("float32")
    print(query_embedding.shape)
    distances, indices = index.search(query_embedding.reshape(1, -1), top_k)
    return [documents[i] for i in indices[0]]

from ollama import Client

client = Client(host='http://localhost:11434')

def generate_answer(context, query):
    prompt = f"""
    You are an AI assistant that answers questions about air quality.
    Context: {context}
    Question: {query}
    Answer:
    """
    response = client.generate(
        model='deepseek-r1:8b',
        prompt=prompt,
        stream=False
    )
    return response['response']

Question = "What was the air quality in Lyon on 2025-08-30 ?"
context = search_(Question)
print(context)
answer = generate_answer(context, Question)

print(answer)
