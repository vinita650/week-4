import tomllib
import re
from pathlib import Path
from openai import OpenAI
from pypdf import PdfReader
import chromadb

# Read API key from secrets
with open('.streamlit/secrets.toml', 'rb') as f:
    secrets = tomllib.load(f)
    api_key = secrets['OPENAI_API_KEY']

# Create OpenAI client
client = OpenAI(api_key=api_key)

# Read PDF
pdf_reader = PdfReader('handbook1.pdf')
num_pages = len(pdf_reader.pages)

# Process pages and create chunks
chunks = []
chunk_id = 0

for page_num, page in enumerate(pdf_reader.pages):
    text = page.extract_text()
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Split into chunks with overlap
    chunk_size = 1000
    overlap = 150
    step = chunk_size - overlap

    # Create chunks from this page
    for i in range(0, len(text), step):
        chunk_text = text[i:i + chunk_size]
        chunks.append({
            'text': chunk_text,
            'page': page_num + 1,
            'id': f'chunk-{chunk_id}'
        })
        chunk_id += 1

        # Stop if we've covered the end of the text
        if i + chunk_size >= len(text):
            break

print(f'{num_pages} pages, {len(chunks)} chunks')

# Embed chunks in batches of 100
embeddings = []
for i in range(0, len(chunks), 100):
    batch = chunks[i:i + 100]
    batch_texts = [chunk['text'] for chunk in batch]

    response = client.embeddings.create(
        input=batch_texts,
        model='text-embedding-3-small'
    )

    for embedding in response.data:
        embeddings.append(embedding.embedding)

# Store in chromadb
client_chroma = chromadb.PersistentClient(path='./chroma_db')
collection = client_chroma.get_or_create_collection(name='docs')

# Add to collection
documents = [chunk['text'] for chunk in chunks]
ids = [chunk['id'] for chunk in chunks]
metadatas = [{'page': chunk['page']} for chunk in chunks]

collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=documents,
    metadatas=metadatas
)

print('stored', collection.count(), 'chunks')
