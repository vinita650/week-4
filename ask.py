import tomllib
from openai import OpenAI
import chromadb

THRESHOLD = 1.4

# Read API key from secrets
with open('.streamlit/secrets.toml', 'rb') as f:
    secrets = tomllib.load(f)
    api_key = secrets['OPENAI_API_KEY']

# Create OpenAI client and Chroma client
client = OpenAI(api_key=api_key)
chroma_client = chromadb.PersistentClient(path='./chroma_db')
collection = chroma_client.get_collection(name='docs')

# Ask for a question
question = input()

# Embed the question
response = client.embeddings.create(
    input=[question],
    model='text-embedding-3-small'
)
question_embedding = response.data[0].embedding

# Query the collection
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)

# Print the three distances
distances = results['distances'][0]
print(f'{distances[0]:.2f}, {distances[1]:.2f}, {distances[2]:.2f}')

# Check if closest distance exceeds threshold
if distances[0] > THRESHOLD:
    print('I could not find that in your document.')
else:
    # Build system message
    system_message = "Answer using ONLY the sources below, cite the page after each fact like [page 3], and if the answer is not in the sources reply exactly 'I could not find that in your document.'\n\n"

    documents = results['documents'][0]
    metadatas = results['metadatas'][0]

    for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
        page = metadata['page']
        system_message += f"Source {i+1} (page {page}):\n{doc}\n\n"

    # Call the chat model
    response = client.chat.completions.create(
        model='gpt-4.5-mini',
        messages=[
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': question}
        ]
    )

    print(response.choices[0].message.content)
