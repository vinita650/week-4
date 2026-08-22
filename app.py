import streamlit as st
from openai import OpenAI
import chromadb

THRESHOLD = 1.4

st.set_page_config(page_title='Chat with my document')
st.title("RAG Chat Application")
st.caption("Ask questions about Clean Code: A Handbook of Agile Software Craftsmanship by Robert C. Martin. Try: *What does Uncle Bob say about zombie code?*")

# Initialize clients
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
chroma_client = chromadb.PersistentClient(path='./chroma_db')
collection = chroma_client.get_collection(name='docs')

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Display conversation history
for message in st.session_state.history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        if 'caption' in message:
            st.caption(message['caption'])
        if 'sources' in message:
            with st.expander('Sources'):
                for source in message['sources']:
                    st.write(f"**Page {source['page']}** (distance: {source['distance']:.2f})")
                    st.write(source['text'][:300])

# Get user input
if question := st.chat_input("Ask a question about the document"):
    # Display user message
    with st.chat_message('user'):
        st.markdown(question)

    # Embed the question
    response = openai_client.embeddings.create(
        input=[question],
        model='text-embedding-3-small'
    )
    question_embedding = response.data[0].embedding

    # Query the collection
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )

    distances = results['distances'][0]

    # Check if closest distance exceeds threshold
    if distances[0] > THRESHOLD:
        reply = 'I could not find that in your document.'
        caption = f"closest match: {distances[0]:.2f} (threshold: {THRESHOLD})"

        # Add to history
        st.session_state.history.append({'role': 'user', 'content': question})
        st.session_state.history.append({
            'role': 'assistant',
            'content': reply,
            'caption': caption
        })

        with st.chat_message('assistant'):
            st.markdown(reply)
            st.caption(caption)
    else:
        # Build system message
        system_message = "Answer using ONLY the sources below, cite the page after each fact like [page 3], and if the answer is not in the sources reply exactly 'I could not find that in your document.'\n\n"

        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        sources = []

        for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
            page = metadata['page']
            system_message += f"Source {i+1} (page {page}):\n{doc}\n\n"
            sources.append({
                'page': page,
                'distance': distances[i],
                'text': doc
            })

        # Build messages for API
        messages = [{'role': 'system', 'content': system_message}]

        # Add last 6 messages from history
        for msg in st.session_state.history[-6:]:
            messages.append({
                'role': msg['role'],
                'content': msg['content']
            })

        # Add current question
        messages.append({'role': 'user', 'content': question})

        # Call the chat model
        response = openai_client.chat.completions.create(
            model='gpt-4o-mini',
            messages=messages
        )

        reply = response.choices[0].message.content

        # Add to history
        st.session_state.history.append({'role': 'user', 'content': question})
        st.session_state.history.append({
            'role': 'assistant',
            'content': reply,
            'sources': sources
        })

        with st.chat_message('assistant'):
            st.markdown(reply)
            with st.expander('Sources'):
                for source in sources:
                    st.write(f"**Page {source['page']}** (distance: {source['distance']:.2f})")
                    st.write(source['text'][:300])
