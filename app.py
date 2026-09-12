"""AI-powered document Q&A assistant using retrieval-augmented generation."""
import streamlit as st
from openai import OpenAI
import chromadb
from pypdf import PdfReader
import os
import uuid

UPLOADS_DIR = './uploads'

st.set_page_config(page_title='Document Q&A Assistant')
st.title("Document Q&A Assistant")
st.caption("Ask questions about your documents. Start asking anything and get answers with source citations.")

with st.expander("❓ How this works"):
	st.markdown("""
	This app uses **Retrieval-Augmented Generation (RAG)** to answer your questions accurately:

	1. **Read**: When you upload a document, the app reads every page and breaks it into passages.
	2. **Find**: When you ask a question, the app finds the passages most similar to your question.
	3. **Answer**: The app reads only those passages and answers your question based on what's there — nothing made up.

	Think of it like a smart librarian: instead of making up answers, it searches the book and reads the relevant sections aloud to you. You always see which pages it used.
	""")


os.makedirs(UPLOADS_DIR, exist_ok=True)

# Initialize clients
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Use in-memory database to avoid Streamlit Cloud filesystem issues
if 'chroma_client' not in st.session_state:
	st.session_state.chroma_client = chromadb.EphemeralClient()
	st.session_state.collection = st.session_state.chroma_client.create_collection(name='docs')

chroma_client = st.session_state.chroma_client
collection = st.session_state.collection

def extract_pdf_text(pdf_file):
	pdf_reader = PdfReader(pdf_file)
	documents = []
	for page_num, page in enumerate(pdf_reader.pages):
		text = page.extract_text().strip()
		if text and len(text) > 50:
			documents.append({'text': text, 'page': page_num + 1})
	return documents

def chunk_text(text, max_length=8000):
	chunks = []
	if len(text) <= max_length:
		return [text]
	words = text.split()
	current_chunk = []
	current_length = 0
	for word in words:
		word_length = len(word) + 1
		if current_length + word_length > max_length:
			if current_chunk:
				chunks.append(' '.join(current_chunk))
			current_chunk = [word]
			current_length = word_length
		else:
			current_chunk.append(word)
			current_length += word_length
	if current_chunk:
		chunks.append(' '.join(current_chunk))
	return chunks

def add_documents_to_collection(documents, doc_name):
	doc_id = str(uuid.uuid4())
	added_count = 0
	for doc in documents:
		chunks = chunk_text(doc['text'])
		for chunk_idx, chunk in enumerate(chunks):
			if not chunk.strip():
				continue
			try:
				response = openai_client.embeddings.create(
					input=[chunk],
					model='text-embedding-3-small'
				)
				embedding = response.data[0].embedding
				collection.add(
					documents=[chunk],
					embeddings=[embedding],
					metadatas=[{'page': doc['page'], 'source': doc_name}],
					ids=[f"{doc_id}_{doc['page']}_{chunk_idx}"]
				)
				added_count += 1
			except Exception as e:
				st.error(f"Error processing page {doc['page']}: {str(e)}")
				continue
	return added_count

# Initialize session state for threshold and loaded docs
if 'threshold' not in st.session_state:
	st.session_state.threshold = 1.4
if 'loaded_docs' not in st.session_state:
	st.session_state.loaded_docs = []

# File uploader and controls in sidebar
with st.sidebar:
	st.header("📄 Upload Document")
	uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
	if uploaded_file is not None:
		if st.button("Process PDF"):
			with st.spinner("Processing PDF..."):
				try:
					documents = extract_pdf_text(uploaded_file)
					if not documents:
						st.error("❌ No readable text found in PDF. Try a different file.")
					else:
						added_count = add_documents_to_collection(documents, uploaded_file.name)
						if uploaded_file.name not in st.session_state.loaded_docs:
							st.session_state.loaded_docs.append(uploaded_file.name)
						st.success(f"✅ {uploaded_file.name} loaded! ({added_count} passages indexed)")
				except Exception as e:
					st.error(f"❌ Error processing PDF: {str(e)}")

	st.divider()
	st.subheader("⚙️ Settings")

	st.session_state.threshold = st.slider(
		"Match Threshold",
		min_value=0.5,
		max_value=2.5,
		value=st.session_state.threshold,
		step=0.1,
		help="Lower = more lenient matches, Higher = stricter matches. If no passage meets this threshold, the app says it can't find the answer."
	)

	if st.session_state.loaded_docs:
		st.write("**📚 Loaded Documents:**")
		for doc in st.session_state.loaded_docs:
			st.caption(f"✓ {doc}")

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
    if distances[0] > st.session_state.threshold:
        reply = 'I could not find that in your document.'
        caption = f"closest match: {distances[0]:.2f} (threshold: {st.session_state.threshold})"

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

        # Add only the current question (fixes follow-up trap by ensuring
        # answers come strictly from retrieved sources, not conversation context)
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
