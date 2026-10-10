import os
import tempfile
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np

st.set_page_config(page_title="RAG Document Assistant", page_icon="📚", layout="wide")

st.title("📚 RAG Document Assistant")
st.write("Upload a PDF or text file, then ask questions. Answers are based on retrieved passages.")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def read_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        pages = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"text": text, "source": uploaded_file.name, "page": page_num})
        return pages
    raw = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    return [{"text": raw, "source": uploaded_file.name, "page": None}] if raw.strip() else []

def split_text(text, chunk_size=700, overlap=120):
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks

def build_chunks(uploaded_files):
    chunks = []
    for uploaded_file in uploaded_files:
        for page in read_uploaded_file(uploaded_file):
            for piece in split_text(page["text"]):
                chunks.append({
                    "text": piece,
                    "source": page["source"],
                    "page": page["page"]
                })
    return chunks

def retrieve(question, chunks, model, top_k=3):
    corpus = [chunk["text"] for chunk in chunks]
    vectors = model.encode(corpus, normalize_embeddings=True)
    query_vector = model.encode([question], normalize_embeddings=True)[0]
    scores = np.dot(vectors, query_vector)
    best_indices = np.argsort(scores)[::-1][:top_k]
    return [(float(scores[i]), chunks[int(i)]) for i in best_indices]

def answer_from_context(question, retrieved):
    # Beginner-friendly baseline: show relevant text instead of inventing an AI-generated answer.
    context = "\n\n".join(
        f"Source: {item['source']}" +
        (f", page {item['page']}" if item["page"] else "") +
        f"\n{item['text']}"
        for score, item in retrieved
    )
    return context

model = load_model()

uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Reading documents and preparing search..."):
        chunks = build_chunks(uploaded_files)

    if not chunks:
        st.error("No readable text was found. Scanned PDFs may need OCR first.")
    else:
        st.success(f"Ready! Created {len(chunks)} searchable text chunks.")
        question = st.text_input("Ask a question about your documents")
        top_k = st.slider("Number of passages to retrieve", min_value=1, max_value=min(5, len(chunks)), value=min(3, len(chunks)))

        if st.button("Find answer", type="primary"):
            if not question.strip():
                st.warning("Please type a question.")
            else:
                with st.spinner("Searching for relevant passages..."):
                    results = retrieve(question, chunks, model, top_k=top_k)

                best_score = results[0][0]
                st.subheader("Retrieved information")
                st.caption(f"Best similarity score: {best_score:.2f}. This is a similarity measure, not a probability of correctness.")

                if best_score < 0.25:
                    st.warning("These passages may not answer the question. Try different wording or check whether the documents contain the answer.")

                for rank, (score, item) in enumerate(results, start=1):
                    location = f" • Page {item['page']}" if item["page"] else ""
                    with st.expander(f"{rank}. {item['source']}{location} • Similarity {score:.2f}", expanded=(rank == 1)):
                        st.write(item["text"])

                st.subheader("Context for an answer")
                st.write("This starter version displays the most relevant passages rather than generating unsupported claims. You can add a language model later to summarize only these passages.")
else:
    st.info("Start by uploading a PDF or TXT document.")
    st.markdown("**Privacy:** files are processed in this running app; do not upload confidential documents to a public deployment.")
