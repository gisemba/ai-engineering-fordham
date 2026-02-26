# Fordham RAG App
import streamlit as st
import pandas as pd
import numpy as np
import litellm
import textwrap
import pickle
from helpers import build_index, score_bm25, normalize_scores

# --- CONFIG & CACHING ---
st.set_page_config(page_title="Fordham RAG Assistant", page_icon="🐏")

@st.cache_resource
def load_resources():
    """Load heavy assets once and keep them in memory."""
    # Load metadata
    metadata_df = pd.read_parquet('temp/fordham_metadata.parquet')
    
    # Load Embeddings
    normalized_embeddings = np.load('temp/fordham_norm_embeddings.npy')
    
    # Load or Rebuild BM25 Index 
    # (Rebuilding is fast, but you can also pickle it in the notebook)
    bm25_docs = metadata_df["chunk_content"].astype(str).tolist()
    bm25_index, bm25_doc_lengths = build_index(bm25_docs)
    
    return metadata_df, normalized_embeddings, bm25_index, bm25_doc_lengths

metadata_df, normalized_embeddings, bm25_index, bm25_doc_lengths = load_resources()
num_docs = len(metadata_df)

# --- RAG LOGIC (Extracted from your notebook) ---

def _embed_query(query: str, model: str = "openai/text-embedding-3-small") -> np.ndarray:
    resp = litellm.embedding(model=model, input=query)
    vec = np.array(resp["data"][0]["embedding"], dtype=np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-9)
    return vec

def retrieve_hybrid(question: str, k: int = 8, alpha: float = 0.6) -> pd.DataFrame:
    q_vec = _embed_query(question)
    semantic_scores = normalized_embeddings @ q_vec
    bm25_scores = score_bm25(question, bm25_index, num_docs, bm25_doc_lengths)
    
    sem_norm = normalize_scores(semantic_scores)
    bm25_norm = normalize_scores(bm25_scores)
    hybrid_scores = alpha * sem_norm + (1.0 - alpha) * bm25_norm

    top_idx = np.argsort(-hybrid_scores)[:k]
    results = metadata_df.iloc[top_idx].copy()
    results["hybrid_score"] = hybrid_scores[top_idx]
    return results

def generate_answer(question: str, retrieved_chunks: pd.DataFrame) -> str:
    context_pieces = [f"[SOURCE: {row.get('url', '')}]\n{row['chunk_content']}" for _, row in retrieved_chunks.iterrows()]
    context = "\n\n---\n\n".join(context_pieces)[:6000]

    system_msg = (
        "You are a helpful, professional assistant for Fordham University. "
        "Your task is to answer user questions using ONLY the provided context. "
        "\n\nGUIDELINES:"
        "\n- If the answer is not in the context, say: 'I'm sorry, I don't have enough information to answer that based on current documents I have.' Do not make up facts."
        "\n- Cite your sources by mentioning the page name where the information was found."
        "\n- Keep your tone friendly and professional."
    )
    user_msg = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

    resp = litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        temperature=0.2,
    )
    return resp["choices"][0]["message"]["content"].strip()

# --- STREAMLIT UI ---
LOGO_PATH = "images/fordhamlogo.svg"
st.image(LOGO_PATH, width=200)

st.title("Fordham University AMA")
st.markdown("Curious about Fordham? Ask and we'll do our best to answer your questions.")

# Sidebar for parameters
st.sidebar.header("Settings")
k_val = st.sidebar.slider("Number of chunks (k)", 1, 15, 8)
alpha_val = st.sidebar.slider("Hybrid Weight (Alpha)", 0.0, 1.0, 0.6, help="1.0 is pure semantic, 0.0 is pure BM25")

query = st.text_input("Ask a question about Fordham:", placeholder="e.g. What is the Gabelli School of Business?")

if query:
    with st.spinner("Searching and generating..."):
        # Run Pipeline
        retrieved = retrieve_hybrid(query, k=k_val, alpha=alpha_val)
        answer = generate_answer(query, retrieved)
        
        # Display Answer
        st.subheader("Answer")
        st.write(answer)
        
        # Display Sources
        st.subheader("Sources")
        for _, row in retrieved.iterrows():
            with st.expander(f"Source: {row['page_name']} (Score: {row['hybrid_score']:.3f})"):
                st.write(row['chunk_content'])
                st.write(f"[Open Page]({row['url']})")