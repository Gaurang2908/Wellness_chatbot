import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def load_index_and_insights():
    with open("embeddings/faiss_index.pkl", "rb") as f:
        index, insights = pickle.load(f)
    return index, insights

def search_insight(query, model, index, insights):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), k=1)
    return insights[I[0][0]]

st.set_page_config(page_title="Wellness Analytics Chatbot", layout="centered")
st.title("🏥 Wellness Analytics Chatbot")

query = st.text_input("Ask a question about employee wellness:")

if query:
    model = load_model()
    index, insights = load_index_and_insights()
    matched = search_insight(query, model, index, insights)

    st.subheader("📊 Insight:")
    st.write(matched["insight"])

    if matched.get("chart_path"):
        img = Image.open(matched["chart_path"])
        st.image(img, caption="Chart", use_column_width=True)
