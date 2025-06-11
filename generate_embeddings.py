from sentence_transformers import SentenceTransformer
import faiss, json, pickle

model = SentenceTransformer('all-MiniLM-L6-v2')

with open("insights/wellness_insights.json") as f:
    insights = json.load(f)

questions = [i['question'] for i in insights]
embeddings = model.encode(questions)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

with open("embeddings/faiss_index.pkl", "wb") as f:
    pickle.dump((index, insights), f)

print("Index built and saved.")
