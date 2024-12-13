from typing import Optional
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

app = FastAPI()

# Load dataset
data = pd.read_csv("dataset.csv")

# Load model and create FAISS index
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(data['question'].values)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Retrieve function
def retrieve(query, top_k=1):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return data.iloc[indices[0]]['answer'].values[0]

# Serve the chatbot HTML
@app.get("/")
async def get_index():
    return FileResponse("index.html")

# WebSocket for real-time chat
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        question = await websocket.receive_text()
        answer = retrieve(question)
        await websocket.send_text(answer)

# Existing API endpoints
@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}
