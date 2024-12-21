# import re
# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from fastapi.middleware.cors import CORSMiddleware
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# import pandas as pd
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from os import environ as env

# # Initialize the FastAPI app
# app = FastAPI()

# # Add CORS middleware to allow all origins (adjust in production)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load the embedding model
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Small and efficient model
# model = AutoModelForCausalLM.from_pretrained("models/distilgpt2_model")
# tokenizer = AutoTokenizer.from_pretrained("models/distilgpt2_model")

# # Load data
# try:
#     texts = pd.read_csv("data/content_data.csv")

#     # Generate embeddings for the text column
#     embeddings = embedding_model.encode(
#         texts["detail"].tolist(), show_progress_bar=True
#     )

#     # Normalize embeddings for better similarity search
#     embeddings = np.array(embeddings)
#     embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

#     # Initialize FAISS index
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dimension)  # L2 distance-based similarity
#     index.add(embeddings)

#     # Store text data alongside FAISS index
#     id_to_text = {i: texts["detail"].tolist()[i] for i in range(len(texts))}

# except Exception as e:
#     print(f"Error loading data: {e}")
#     texts = pd.DataFrame({"text": ["Sample text for fallback"]})
#     embeddings = embedding_model.encode(texts["text"].tolist())
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dimension)
#     index.add(embeddings)
#     id_to_text = {0: "Sample text for fallback"}


# def generate_response_with_local_model(query):
#     try:
#         # Brief overview context for the FAQ chatbot (Dynode context)
#         chatbot_context = """
# Dynode Company Overview:
# - Specializes in distribution management solutions
# - Offers wholesale and retail management automation
# - Founded to streamline business operations
# - Key services: inventory management, order processing, automation tools
# - Target industries: wholesale, retail, supply chain management

# Mission: Simplify complex business operations through intelligent automation
# """

#         # Generate embedding for the query
#         query_embedding = embedding_model.encode([query])
#         query_embedding = query_embedding / np.linalg.norm(
#             query_embedding, axis=1, keepdims=True
#         )

#         # Search FAISS for the nearest neighbors
#         distances, indices = index.search(
#             query_embedding, k=3
#         )  # Retrieve top-3 matches

#         # Retrieve the corresponding texts (context related to the query)
#         context = " ".join([str(id_to_text[i]) for i in indices[0]])

#         # Formulate the prompt to directly focus on the answer, including only the essential context
#         prompt = f"""
# Context: {chatbot_context} {context}

# Question: "{query}"

# Answer:
# """

#         # Tokenize the prompt
#         input_ids = tokenizer.encode(prompt, return_tensors="pt")
#         attention_mask = torch.ones(input_ids.shape, device=input_ids.device)

#         # Generate a response using the model
#         outputs = model.generate(
#             input_ids,
#             attention_mask=attention_mask,
#             max_new_tokens=100,  # Limit to the number of tokens for a concise response
#             num_return_sequences=1,
#             no_repeat_ngram_size=2,
#             temperature=0.7,
#             top_k=40,
#             top_p=0.9,
#             do_sample=True,
#         )

#         # Decode the response
#         response = tokenizer.decode(outputs[0], skip_special_tokens=True)

#         # Post-process the response to ensure only the answer is returned
#         cleaned_response = response.split("Answer:")[-1].strip()

#         return cleaned_response if cleaned_response else "Not enough information"

#     except Exception as e:
#         print(f"Error generating response: {e}")
#         return "I apologize, but I cannot generate a response at the moment."


# # WebSocket endpoint for chatbot
# @app.websocket("/ws/chatbot")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     try:
#         while True:
#             # Receive message from client
#             query = await websocket.receive_text()

#             # Generate response using the local model
#             response = generate_response_with_local_model(query)

#             # Send response back to client
#             await websocket.send_text(response)

#     except WebSocketDisconnect:
#         print("WebSocket connection closed")


# # Optional: Health check endpoint
# @app.get("/health")
# async def health_check():
#     return {"status": "healthy"}


import re
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from os import environ as env

# Initialize the FastAPI app
app = FastAPI()

# Add CORS middleware to allow all origins (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Enhanced Embedding Strategy
def preprocess_text(text):
    """
    Preprocess and normalize text for embedding
    """
    # Convert to lowercase
    text = str(text).lower()
    # Remove punctuation and special characters
    text = re.sub(r"[^\w\s]", "", text)
    # Remove extra whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Load a more comprehensive embedding model
embedding_model = SentenceTransformer("all-mpnet-base-v2")
model = AutoModelForCausalLM.from_pretrained("models/distilgpt2_model")
tokenizer = AutoTokenizer.from_pretrained("models/distilgpt2_model")

# Load data with enhanced preprocessing
try:
    # Load original data
    texts = pd.read_csv("data/content_data.csv")

    # Preprocess text column
    texts["processed_detail"] = texts["detail"].apply(preprocess_text)

    # Generate embeddings with batch processing
    embeddings = embedding_model.encode(
        texts["processed_detail"].tolist(),
        show_progress_bar=True,
        batch_size=32,  # Optimize batch processing
        convert_to_numpy=True,
    )

    # Normalize embeddings for better similarity search
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Adaptive FAISS Indexing for small datasets
    dimension = embeddings.shape[1]

    # Use a simple flat index for small number of vectors
    if len(embeddings) <= 20:
        index = faiss.IndexFlatL2(dimension)
    else:
        # Create a quantizer for more efficient search
        quantizer = faiss.IndexFlatL2(dimension)

        # Dynamically adjust clustering based on dataset size
        num_clusters = min(100, max(int(len(embeddings) / 5), 10))

        # Use Inverted File (IVF) index
        index = faiss.IndexIVFFlat(
            quantizer,  # Quantizer
            dimension,  # Dimension of embeddings
            num_clusters,  # Number of clusters
        )

        # Train the index
        index.train(embeddings)

    # Add embeddings to the index
    index.add(embeddings)

    # Store text data alongside FAISS index
    id_to_text = {i: texts["detail"].tolist()[i] for i in range(len(texts))}

except Exception as e:
    print(f"Error loading data: {e}")
    # Fallback mechanism
    texts = pd.DataFrame({"text": ["Sample fallback text for Dynode company"]})
    texts["processed_detail"] = texts["text"].apply(preprocess_text)
    embeddings = embedding_model.encode(texts["processed_detail"].tolist())
    dimension = embeddings.shape[1]

    # Simple fallback index
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    id_to_text = {0: "Sample fallback text"}


def generate_response_with_local_model(query):
    try:
        # Brief overview context for the FAQ chatbot (Dynode context)
        chatbot_context = """
Dynode Company Overview:
- Specializes in distribution management solutions
- Offers wholesale and retail management automation
- Founded to streamline business operations
- Key services: inventory management, order processing, automation tools
- Target industries: wholesale, retail, supply chain management

Mission: Simplify complex business operations through intelligent automation
"""

        # Preprocess the query
        processed_query = preprocess_text(query)

        # Generate embedding for the query
        query_embedding = embedding_model.encode([processed_query])
        query_embedding = query_embedding / np.linalg.norm(
            query_embedding, axis=1, keepdims=True
        )

        # Safely search FAISS for the nearest neighbors
        k = min(3, len(id_to_text))
        distances, indices = index.search(query_embedding, k)  # Retrieve top matches

        # Retrieve the corresponding texts (context related to the query)
        context = " ".join(
            [str(id_to_text.get(i, "")) for i in indices[0] if i in id_to_text]
        )

        # Formulate the prompt to directly focus on the answer, including only the essential context
        prompt = f"""
Context: {chatbot_context} {context}

Question: "{query}"

Answer: 
"""

        # Tokenize the prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        attention_mask = torch.ones(input_ids.shape, device=input_ids.device)

        # Generate a response using the model
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=100,  # Limit to the number of tokens for a concise response
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            do_sample=True,
        )

        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Post-process the response to ensure only the answer is returned
        cleaned_response = response.split("Answer:")[-1].strip()

        return cleaned_response if cleaned_response else "Not enough information"

    except Exception as e:
        print(f"Error generating response: {e}")
        return "I apologize, but I cannot generate a response at the moment."


# Rest of the code remains the same...


# WebSocket endpoint remains the same
@app.websocket("/ws/chatbot")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive message from client
            query = await websocket.receive_text()

            # Generate response using the local model
            response = generate_response_with_local_model(query)

            # Send response back to client
            await websocket.send_text(response)

    except WebSocketDisconnect:
        print("WebSocket connection closed")


# Optional: Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
