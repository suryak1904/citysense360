import os
import json
import faiss
import numpy as np
import requests
from typing import List

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


class EmergencyRAGService:
    """
    Emergency Incident Notification System
    RAG + Local Ollama Qwen 2.5
    """

    def __init__(
        self,
        docs_dir: str = "data/emergency_docs",
        model_name: str = "qwen2.5:3b",
        top_k: int = 3,
    ):
        self.docs_dir = docs_dir
        self.model_name = model_name
        self.top_k = top_k

        # Embedding model (lightweight, local)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Load + index SOP documents
        self.chunks = self._load_and_chunk_docs()
        self.index = self._build_faiss_index(self.chunks)


    def _ollama_generate(self, prompt: str, max_tokens: int = 300) -> str:
        url = "http://localhost:11434/api/generate"

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": max_tokens
            }
        }

        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()

        return response.json()["response"]

    def _load_and_chunk_docs(self) -> List[str]:
        documents = []

        for file in os.listdir(self.docs_dir):
            if file.endswith(".txt"):
                with open(os.path.join(self.docs_dir, file), "r", encoding="utf-8") as f:
                    documents.append(f.read())

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )

        chunks = []
        for doc in documents:
            chunks.extend(splitter.split_text(doc))

        return chunks

    def _build_faiss_index(self, chunks: List[str]):
        embeddings = self.embedder.encode(chunks, convert_to_numpy=True)
        dim = embeddings.shape[1]

        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        return index

    def retrieve_context(self, query: str) -> str:
        query_emb = self.embedder.encode([query])
        _, idxs = self.index.search(query_emb, self.top_k)

        return "\n".join([self.chunks[i] for i in idxs[0]])


    def handle_incident(self, incident_text: str) -> dict:
        context = self.retrieve_context(incident_text)

        prompt = f"""
You are a city emergency response AI.

Use ONLY the context below (official SOPs).
Do not hallucinate.

Context:
{context}

Incident:
{incident_text}

Generate:
1. Incident summary
2. Severity (Low / Medium / High)
3. Recommended actions
4. Department to notify
"""

        response = self._ollama_generate(prompt)

        return {
            "incident": incident_text,
            "response": response
        }



if __name__ == "__main__":
    rag = EmergencyRAGService()

    test_incident = """
    CCTV detected a vehicle collision.
    Crowd gathering observed.
    Possible injuries reported.
    """

    result = rag.handle_incident(test_incident)
    print(json.dumps(result, indent=2))
