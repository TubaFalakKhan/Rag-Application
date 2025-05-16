from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA

from ingest_folder import ingest_folder
from ingest_file import ingest_file
from store import vector_store

import os
import tempfile

# 1. Initialize FastAPI app
app = FastAPI()

# 2. Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. LLM Setup (Ollama with LLaMA2)
llm = ChatOllama(model="llama2")

# 4. Retrieval Chain
retriever = vector_store.as_retriever()
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 5. Ask endpoint
@app.post("/ask")
def ask_api(query: str = Form(...)):
    try:
        answer = rag_chain.run(query)
        return {"query": query, "answer": answer}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# 6. Ingest File endpoint
@app.post("/ingest-file")
async def ingest_file_api(tenant_id: str = Form(...), file: UploadFile = Form(...)):
    if not file.filename.endswith((".pdf", ".txt", ".docx")):
        return JSONResponse(status_code=400, content={"error": "Unsupported file type."})
    try:
        # Save uploaded file to a temporary path
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        # Ingest file
        chunk_count = ingest_file(tmp_path, tenant_id)

        # Cleanup
        os.remove(tmp_path)

        return {"message": "Ingestion complete.", "chunks": chunk_count}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# 7. Ingest Folder endpoint
@app.post("/ingest-folder")
def ingest_folder_api(tenant_id: str = Form(...), folder_path: str = Form(...)):
    if not os.path.exists(folder_path):
        return JSONResponse(status_code=404, content={"error": "Folder path does not exist."})
    try:
        success, errors = ingest_folder(folder_path, tenant_id)
        return {
            "message": "Ingestion complete.",
            "success": success,
            "errors": errors
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# 8. Health check (optional but useful)
@app.get("/health")
def health_check():
    return {"status": "running"}
