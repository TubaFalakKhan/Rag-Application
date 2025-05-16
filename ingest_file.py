import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DedocFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import PGVector

# Load embedding model and connection string
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
CONNECTION_STRING = "postgresql+psycopg2://postgres:postgres@localhost:5432/ragdb"

# Chunking setup
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", " ", ""]
)

def load_and_split(file_path: str):
    if file_path.endswith(".pdf"):
        return text_splitter.split_documents(PyPDFLoader(file_path).load())
    elif file_path.endswith(".txt"):
        return text_splitter.split_documents(TextLoader(file_path).load())
    elif file_path.endswith(".docx"):
        return text_splitter.split_documents(DedocFileLoader(file_path).load())
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def ingest_file(file_path: str, tenant_id: str) -> int:
    chunks = load_and_split(file_path)
    for doc in chunks:
        doc.metadata["tenantId"] = tenant_id
        doc.metadata["source_file"] = os.path.basename(file_path)

    print(f"Ingesting {len(chunks)} chunks...")

    # Instead of using a pre-made vector_store, create it here
    vector_store = PGVector.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name="multi_format_docs",
        connection_string=CONNECTION_STRING,
    )

    return len(chunks)

# Example usage:
# ingest_file("your_file.pdf", tenant_id="tenant_123")
