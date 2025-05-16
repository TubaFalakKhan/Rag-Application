# store.py

from langchain_community.vectorstores.pgvector import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document  # for testing/demo only

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

CONNECTION_STRING = "postgresql+psycopg2://postgres:postgres@localhost:5432/ragdb"

# Initialize with dummy data or move this to ingest step
dummy_doc = [Document(page_content="Hello world", metadata={"source": "test"})]

vector_store = PGVector.from_documents(
    documents=dummy_doc,  # Replace with real docs in your ingest script
    embedding=embedding_model,
    collection_name="multi_format_docs",
    connection_string=CONNECTION_STRING,
)
