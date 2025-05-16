import os
from typing import List, Tuple, Dict
from langchain_community.document_loaders import TextLoader, DedocFileLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from store import vector_store

text_splitter = RecursiveCharacterTextSplitter(
chunk_size=1000,
chunk_overlap=150,
separators=["\n\n", "\n", " ", ""]
)

def load_and_split_pdf(pdf_path: str):
    pages = PyPDFLoader(pdf_path).load()
    return text_splitter.split_documents(pages)

def load_and_split_txt(txt_path: str):
    docs = TextLoader(txt_path).load()
    return text_splitter.split_documents(docs)

def load_and_split_docx(docx_path: str):
    docs = DedocFileLoader(docx_path).load()
    return text_splitter.split_documents(docs)

def ingest_document(file_path: str, tenant_id: str) -> int:
    if file_path.endswith(".pdf"):
        chunks = load_and_split_pdf(file_path)
    elif file_path.endswith(".txt"):
        chunks = load_and_split_txt(file_path)
    elif file_path.endswith(".docx"):
        chunks = load_and_split_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def process_and_add_documents(chunks, tenant_id, file_path, vector_store):
    for doc in chunks:
        doc.metadata["tenantId"] = tenant_id
        doc.metadata["source_file"] = os.path.basename(file_path)

    vector_store.add_documents(chunks)
    return len(chunks)


def ingest_folder(folder_path: str, tenant_id: str) -> Tuple[List[Dict], List[Dict]]:
    success, errors = [], []
    for file_name in os.listdir(folder_path):
        if file_name.endswith((".pdf", ".txt", ".docx")):
            try:
                full_path = os.path.join(folder_path, file_name)
                chunk_count = ingest_document(full_path, tenant_id)
                success.append({file_name: chunk_count})
            except Exception as e:
                errors.append({file_name: str(e)})
                return success, errors