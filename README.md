This is a lightweight Retrieval-Augmented Generation (RAG) application built using FastAPI, LangChain, HuggingFace Embeddings, Ollama (LLaMA2), and Elasticsearch. It allows ingestion of documents tenant-wise and querying them using natural language through an LLM-powered chatbot.

🧠 Problem Statement
Generic chatbots lack context and fail when users ask domain-specific queries. Organizations require private, document-based Q&A systems that offer relevant and accurate answers based on internal knowledge and are tenant-aware for data isolation.

💡 Solution
We developed a FastAPI-powered RAG chatbot that:

Supports multi-tenancy via tenant ID

Ingests documents from files or folders

Stores vector embeddings using HuggingFace models

Retrieves relevant chunks using Elasticsearch

Answers using Ollama LLaMA2 via LangChain’s RetrievalQA

⚙️ Tech Stack
Component	Technology Used
Backend Framework	FastAPI
Embeddings	HuggingFace (all-MiniLM-L6-v2)
Vector Store	Elasticsearch (LangChain wrapper)
LLM	Ollama (with LLaMA2)
RAG Orchestration	LangChain
File Types	.pdf, .docx, .txt
API Testing	Postman, cURL

