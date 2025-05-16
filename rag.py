def ask_question(query: str) -> str:
    return rag_chain.run(query)

@app.post("/ask")
def ask_api(query: str):
    try:
        answer = ask_question(query)
        return {"query": query, "answer": answer}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/ingest-file")
async def ingest_file(tenant_id: str, file: UploadFile):
    # Validate file type
    if not file.filename.endswith((".pdf", ".txt", ".docx")):
        return JSONResponse(status_code=400, content={"error": "Unsupported file type."})

    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        # Ingest the file
        chunk_count = ingest_file(tmp_path, tenant_id)
        # Clean up the temporary file
        os.remove(tmp_path)

        return {"message": "Ingestion complete.", "chunks": chunk_count}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/ingest-folder")
def ingest_folder(tenant_id: str, folder_path: str):
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
