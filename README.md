# Local Persistent RAG

A local Streamlit RAG app for PDF, TXT, CSV and XLSX files using OpenAI embeddings/chat completions and a persistent ChromaDB vector store.

## Features

- Upload PDF, TXT, CSV and XLSX documents.
- Save original files locally in `data/documents/`.
- Persist embeddings, text chunks and metadata locally in `data/chroma/`.
- Restart the application without re-uploading documents.
- Avoid re-indexing identical file content using SHA-256 hashes.
- Ask questions in natural language and inspect retrieved source passages.

## Setup (Windows / VS Code)

1. Install Python 3.10 or newer.
2. Extract this folder and open it in VS Code.
3. In the terminal:

   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env`.
5. Put your OpenAI API key in `.env`:

   ```text
   OPENAI_API_KEY=your_real_key_here
   ```

6. Start the app:

   ```powershell
   streamlit run app.py
   ```

7. Open the local URL printed by Streamlit (normally `http://localhost:8501`).

## Persistence

Do not delete the `data/` folder. The source files are stored in `data/documents/` and ChromaDB persists its index in `data/chroma/`. Restarting Streamlit does not erase these folders. Back them up if the data matters.

## File Explanation:

WebscrapeWithModal.py will load data from the link "https://www.tnagrisnet.tn.gov.in/people_app/GoScheme" which scrapes data from the modal window.

WebscrapeWithoutModal.py will load data from the link "https://agritech.tnau.ac.in/expert_system/paddy/Schemes.html" which does not have the modal window.

app.py will load the UI with Streamlit.

## Notes / limitations

- Scanned/image-only PDFs may need OCR; this starter app extracts selectable PDF text only.
- XLSX is supported; legacy XLS is not included.
- OpenAI API access and billing are required for embeddings and answers. The API key is read from `.env` and is not stored in the source code.
- The original documents and vector index stay local, but document text used for embeddings and retrieved context sent for answers is sent to OpenAI.
- A retrieved answer depends on extraction quality, chunking and retrieval settings. Check the source snippets for important decisions.
- Do not expose `.env` or commit it to a public repository.
- To completely reset the knowledge base, stop the app and back up/delete `data/chroma/` and `data/documents/`.
