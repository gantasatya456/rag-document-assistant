# RAG Document Assistant (Beginner Project)

A beginner-friendly Retrieval-Augmented Generation (RAG) starter app. It reads PDF/TXT files, splits text into chunks, uses sentence embeddings to retrieve relevant passages, and displays sources.

## Run locally in VS Code

1. Install Python 3.10 or newer.
2. Open this folder in VS Code.
3. Open the terminal in VS Code.
4. Create a virtual environment (optional but recommended):

   Windows:
   ```bash
   py -m venv .venv
   .venv\Scripts\activate
   ```

   macOS/Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

5. Install libraries:
   ```bash
   pip install -r requirements.txt
   ```
6. Run the app:
   ```bash
   streamlit run app.py
   ```

The first run downloads the sentence-transformer model, so it may take a few minutes and requires internet access.

## How it works

1. Upload PDF/TXT files.
2. Extract text.
3. Split text into overlapping chunks.
4. Convert chunks and question into sentence embeddings.
5. Rank chunks by cosine similarity.
6. Show the best matching passages and source page.

## Current limitation

This starter app retrieves and displays relevant passages; it does not generate a free-form AI answer yet. This is intentional for a simple, source-grounded first version. Scanned PDFs may need OCR.
