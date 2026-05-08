import os
import fitz
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

CHROMA_PATH = "chroma_db"
PDF_FOLDER = "data/pdfs"

def load_pdfs(folder):
    docs = []
    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            path = os.path.join(folder, filename)
            pdf = fitz.open(path)
            for i, page in enumerate(pdf):
                text = page.get_text()
                if text.strip():
                    docs.append({
                        "text": text,
                        "metadata": {"source": filename, "page": i + 1}
                    })
            print(f"Loaded {filename} — {len(pdf)} pages")
    return docs

def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks, metadatas = [], []
    for doc in docs:
        splits = splitter.split_text(doc["text"])
        chunks.extend(splits)
        metadatas.extend([doc["metadata"]] * len(splits))
    print(f"Created {len(chunks)} chunks")
    return chunks, metadatas

def ingest():
    os.makedirs(PDF_FOLDER, exist_ok=True)
    os.makedirs(CHROMA_PATH, exist_ok=True)

    docs = load_pdfs(PDF_FOLDER)
    if not docs:
        print("No PDFs found in data/pdfs/ — add some PDFs and run again.")
        return

    chunks, metadatas = chunk_docs(docs)

    #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=CHROMA_PATH
    )
    print(f"Saved {len(chunks)} chunks to ChromaDB at '{CHROMA_PATH}'")

if __name__ == "__main__":
    ingest()