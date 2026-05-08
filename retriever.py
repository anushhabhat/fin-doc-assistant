from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

CHROMA_PATH = "chroma_db"

def get_retriever(k=5):
    #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})

def retrieve(question, k=5):
    retriever = get_retriever(k)
    docs = retriever.invoke(question)
    return docs