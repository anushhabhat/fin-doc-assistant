from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from retriever import retrieve

load_dotenv()

PROMPT_TEMPLATE = """
You are a financial analyst assistant. Answer the user's question using ONLY 
the context below. If the answer isn't in the context, say 
"I couldn't find that in the uploaded documents."

Always end your answer with a "Sources:" line listing the document name and 
page number for each chunk you used.

Context:
{context}

Question: {question}

Answer:
"""

def answer(question):
    docs = retrieve(question)
    if not docs:
        return "No relevant content found in the uploaded documents."

    context = "\n\n---\n\n".join([
        f"[{d.metadata.get('source', 'unknown')} | Page {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    ])

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE
    )

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    chain = prompt | llm

    response = chain.invoke({"context": context, "question": question})
    return response.content