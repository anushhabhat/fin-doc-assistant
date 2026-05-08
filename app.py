import streamlit as st
import os
from ingest import ingest, PDF_FOLDER
from chain import answer

st.set_page_config(page_title="Financial Document Assistant", page_icon="📄")
st.title("Financial Document Intelligence Assistant")
st.caption("Upload financial PDFs and ask questions about them")

with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader(
        "Upload PDFs", type="pdf", accept_multiple_files=True
    )
    if uploaded:
        os.makedirs(PDF_FOLDER, exist_ok=True)
        for f in uploaded:
            path = os.path.join(PDF_FOLDER, f.name)
            with open(path, "wb") as out:
                out.write(f.read())
        st.success(f"{len(uploaded)} file(s) saved")

    if st.button("Ingest documents", type="primary"):
        with st.spinner("Processing PDFs..."):
            ingest()
        st.success("Done! You can now ask questions.")

    st.divider()
    st.markdown("**How it works**")
    st.markdown("1. Upload PDFs\n2. Click Ingest\n3. Ask questions below")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            response = answer(question)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})