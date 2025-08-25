import os
from dotenv import load_dotenv
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Load env vars
load_dotenv()

st.set_page_config(page_title="GenAI RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 GenAI RAG Chatbot")

# --- Sidebar File Uploader ---
with st.sidebar:
    st.header("📂 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs", type=["pdf"], accept_multiple_files=True
    )
    if st.button("🗑️ Reset Vectorstore"):
        if os.path.exists("vectorstore"):
            import shutil

            shutil.rmtree("vectorstore")
            st.success("Vectorstore has been reset. Upload new files to rebuild it.")
            st.stop()

# --- Build or Load Vectorstore ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

if uploaded_files:
    docs = []
    for file in uploaded_files:
        temp_path = os.path.join("temp", file.name)
        os.makedirs("temp", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())
        loader = PyPDFLoader(temp_path)
        docs.extend(loader.load())

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Check if vectorstore already exists → append instead of overwrite
    if os.path.exists("vectorstore"):
        vs = FAISS.load_local(
            "vectorstore", embeddings, allow_dangerous_deserialization=True
        )
        vs.add_documents(splits)
        vs.save_local("vectorstore")
    else:
        vs = FAISS.from_documents(splits, embeddings)
        vs.save_local("vectorstore")
else:
    if os.path.exists("vectorstore"):
        vs = FAISS.load_local(
            "vectorstore", embeddings, allow_dangerous_deserialization=True
        )
    else:
        st.warning(
            "⚠️ No PDFs uploaded and no existing vectorstore found. Please upload documents."
        )
        st.stop()

retriever = vs.as_retriever(search_kwargs={"k": 3})

# --- Load OpenAI LLM ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- RetrievalQA Chain ---
qa = RetrievalQA.from_chain_type(
    llm=llm, retriever=retriever, chain_type="stuff", return_source_documents=True
)

# --- Chat UI ---
if "history" not in st.session_state:
    st.session_state.history = []

query = st.chat_input("Ask me something about your docs...")

if query:
    result = qa.invoke({"query": query})
    answer = result["result"]
    sources = result["source_documents"]

    # Save chat to history
    st.session_state.history.append((query, answer, sources))

# Display chat history
for q, a, s in st.session_state.history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)
        if s:
            with st.expander("📖 Sources"):
                for i, doc in enumerate(s, 1):
                    filename = os.path.basename(doc.metadata.get("source", "Unknown"))
                    st.markdown(
                        f"**{i}. {filename} (page {doc.metadata.get('page', '?')})**"
                    )
                    st.caption(doc.page_content[:300] + "...")
