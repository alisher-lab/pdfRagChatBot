# app.py — Streamlit PDF RAG Chatbot (Qwen2.5-3B via Hugging Face Inference API, streaming)

import streamlit as st
import torch
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from huggingface_hub import login
from langchain_openai import ChatOpenAI

# 1. Create a sidebar input for the token
with st.sidebar:
    st.subheader("Authentication")
    manual_token = st.text_input(
        "Enter your Hugging Face Token:",
        type="password",
        help="Get a token from huggingface.co/settings/tokens"
    )
    auth_button = st.button("Authenticate")

# 2. Authenticate when the user provides the token
if manual_token:
    try:
        login(token=manual_token)
        os.environ["HF_TOKEN"] = manual_token
        st.sidebar.success("Successfully authenticated!")
    except Exception as e:
        st.sidebar.error(f"Authentication failed: {e}")
else:
    st.sidebar.warning("Please enter your Hugging Face Token to load the model.")
    st.stop()

st.set_page_config(page_title="PDF Chatbot (Qwen2.5-3B)", page_icon="📄")

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct:featherless-ai"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PROMPT_TEMPLATE = """You are a helpful assistant that answers questions using ONLY the provided context from a PDF document. If the answer is not in the context, say "I don't know based on the document."

Context:
{context}

Question: {question}"""

PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])


# ---------- Cached resources: built ONCE per server process, reused across every rerun ----------

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": device},
    )


@st.cache_resource(show_spinner="Connecting to Qwen2.5-3B via Hugging Face Inference API...")
def load_llm(_token: str):
    return ChatOpenAI(
        model=MODEL_NAME,
        api_key=_token,
        base_url="https://router.huggingface.co/v1",
        max_tokens=512,
        temperature=0.1,
        streaming=True,   # enables token-by-token streaming
    )


@st.cache_resource(show_spinner="Processing PDF and building Chroma index...")
def build_vectorstore(pdf_path: str, _embeddings):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,  # overlapping technique
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    persist_dir = f"chroma_db_{os.path.basename(pdf_path)}"
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=_embeddings,
        persist_directory=persist_dir,
    )
    return vectorstore, len(chunks)


def retrieve_context(vectorstore, question: str, k: int = 4):
    """Runs retrieval only — no generation — so we can stream the LLM part separately."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    return context, docs


def stream_answer(llm, context: str, question: str):
    """Generator that yields text chunks as they arrive — fed straight into st.write_stream."""
    prompt_text = PROMPT.format(context=context, question=question)
    for chunk in llm.stream(prompt_text):
        if chunk.content:
            yield chunk.content


# ---------- App UI ----------

st.title("📄 PDF Chatbot — Qwen2.5-3B (via HF Inference API)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    pdf_path = f"/tmp/{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    embeddings = load_embeddings()
    llm = load_llm(manual_token)
    vectorstore, num_chunks = build_vectorstore(pdf_path, embeddings)

    st.success(f"PDF indexed into {num_chunks} chunks. Ask away.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask something about the PDF...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching the document..."):
                    context, sources = retrieve_context(vectorstore, question)

                # st.write_stream consumes the generator chunk-by-chunk, rendering
                # each piece as it arrives, and returns the full concatenated text.
                answer = st.write_stream(stream_answer(llm, context, question))

                with st.expander("Sources"):
                    for doc in sources:
                        page = doc.metadata.get("page", "?")
                        st.markdown(f"**Page {page}:** {doc.page_content[:200]}...")
            except Exception as e:
                answer = f"Error calling the model: {e}"
                st.error(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Upload a PDF to start chatting.")
