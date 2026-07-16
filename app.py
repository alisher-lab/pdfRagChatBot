# app.py — Streamlit PDF RAG Chatbot (Qwen2.5-3B via Hugging Face Inference API, streaming)

import streamlit as st
import torch
import os

from dotenv import load_dotenv
from huggingface_hub import HfApi

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Load variables from a local .env file (if one exists) into os.environ.
# .env should contain a line like: HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
load_dotenv()

st.set_page_config(page_title="PDF Chatbot (Qwen2.5-3B)", page_icon="📄")


def validate_hf_token(token: str):
    """Checks the token is real and active by asking HF who it belongs to."""
    try:
        api = HfApi()
        user_info = api.whoami(token=token)
        return True, user_info.get("name", "unknown user")
    except Exception as e:
        return False, str(e)


# ---------- Token resolution: st.secrets -> .env -> manual input ----------

def get_secrets_token():
    """st.secrets works both locally (.streamlit/secrets.toml) and on
    Streamlit Community Cloud (dashboard-configured secrets). Accessing it
    when no secrets file exists at all doesn't error — it's just empty."""
    try:
        return st.secrets.get("HF_TOKEN")
    except Exception:
        return None


with st.sidebar:
    st.subheader("Authentication")

    secrets_token = get_secrets_token()
    env_token = os.getenv("HF_TOKEN")

    if secrets_token:
        source = "st.secrets"
        candidate_token = secrets_token
    elif env_token:
        source = ".env"
        candidate_token = env_token
    else:
        candidate_token = None

    if candidate_token:
        valid, info = validate_hf_token(candidate_token)
        if valid:
            st.success(f"Authenticated via {source} as **{info}**")
            manual_token = candidate_token
        else:
            st.error(f"Token found in {source} but invalid: {info}")
            st.stop()
    else:
        st.info("No secrets.toml or .env token found — enter one manually.")
        manual_token = st.text_input(
            "Hugging Face Token:",
            type="password",
            help="Get a token from huggingface.co/settings/tokens",
        )
        if manual_token:
            valid, info = validate_hf_token(manual_token)
            if valid:
                st.success(f"Authenticated as **{info}**")
            else:
                st.error(f"Authentication failed: {info}")
                st.stop()
        else:
            st.warning("Please provide a Hugging Face token to continue.")
            st.stop()

os.environ["HF_TOKEN"] = manual_token


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
        streaming=True,
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
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    return context, docs


def stream_answer(llm, context: str, question: str):
    prompt_text = PROMPT.format(context=context, question=question)
    for chunk in llm.stream(prompt_text):
        if chunk.content:
            yield chunk.content


GREETINGS = {
    "hi", "hii", "hello", "hey", "heya", "yo", "hola",
    "good morning", "good afternoon", "good evening", "greetings",
}


def is_greeting(text: str) -> bool:
    cleaned = text.strip().lower().strip("!.?")
    return cleaned in GREETINGS


def stream_greeting():
    """Bypasses retrieval + the strict context-only prompt entirely, since
    forcing a plain 'hi' through that prompt is what produced the stiff,
    over-formal reply. Streamed word-by-word to match the normal answer UX."""
    reply = (
        "Hey there! 👋 I'm ready to help you with this document — "
        "ask me anything about it, like a summary, a specific detail, "
        "or an explanation of something in it. What would you like to know?"
    )
    for word in reply.split(" "):
        yield word + " "


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
                if is_greeting(question):
                    answer = st.write_stream(stream_greeting())
                else:
                    with st.spinner("Searching the document..."):
                        context, sources = retrieve_context(vectorstore, question)

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
