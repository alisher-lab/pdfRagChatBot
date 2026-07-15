# app.py — Streamlit PDF RAG Chatbot (matches your Colab pipeline exactly)

import streamlit as st
import torch
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login  

# 1. Create a sidebar (or main page) input for the token
with st.sidebar:
    st.subheader("Authentication")
    manual_token = st.text_input(
        "Enter your Hugging Face Token:", 
        type="password", 
        help="Get a token from huggingface.co/settings/tokens"
    )
    
    # Optional: Save button to trigger authentication
    auth_button = st.button("Authenticate")

# 2. Authenticate when the user provides the token
if manual_token:
    try:
        # Log in to the Hugging Face Hub
        login(token=manual_token)
        
        # Set environment variable for deep integrations (like LangChain)
        os.environ["HF_TOKEN"] = manual_token
        
        st.sidebar.success("Successfully authenticated!")
        
    except Exception as e:
        st.sidebar.error(f"Authentication failed: {e}")
else:
    st.sidebar.warning("Please enter your Hugging Face Token to load the model.")
    # Prevent the rest of your app from running until the token is provided
    st.stop()

st.set_page_config(page_title="PDF Chatbot (Qwen2.5-3B)", page_icon="📄")

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PROMPT_TEMPLATE = """<|im_start|>system
You are a helpful assistant that answers questions using ONLY the provided context from a PDF document. If the answer is not in the context, say "I don't know based on the document."<|im_end|>
<|im_start|>user
Context:
{context}
Question: {question}<|im_end|>
<|im_start|>assistant
"""


# ---------- Cached resources: built ONCE per server process, reused across every rerun ----------

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": device},
    )


@st.cache_resource(show_spinner="Loading Qwen2.5-1.5B model (this can take a minute)...")
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True,
        repetition_penalty=1.1,
    )
    return HuggingFacePipeline(pipeline=gen_pipeline)


@st.cache_resource(show_spinner="Processing PDF and building Chroma index...")
def build_vectorstore(pdf_path: str, _embeddings):
    # leading underscore on _embeddings tells Streamlit's cache not to
    # try to hash that object (it isn't hashable) — it's still used normally
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,  # overlapping technique
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    # unique persist dir per PDF so re-uploading a different file doesn't collide
    persist_dir = f"chroma_db_{os.path.basename(pdf_path)}"
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=_embeddings,
        persist_directory=persist_dir,
    )
    return vectorstore, len(chunks)


@st.cache_resource(show_spinner=False)
def build_qa_chain(_vectorstore, _llm, k=4):
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
    retriever = _vectorstore.as_retriever(search_kwargs={"k": k})
    return RetrievalQA.from_chain_type(
        llm=_llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )


def ask(qa_chain, question: str):
    result = qa_chain.invoke({"query": question})
    answer = result["result"].split("<|im_start|>assistant")[-1].strip()
    return answer, result["source_documents"]


# ---------- App UI ----------

st.title("📄 PDF Chatbot — Qwen2.5-3B")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    pdf_path = f"/tmp/{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    embeddings = load_embeddings()
    llm = load_llm()
    vectorstore, num_chunks = build_vectorstore(pdf_path, embeddings)
    qa_chain = build_qa_chain(vectorstore, llm)

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
            with st.spinner("Thinking..."):
                answer, sources = ask(qa_chain, question)
                st.write(answer)

                with st.expander("Sources"):
                    for doc in sources:
                        page = doc.metadata.get("page", "?")
                        st.markdown(f"**Page {page}:** {doc.page_content[:200]}...")

        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Upload a PDF to start chatting.")
