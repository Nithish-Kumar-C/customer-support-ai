"""
AI Brain — combines fine-tuned TinyLlama + RAG pipeline

Flow:
1. Load fine-tuned TinyLlama model (trained in Google Colab)
2. Load PDF knowledge base documents
3. Split into chunks → create embeddings → store in ChromaDB
4. On every question: retrieve relevant chunks → pass to LLM → return answer
"""

import os
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ── Global variables (loaded once when server starts) ──────────────────────────
qa_chain = None
vectorstore = None


# ── STEP 1: Load the fine-tuned TinyLlama model ───────────────────────────────
def load_llm(model_path=None):
    """
    Try to load fine-tuned model first.
    If not found, fall back to Ollama TinyLlama (base model).
    """
    if model_path and os.path.exists(model_path):
        print(f"[AI] Loading fine-tuned model from: {model_path}")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            from langchain_community.llms import HuggingFacePipeline

            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                repetition_penalty=1.1,
            )

            llm = HuggingFacePipeline(pipeline=pipe)
            print("[AI] Fine-tuned model loaded successfully!")
            return llm

        except Exception as e:
            print(f"[AI] Fine-tuned model load failed: {e}")
            print("[AI] Falling back to Ollama...")

    # Fallback: use Ollama (make sure Ollama is running: ollama serve)
    print("[AI] Loading TinyLlama via Ollama (base model)...")
    from langchain_community.llms import Ollama
    return Ollama(model="tinyllama")


# ── STEP 2: Build RAG pipeline ─────────────────────────────────────────────────
def build_rag_pipeline(pdf_path, chroma_path, model_path=None):
    """
    Full RAG pipeline:
    PDF → chunks → embeddings → ChromaDB → retriever → LLM chain
    """
    global qa_chain, vectorstore

    # 2a. Load PDF
    print(f"[RAG] Loading PDF: {pdf_path}")
    if not os.path.exists(pdf_path):
        # Create a sample document if no PDF uploaded yet
        print("[RAG] No PDF found — using sample customer support data")
        from langchain.schema import Document
        sample_docs = [
            Document(page_content="To reset your password, go to the login page and click Forgot Password. Enter your registered email and follow the link sent to your inbox."),
            Document(page_content="For refund requests, contact support within 30 days of purchase. Refunds are processed within 5-7 business days to your original payment method."),
            Document(page_content="To cancel your subscription, go to Account Settings, then Subscription, then click Cancel Plan. Your access continues until the billing period ends."),
            Document(page_content="If you were charged twice, please share your order ID with our support team. We will investigate and refund the duplicate charge within 3-5 business days."),
            Document(page_content="Delivery typically takes 5-7 business days. You will receive a tracking number via email once your order is shipped."),
            Document(page_content="To update billing information, go to Account Settings, then Billing, then Update Payment Method. Enter your new card details and click Save."),
            Document(page_content="For damaged products, send photos to support@company.com. We will arrange a replacement or refund within 48 hours."),
            Document(page_content="Our customer support is available Monday to Friday, 9am to 6pm. You can reach us via live chat, email at support@company.com, or call 1-800-123-4567."),
        ]
        chunks = sample_docs
    else:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"[RAG] Loaded {len(documents)} pages from PDF")

        # 2b. Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = splitter.split_documents(documents)
        print(f"[RAG] Split into {len(chunks)} chunks")

    # 2c. Create embeddings using HuggingFace (free, no API key needed)
    print("[RAG] Creating embeddings with HuggingFace all-MiniLM-L6-v2...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )

    # 2d. Store in ChromaDB (vector database)
    print(f"[RAG] Storing vectors in ChromaDB at: {chroma_path}")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_path
    )
    print(f"[RAG] ChromaDB ready with {vectorstore._collection.count()} vectors")

    # 2e. Load LLM
    llm = load_llm(model_path)

    # 2f. Create custom prompt template
    prompt_template = """You are a helpful customer support assistant.
Use the following context to answer the customer's question clearly and politely.
If you don't know the answer, say "Please contact our support team for further help."

Context:
{context}

Customer Question: {question}

Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # 2g. Build RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}   # retrieve top 3 relevant chunks
        ),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=False
    )

    print("[AI] Full RAG pipeline ready!")
    return qa_chain


# ── STEP 3: Answer a question ──────────────────────────────────────────────────
def get_answer(question, pdf_path=None, chroma_path=None, model_path=None):
    """
    Main function called by Django view.
    Loads the chain once, then reuses it for every question.
    """
    global qa_chain

    # Load pipeline on first request
    if qa_chain is None:
        from django.conf import settings
        pdf_path = pdf_path or settings.PDF_PATH
        chroma_path = chroma_path or settings.CHROMA_DB_PATH
        model_path = model_path or settings.FINETUNED_MODEL_PATH
        build_rag_pipeline(pdf_path, chroma_path, model_path)

    # Get answer
    try:
        result = qa_chain({"query": question})
        answer = result.get("result", "Sorry, I could not find an answer.")
        # Clean up the answer (remove "Answer:" prefix if model repeats it)
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        return answer
    except Exception as e:
        print(f"[AI] Error getting answer: {e}")
        return "Sorry, something went wrong. Please try again."
