# Customer Support AI Assistant
### TinyLlama Fine-tuning + RAG + Django

---

## What this project does

User types a question → Django receives it → LangChain retrieves
relevant context from ChromaDB → Fine-tuned TinyLlama generates answer → Shown in chat UI

---

## Full Project Structure

```
customer_support_ai/
├── manage.py                          ← Run Django from here
├── requirements.txt                   ← All libraries
├── colab_finetune.py                  ← Fine-tuning code (run in Google Colab)
├── tinyllama-finetuned/               ← Downloaded from Colab (after training)
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── tokenizer files...
├── data/
│   └── support_docs.pdf               ← Your knowledge base PDF
├── chroma_db/                         ← Auto-created vector database
├── customer_support_ai/
│   ├── settings.py                    ← Django settings
│   ├── urls.py                        ← Main URL router
│   └── wsgi.py
└── chatbot/
    ├── views.py                       ← Backend API logic
    ├── urls.py                        ← Chatbot URL routes
    ├── rag_engine.py                  ← AI brain (LangChain + ChromaDB)
    └── templates/chatbot/
        └── index.html                 ← Frontend chat UI
```

---

## STEP 1: Fine-tune TinyLlama (Google Colab)

Do this FIRST before setting up Django.

1. Go to https://colab.research.google.com
2. Click Runtime → Change runtime type → Select **T4 GPU** → Save
3. Open `colab_finetune.py` from this project
4. Copy each cell (between the triple quotes) into Colab cells
5. Run cells 1 to 8 in order
6. Cell 8 will download `tinyllama-finetuned.zip` to your computer
7. Unzip it and place the `tinyllama-finetuned/` folder in this project root

**Time needed:** About 15-20 minutes total

---

## STEP 2: Set up Django project

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# 3. Install all libraries
pip install -r requirements.txt

# 4. Run Django migrations
python manage.py migrate

# 5. Start the server
python manage.py runserver
```

Open browser: http://127.0.0.1:8000

---

## STEP 3: Upload your PDF (optional)

- In the chat UI, click "Choose PDF"
- Upload any customer support document
- The RAG pipeline will rebuild with your document

**If no PDF is uploaded:** The project uses built-in sample data about
refunds, orders, billing, and account issues — good enough for the interview demo.

---

## How each file works

### Frontend (index.html)
- Chat UI built with HTML + CSS + JavaScript
- User types question → JavaScript sends POST request to /ask/
- Receives JSON response → displays as chat bubble
- PDF upload → sends to /upload/ endpoint

### Backend (views.py)
- `index()` → renders the HTML page
- `ask()` → receives question as JSON → calls rag_engine → returns answer as JSON
- `upload_pdf()` → saves uploaded PDF → resets RAG pipeline

### AI Brain (rag_engine.py)
- Loads fine-tuned TinyLlama model
- Loads PDF → splits into 500-word chunks
- Creates embeddings using HuggingFace all-MiniLM-L6-v2
- Stores vectors in ChromaDB
- On each question: finds top 3 similar chunks → passes to LLM → returns answer

---

## Interview answers you must know

**Q: What is LoRA?**
LoRA stands for Low-Rank Adaptation. Instead of retraining all 1.1 billion
parameters of TinyLlama, LoRA adds small trainable matrices to specific attention
layers. Only 0.19% of parameters are trained, making it possible on a free GPU.

**Q: What is RAG?**
Retrieval-Augmented Generation. Instead of relying only on what the model learned
during training, RAG retrieves relevant information from a knowledge base at
runtime. This keeps answers accurate and up to date.

**Q: Why combine fine-tuning AND RAG?**
Fine-tuning teaches the model to respond in a specific customer support style and
tone. RAG provides accurate factual information from company documents. Together
they give both style and accuracy.

**Q: What is ChromaDB?**
A vector database that stores text as numerical vectors called embeddings.
When a question comes in, ChromaDB finds the most similar chunks using cosine
similarity and returns them to the LLM as context.

**Q: What is an embedding?**
A way to convert text into a list of numbers that captures its meaning. Similar
sentences have embeddings that are close together in vector space. I used
HuggingFace's all-MiniLM-L6-v2 model to create embeddings.

**Q: Explain your project architecture end to end.**
The user types a question in the Django frontend. JavaScript sends it as a POST
request to the /ask/ API endpoint. The Django view calls rag_engine.py which
retrieves the top 3 relevant chunks from ChromaDB using cosine similarity on
HuggingFace embeddings. These chunks plus the question are passed to the
fine-tuned TinyLlama model which generates a customer support response.
The answer is returned as JSON and displayed in the chat UI.
