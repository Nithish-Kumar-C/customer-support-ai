"""
Django Views — the backend layer

Two views:
1. index()  → renders the chat UI (GET /)
2. ask()    → receives question, returns AI answer (POST /ask/)
3. upload() → accepts PDF upload, rebuilds RAG pipeline (POST /upload/)
"""

import os
import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from .rag_engine import get_answer, build_rag_pipeline


# ── View 1: Render the chat UI ─────────────────────────────────────────────────
def index(request):
    return render(request, 'chatbot/index.html')


# ── View 2: Handle question from frontend ─────────────────────────────────────
@csrf_exempt
def ask(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST request required'}, status=405)

    try:
        body = json.loads(request.body)
        question = body.get('question', '').strip()

        if not question:
            return JsonResponse({'error': 'Question cannot be empty'}, status=400)

        if len(question) > 500:
            return JsonResponse({'error': 'Question too long (max 500 characters)'}, status=400)

        # Get answer from RAG pipeline
        answer = get_answer(question)

        return JsonResponse({
            'answer': answer,
            'status': 'success'
        })

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': f'Server error: {str(e)}'}, status=500)


# ── View 3: Upload PDF and rebuild knowledge base ─────────────────────────────
@csrf_exempt
def upload_pdf(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST request required'}, status=405)

    try:
        if 'pdf' not in request.FILES:
            return JsonResponse({'error': 'No PDF file provided'}, status=400)

        pdf_file = request.FILES['pdf']

        if not pdf_file.name.endswith('.pdf'):
            return JsonResponse({'error': 'File must be a PDF'}, status=400)

        # Save the uploaded PDF
        os.makedirs(os.path.dirname(settings.PDF_PATH), exist_ok=True)
        with open(settings.PDF_PATH, 'wb') as f:
            for chunk in pdf_file.chunks():
                f.write(chunk)

        # Rebuild RAG pipeline with new PDF
        from .rag_engine import qa_chain
        import chatbot.rag_engine as engine
        engine.qa_chain = None  # reset so it reloads with new PDF

        return JsonResponse({
            'message': f'PDF "{pdf_file.name}" uploaded successfully! Knowledge base updated.',
            'status': 'success'
        })

    except Exception as e:
        return JsonResponse({'error': f'Upload failed: {str(e)}'}, status=500)
