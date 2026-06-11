# ============================================================
# GOOGLE COLAB FINE-TUNING NOTEBOOK
# Run this file cell by cell in Google Colab
# Runtime → Change runtime type → T4 GPU  (do this FIRST)
# ============================================================

# ── CELL 1: Install libraries ────────────────────────────────
# Paste this in Colab Cell 1 and run

"""
!pip install transformers datasets peft accelerate bitsandbytes trl -q
"""


# ── CELL 2: Imports ──────────────────────────────────────────
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import Dataset
import pandas as pd

print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
"""


# ── CELL 3: Create dataset ───────────────────────────────────
"""
data = [
    {"instruction": "How do I reset my password?",
     "response": "Go to the login page and click Forgot Password. Enter your email and follow the reset link sent to your inbox."},

    {"instruction": "I was charged twice for my order.",
     "response": "We apologize for the inconvenience. Please share your order ID and we will process a refund for the duplicate charge within 3-5 business days."},

    {"instruction": "How do I cancel my subscription?",
     "response": "Go to Account Settings then Subscription then Cancel Plan. Your access continues until the end of the billing period."},

    {"instruction": "My order has not arrived yet.",
     "response": "Orders arrive within 5-7 business days. Please share your order ID so we can track the current status for you."},

    {"instruction": "How do I update my billing information?",
     "response": "Go to Account Settings then Billing then Update Payment Method. Enter your new card details and click Save."},

    {"instruction": "I received a damaged product.",
     "response": "We are sorry to hear that. Please send a photo of the damaged item to support@company.com and we will send a replacement immediately."},

    {"instruction": "How do I contact customer support?",
     "response": "You can reach us via live chat, email at support@company.com, or call 1-800-123-4567 between 9am to 6pm Monday to Friday."},

    {"instruction": "Can I change my delivery address?",
     "response": "If your order has not shipped yet, go to Order History then select your Order then click Edit Address. Once shipped, address changes are not possible."},

    {"instruction": "What is your refund policy?",
     "response": "We offer full refunds within 30 days of purchase. Items must be unused and in original packaging. Refunds are processed within 5-7 business days."},

    {"instruction": "How do I track my order?",
     "response": "Once your order ships, you will receive a tracking number via email. You can also check Order History in your account for real-time tracking."},

    {"instruction": "My account has been locked.",
     "response": "Your account may be locked after multiple failed login attempts. Please click Forgot Password to reset it, or contact support@company.com for immediate help."},

    {"instruction": "How do I delete my account?",
     "response": "To delete your account, go to Account Settings then Privacy then Delete Account. Note that this action is permanent and cannot be undone."},

    {"instruction": "Can I get a replacement for my order?",
     "response": "Yes, we offer replacements for damaged, defective, or incorrect items within 30 days of delivery. Please contact support with your order ID and photos of the issue."},

    {"instruction": "Why was my payment declined?",
     "response": "Payments can be declined due to insufficient funds, incorrect card details, or bank restrictions. Please check your card details or try a different payment method."},

    {"instruction": "How long does delivery take?",
     "response": "Standard delivery takes 5-7 business days. Express delivery takes 1-2 business days and is available for an additional charge at checkout."},
]

def format_prompt(row):
    return f"### Instruction:\n{row['instruction']}\n\n### Response:\n{row['response']}"

df = pd.DataFrame(data)
df["text"] = df.apply(format_prompt, axis=1)
dataset = Dataset.from_pandas(df[["text"]])

print(f"Dataset size: {len(dataset)} samples")
print("\nSample entry:")
print(dataset[0]["text"])
"""


# ── CELL 4: Load TinyLlama with 4-bit quantization ───────────
"""
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# 4-bit quantization so it fits in free T4 GPU (16GB)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading model (this takes 2-3 minutes)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
model.config.use_cache = False

print("Model loaded successfully!")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
"""


# ── CELL 5: Apply LoRA ───────────────────────────────────────
"""
lora_config = LoraConfig(
    r=8,              # rank: size of the LoRA matrices (8 is standard)
    lora_alpha=16,    # scaling factor (usually 2x rank)
    target_modules=["q_proj", "v_proj"],  # which attention layers to adapt
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Output will look like:
# trainable params: 2,097,152 || all params: 1,102,000,000 || trainable%: 0.1903
# This means ONLY 0.19% of parameters are trained — that's the power of LoRA!
"""


# ── CELL 6: Train the model ──────────────────────────────────
"""
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="./tinyllama-finetuned",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=2,
        save_strategy="epoch",
        eval_strategy="no",
        report_to="none",
        optim="paged_adamw_8bit",
        packing=False,
    ),
)

print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning complete!")
"""


# ── CELL 7: Test your fine-tuned model ───────────────────────
"""
model.eval()

def test_model(question):
    prompt = f"### Instruction:\n{question}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the response part
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    return response

# Test it
print("Test 1:", test_model("How do I get a refund?"))
print()
print("Test 2:", test_model("My order did not arrive"))
print()
print("Test 3:", test_model("How do I reset my password?"))
"""


# ── CELL 8: Save and download the model ──────────────────────
"""
print("Saving fine-tuned model...")
model.save_pretrained("./tinyllama-finetuned")
tokenizer.save_pretrained("./tinyllama-finetuned")
print("Model saved!")

# Zip and download to your computer
import shutil
from google.colab import files

print("Zipping model files...")
shutil.make_archive("tinyllama-finetuned", "zip", "./tinyllama-finetuned")
print("Downloading...")
files.download("tinyllama-finetuned.zip")
print("Done! Unzip this file into your Django project folder.")
"""

# ============================================================
# AFTER DOWNLOADING:
# 1. Unzip tinyllama-finetuned.zip
# 2. Place the folder inside your Django project root
# 3. The rag_engine.py will automatically detect and load it
# ============================================================
