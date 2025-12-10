
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)


import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

#final merged LoRA model
MODEL_PATH = "NLP/llama_email_merged_90_10_final"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Streamlit page 
st.set_page_config(page_title="Email Summary & Reply Assistant", layout="wide")
st.title("Email Summary & Reply Assistant")



@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=False,
        fix_mistral_regex=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    ).to(DEVICE)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.generation_config.do_sample = False

    model.eval()
    return tokenizer, model


with st.spinner("Loading model..."):
    tokenizer, model = load_model()


def build_prompt(thread_text: str, task_type: str, user_instruction: str) -> str:

    if task_type.lower() == "summary":
        base = (
            "You are an expert executive assistant. Produce ONLY a summary. "
            "Use ONLY information explicitly present in the email. "
            "Do NOT assume, infer, or invent details."
        )
    else:
        base = (
            "You are a professional email writer. Produce ONLY a reply. "
            "Do NOT summarize the email."
        )

    full_instruction = base + "\n" + user_instruction.strip()

    prompt = f"""<s>[INST]
{full_instruction}

THREAD:
{thread_text}
[/INST]
"""
    return prompt


def generate_email_output(
    thread_text: str,
    task_type: str = "summary",
    user_instruction: str = "",
):

    if not user_instruction.strip():
        if task_type.lower() == "summary":
            user_instruction = "Summarize the email in 2 concise sentences."
        else:
            user_instruction = "Write a polite professional reply."

    prompt = build_prompt(thread_text, task_type, user_instruction)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        add_special_tokens=True,
    ).to(model.device)

    with torch.no_grad():

        
        if task_type.lower() == "summary":
            max_tokens = 80
        else:
            max_tokens = 140

        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

 
    if "[/INST]" in decoded:
        decoded = decoded.split("[/INST]", 1)[1]

    decoded = decoded.strip()

    if "</s>" in decoded:
        decoded = decoded.split("</s>", 1)[0].strip()

    stop_markers = [
        "\nFrom:", "\nTo:", "\nSent:", "\nSubject:",
        "\n-----", "\nOriginal Message", "\n> From:"
    ]
    for marker in stop_markers:
        if marker in decoded:
            decoded = decoded.split(marker, 1)[0].strip()

    if task_type.lower() == "summary":
        forbidden = ("Dear ", "Hi ", "Hello ", "Best,", "Regards,", "Sincerely")
        for f in forbidden:
            if decoded.startswith(f):
                decoded = decoded.split("\n", 1)[-1].strip()
                break

    return decoded



col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Paste Email Thread")
    email_text = st.text_area(
        "",
        height=280,
        placeholder="Paste the email thread here..."
    )

with col2:
    st.subheader("Select Task")
    task_type = st.radio(
        "",
        ["summary", "reply"],
        horizontal=True
    )

    st.subheader("Custom Instruction (Optional)")
    user_instruction = st.text_input(
        "",
        placeholder="e.g., Summarize in 1 sentence / Reply saying I’m available at 2 PM"
    )

    st.markdown("**Examples:**")
    st.markdown("- *Summarize in 2 bullet points*")
    st.markdown("- *Reply that I will be available at 2 PM today*")
    st.markdown("- *Write a firm professional reply*")


# ================================
# GENERATE BUTTON
# ================================
if st.button("Generate Output"):
    if not email_text.strip():
        st.warning("Please paste an email thread.")
    else:
        with st.spinner("Generating output..."):
            result = generate_email_output(
                email_text.strip(),
                task_type=task_type,
                user_instruction=user_instruction
            )

        st.subheader("Model Output")
        st.text_area("", result, height=240)


