
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# final merged LoRA model
MODEL_PATH = "NLP/llama_email_merged_90_10_final"
#GPU if available, otherwise falls back to CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def load_model():
    print(f"Loading model from: {MODEL_PATH}")

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


tokenizer, model = load_model()



def build_prompt(thread_text: str, task_type: str) -> str:

    if task_type.lower() == "summary":
        instruction = (
            "Summarize the following email thread in 2 concise sentences. "
            "Use ONLY information explicitly present in the email. "
            "Do NOT assume, infer, or invent any details."
        )
    elif task_type.lower() == "reply":
        instruction = (
            "Write a polite, professional reply to the following email thread."
        )
    else:
        raise ValueError("task_type must be 'summary' or 'reply'.")

    prompt = f"""<s>[INST]
{instruction}

THREAD:
{thread_text}
[/INST]
"""
    return prompt



def generate_email_output(
    thread_text: str,
    task_type: str = "summary",
) -> str:

    prompt = build_prompt(thread_text, task_type)

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



if __name__ == "__main__":

    sample_thread = """
Hi Paul,

What time should I set up a meeting for the revenue discussion? I am available at 2pm - 5pm on Thursday and 1pm to 2pm on Friday.

Thanks,
Sam
"""

    
    print("SUMMARY \n")
   
    summary = generate_email_output(sample_thread, "summary")
    print(summary )
    print()

  
    print("REPLY \n")
   
    reply = generate_email_output(sample_thread, "reply")
    print(reply)
