import torch
import evaluate
import re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np

MERGED_DIR = "NLP/llama_email_merged_90_10_final" # final merged model used for evaluation
EVAL_FILE  = "NLP/eval_10.jsonl" # Evaluation dataset file

# Generation and batching configuration
MAX_NEW_TOKENS = 128         
MAX_INPUT_LEN  = 512          
BATCH_SIZE = 8                
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR)

# Load the model in evaluation mode
model = AutoModelForCausalLM.from_pretrained(
    MERGED_DIR,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)
model.eval()


eval_ds = load_dataset("json", data_files=EVAL_FILE)["train"]

# Load evaluation metrics
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

summary_preds, summary_refs = [], []
reply_preds, reply_refs     = [], []


def extract_prompt_and_gold(text):
    inst_match = re.search(r"\[INST\](.*?)\[\/INST\]", text, re.S)
    gold_match = re.split(r"\[\/INST\]", text)

    prompt = inst_match.group(1).strip() if inst_match else ""
    gold   = gold_match[1].replace("</s>", "").strip() if len(gold_match) > 1 else ""

    return prompt, gold


prompts, golds, types = [], [], []

for row in eval_ds:
    p, g = extract_prompt_and_gold(row["text"])
    if p and g:
        prompts.append(p)
        golds.append(g)
        types.append(row["type"])


for i in tqdm(range(0, len(prompts), BATCH_SIZE)):
    batch_prompts = prompts[i:i+BATCH_SIZE]
    batch_golds   = golds[i:i+BATCH_SIZE]
    batch_types   = types[i:i+BATCH_SIZE]

    inputs = tokenizer(
        batch_prompts,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LEN,
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )

    batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    for pred, gold, t in zip(batch_preds, batch_golds, batch_types):
        if t == "summary":
            summary_preds.append(pred.strip())
            summary_refs.append(gold.strip())
        else:
            reply_preds.append(pred.strip())
            reply_refs.append(gold.strip())

#  ROUGE scores only for summaries
summary_rouge = rouge.compute(
    predictions=summary_preds,
    references=summary_refs
)

#BERTScore for summaries
summary_bert = bertscore.compute(
    predictions=summary_preds,
    references=summary_refs,
    model_type="roberta-large"   
)

#BERTScore for replies
reply_bert = bertscore.compute(
    predictions=reply_preds,
    references=reply_refs,
    model_type="roberta-large"
)


print("\n SUMMARY EVALUATION \n")
print(f"Samples: {len(summary_preds)}")
print(f"ROUGE-1: {summary_rouge['rouge1']:.4f}")
print(f"ROUGE-2: {summary_rouge['rouge2']:.4f}")
print(f"ROUGE-L: {summary_rouge['rougeL']:.4f}")
print(f"BERTScore F1: {np.mean(summary_bert['f1']):.4f}")

print("\n REPLY EVALUATION \n")
print(f"Samples: {len(reply_preds)}")
print(f"BERTScore Precision: {np.mean(reply_bert['precision']):.4f}")
print(f"BERTScore Recall:    {np.mean(reply_bert['recall']):.4f}")
print(f"BERTScore F1:        {np.mean(reply_bert['f1']):.4f}")
