import json
import pandas as pd
import random


REPLY_JSONL   = "NLP/train_synthetic_replies.jsonl"
SUMMARY_CSV   = "NLP/email_thread_summaries.csv"
THREAD_CSV    = "NLP/email_thread_details.csv"  
OUTPUT_FILE   = "NLP/llama_train_merged_fixed.jsonl" #Final merged output file for LLaMA training

final_data = []


threads_df = pd.read_csv(THREAD_CSV)

#helps quickly retrieve the original thread for each summary
thread_lookup = {
    int(row["thread_id"]): str(row["body"]).strip()
    for _, row in threads_df.iterrows()
    if not pd.isna(row["body"])
}

print(f" Loaded {len(thread_lookup)} email threads")


with open(REPLY_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)

        instruction = row["instruction"].strip()
        input_text  = row["input"].strip()
        output_text = row["output"].strip()

        prompt = f"""<s>[INST]
{instruction}

THREAD:
{input_text}
[/INST]
"""

        final_data.append({
            "text": prompt + output_text + "</s>",
            "type": "reply"
        })

print(f"Loaded {len(final_data)} reply samples")

#helps retrieve the original thread for each summary
summary_df = pd.read_csv(SUMMARY_CSV)

summary_added = 0
summary_skipped = 0

for _, row in summary_df.iterrows():
    try:
        tid = int(row["thread_id"])
        summary = str(row["summary"]).strip()

        
        if tid not in thread_lookup or not summary:
            summary_skipped += 1
            continue

        thread_text = thread_lookup[tid]

        # Build the standard summary instruction prompt
        prompt = f"""<s>[INST]
Summarize the following email thread in 3–4 sentences.

THREAD:
{thread_text}
[/INST]
"""

        final_data.append({
            "text": prompt + summary + "</s>",
            "type": "summary"
        })

        summary_added += 1

    except Exception:
        summary_skipped += 1

print(f" Added {summary_added} aligned summary samples")
print(f" Skipped {summary_skipped} broken summary rows")


random.shuffle(final_data)


# Write the final merged dataset to a JSONL file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for row in final_data:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("\n FINAL MERGED DATASET READY")
print(f"\n Total samples: {len(final_data)}")
print(f"\n File saved to: {OUTPUT_FILE}")
