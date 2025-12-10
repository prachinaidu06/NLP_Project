import os, json, asyncio, random
import pandas as pd
from tqdm import tqdm
from openai import AsyncOpenAI



MODEL = "gpt-4.1-mini" #Model to generate synthetic replies
INPUT_CSV = "NLP/email_thread_details.csv"
OUTPUT_JSONL = "NLP/train_synthetic_replies.jsonl" # Output file to store synthetic replies
CHECKPOINT_FILE = "NLP/progress_checkpoint.json"

MAX_CHARS = 850
MAX_OUTPUT_TOKENS = 160
CONCURRENCY = 8

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))



def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE) as f:
                return int(json.load(f).get("last_idx", 0))
        except Exception:
            return 0
    return 0

def save_checkpoint(i):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"last_idx": i}, f)


def load_threads(csv_path):
    df = pd.read_csv(csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp")

    # Group all messages that belong to the same conversation thread
    threads = []
    for tid, grp in df.groupby("thread_id"):
        text = "\n".join(
            f"From: {r.get('from','')}\n"
            f"To: {r.get('to','')}\n"
            f"Subject: {r.get('subject','')}\n"
            f"Time: {r.get('timestamp','')}\n\n"
            f"{r.get('body','')}\n-----"
            for _, r in grp.iterrows()
        )
        threads.append(text[-MAX_CHARS:])
    return threads



#prompt that will be sent to the language model
def build_prompt(thread):
    return f"""
You are an expert email assistant.

Step 1: Infer the most appropriate instruction for this email.
Step 2: Generate a professional reply that strictly follows that inferred instruction.

Return ONLY valid JSON in this exact format:
{{
  "instruction": "...",
  "reply": "..."
}}

EMAIL THREAD:
{thread}
"""


# Generate a reply for a single email thread
async def generate_one(thread):
    try:
        resp = await client.responses.create(
            model=MODEL,
            input=build_prompt(thread),
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.4
        )

        text = resp.output_text.strip()
        data = json.loads(text)

        return {
            "task": "reply",
            "instruction": data.get("instruction", "").strip(),
            "input": thread,
            "output": data.get("reply", "").strip()
        }

    except Exception:
        return {
            "task": "reply",
            "instruction": "",
            "input": thread,
            "output": "Thank you for your email."
        }



async def run_batch(batch):
    tasks = [generate_one(t) for t in batch]
    return await asyncio.gather(*tasks)



async def main():
    threads = load_threads(INPUT_CSV)
    start = load_checkpoint()

    results = []
    batch = []

    for i in tqdm(range(start, len(threads)), desc="Generating HYBRID replies"):
        batch.append(threads[i])

        if len(batch) == CONCURRENCY:
            out = await run_batch(batch)
            results.extend(out)
            save_checkpoint(i)
            batch.clear()

    if batch:
        out = await run_batch(batch)
        results.extend(out)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for row in results:
            if len(row["output"]) > 8:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n Saved {len(results)} hybrid samples")

if __name__ == "__main__":
    asyncio.run(main())
