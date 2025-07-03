from datasets import load_dataset
import json

subset_size = 2000
truncate_words = 1000

# --------------------------------------
print(f"Loading NarrativeQA (streaming)...")
dataset_stream = load_dataset("deepmind/narrativeqa", split="train", streaming=True)

dataset_all = []
deduped_summaries = []
unique_story_summary_set = set()

for i, sample in enumerate(dataset_stream):
    if i >= subset_size:
        break

    text = sample["document"]["text"]
    summary = sample["document"]["summary"]
    question = sample["question"]
    answers = sample["answers"]

    # Truncate story text
    short_text = " ".join(text.split()[:truncate_words])

    # Handle summary (in NarrativeQA it might be a dict with "text")
    summary_text = summary["text"] if isinstance(summary, dict) else summary

    # ---------- Collect full QA sample ----------
    record = {
        "story": short_text,
        "summary": summary_text,
        "question": question,
        "answers": answers
    }
    dataset_all.append(record)

    # ---------- Deduplicate for summary ----------
    story_summary_key = short_text
    if story_summary_key not in unique_story_summary_set:
        deduped_summaries.append({
            "story": short_text,
            "summary": summary_text
        })
        unique_story_summary_set.add(story_summary_key)

print(f" Collected {len(dataset_all)} QA samples.")
print(f" Collected {len(deduped_summaries)} unique story-summary pairs.")

# Save to JSON


output_file_qa = "Data/train_narrativeqa_qa.json"
with open(output_file_qa, "w", encoding="utf-8") as f:
    json.dump(dataset_all, f, indent=2, ensure_ascii=False)



# deduped_summaries.sort(key=lambda x: x["story"])  # Sort by story text for consistency

output_file_sum = "Data/train_narrativeqa_summary.json"
with open(output_file_sum, "w", encoding="utf-8") as f:
    json.dump(deduped_summaries, f, indent=2, ensure_ascii=False)

print(f" Saved QA dataset to {output_file_qa}.")
print(f" Saved summary dataset to {output_file_sum}.")





# The code below was used to create the small_narrativeqa.json file.
#It is commented out to avoid re-running it unnecessarily. 

# ----------------------------------------------------------------------------------------------------------------------------------

# from datasets import load_dataset
# import json

# subset_size = 200
# truncate_words = 1000  

# # --------------------------------------
# print(f"Loading NarrativeQA (streaming)...")
# dataset_stream = load_dataset("deepmind/narrativeqa", split="train", streaming=True)
# # dataset_val = load_dataset("deepmind/narrativeqa", split="validation", streaming=True)
# # dataset_test = load_dataset("deepmind/narrativeqa", split="test", streaming=True)


# dataset = []
# for i, sample in enumerate(dataset_stream):
#     if i >= subset_size:
#         break

#     text = sample["document"]["text"]
#     summary = sample["document"]["summary"]
#     question = sample["question"]
#     answers = sample["answers"]

#     # Truncate long stories
#     short_text = " ".join(text.split()[:truncate_words])

#     record = {
#         "story": short_text,
#         "summary": summary,
#         "question": question,
#         "answers": answers
#     }

#     dataset.append(record)

# print(f" Collected {len(dataset)} samples.")

# # --------------------------------------
# # Save to JSON
# # --------------------------------------
# output_file = f"Data/train_narrativeqa.json"
# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(dataset, f, indent=2, ensure_ascii=False)

# print(f"Saved dataset to {output_file}.")














