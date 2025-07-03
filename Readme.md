# NLP Project — Story Summarization, Question Answering, and Character Trait Extraction

## 📚 Overview

This project implements an NLP system **built from scratch** (without using any pre-trained language models or LLMs). The system is designed to perform three key tasks on narrative stories:

1. **Story Summarization** — Generate concise summaries of long narrative stories.
2. **Question Answering (QA)** — Answer questions based on the story context.
3. **Character and Trait Extraction** — Identify main characters and extract their traits or descriptive attributes.

---

## 🏗️ Project Structure

Project/
│
├── Src/
│ ├── train.py # Summarization model training
│ ├── train_qa.py # QA model training
│ ├── summarize.py # Generate summaries by using the model
│ ├── qa_generate.py # Generate QA predictions by using the model
| |
│ ├── character_extract.py # Character trait extraction
| |
│ ├── Utils/
│ ├── vocab.py # Vocabulary utilities
│ └── vocab_qa.py # Separate vocab for QA
| |
│ ├── models/
│ └── summarizer.py # Seq2Seq model with attention
│ |
├── Data/
│ ├── small_narrativeqa.json # Processed data for summarization
│ ├── train_narrativeqa_qa.json # Processed data for QA
│ ├── vocab.json # Saved vocab for summarization
│ ├── vocab_qa.json # Saved vocab for QA
│ ├── model_weights.pth # Saved summarization model weights
│ └── model_weights_qa.pth # Saved QA model weights
│
└── README.md # This file


------------Training------------------
Summarization
    python Src/train.py

Question Answering
    python Src/train_qa.py



------Generating outputs--------------
Generate summaries
    python Src/summarize.py

Generate QA predictions
    python Src/qa_generate.py

Extract characters and traits
    python Src/character_extract.py


Design Highlights
--No pre-trained models used —  models were trained from scratch on NarrativeQA subsets.
--Separate vocabularies for summarization and QA tasks to ensure independence.
--Seq2Seq with attention architecture for both summarization and QA, enabling the model to focus on relevant parts of long contexts.