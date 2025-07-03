import json
import torch
from Utils.vocab import  text_to_indices, PAD, SOS, EOS, load_vocab
from models.summarizer import Seq2SeqWithAttention

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open("Data/test_narrativeqa.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Load vocab

vocab = load_vocab("Data/vocab_qa.json")
inv_vocab = {v: k for k, v in vocab.items()}
VOCAB_SIZE = len(vocab)

print("Vocab size (qa_generate.py):", VOCAB_SIZE)

# ----------------------
# Load model
# ----------------------
model = Seq2SeqWithAttention(VOCAB_SIZE, embed_dim=128, hidden_dim=256, pad_idx=vocab[PAD]).to(DEVICE)
model.load_state_dict(torch.load("Data/model_weights_qa.pth", map_location=DEVICE))
model.eval()

# Generate answer function
def generate_answer(model, question_tensor, max_len=30):
    with torch.no_grad():
        embedded = model.embedding(question_tensor.unsqueeze(0).to(DEVICE))
        encoder_outputs, (h_n, c_n) = model.encoder(embedded)

        hidden = (h_n.transpose(0, 1).reshape(1, -1).unsqueeze(0),
                  c_n.transpose(0, 1).reshape(1, -1).unsqueeze(0))

        decoder_input = torch.tensor([[vocab[SOS]]], device=DEVICE)
        answer_indices = []

        for _ in range(max_len):
            emb_dec = model.embedding(decoder_input)
            repeated_hidden = hidden[0].transpose(0, 1).repeat(1, encoder_outputs.size(1), 1)
            attn_input = torch.cat((repeated_hidden, encoder_outputs), dim=2)
            attn_energy = torch.tanh(model.attn(attn_input))
            attn_scores = torch.sum(attn_energy, dim=2)

            attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(1)
            context = torch.bmm(attn_weights, encoder_outputs)

            rnn_input = torch.cat((emb_dec, context), dim=2)
            output, hidden = model.decoder(rnn_input, hidden)
            pred = model.out(output.squeeze(1))
            top1 = pred.argmax(1).item()

            if top1 == vocab[EOS] or top1 == vocab[PAD]:
                break

            answer_indices.append(top1)
            decoder_input = torch.tensor([[top1]], device=DEVICE)

        return answer_indices


# Loop through first few samples (e.g., 5)
# print("=" * 80)
# 
# 
#     
for i, sample in enumerate(data[:5]):
    question_text = sample["question"]["text"]
    question_tensor = torch.tensor(text_to_indices(question_text, vocab, max_len=100))

    gen_indices = generate_answer(model, question_tensor, max_len=50)
    gen_words = [inv_vocab.get(idx, "<UNK>") for idx in gen_indices]
    generated_answer = " ".join(gen_words)

    true_answer = sample["answers"][0]["text"] if sample["answers"] else "No true answer provided"

    print(f"\n Question #{i+1}:")
    print(question_text, "\n")

    print(" True answer:")
    print(true_answer[:500], "...\n")

    print(" Generated answer:")
    print(generated_answer)
    print("=" * 80)














# import json
# import torch
# from Utils.vocab import build_vocab, text_to_indices, PAD, SOS, EOS
# from models.summarizer import Seq2SeqWithAttention
# from Utils.vocab import load_vocab

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load test dataset
# with open("Data/test_narrativeqa.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# # Load vocab
# vocab = load_vocab("Data/vocab.json")
# inv_vocab = {v: k for k, v in vocab.items()}
# VOCAB_SIZE = len(vocab)

# print("Vocab size (qa.py):", VOCAB_SIZE)

# # Load model
# model = Seq2SeqWithAttention(VOCAB_SIZE, embed_dim=128, hidden_dim=256, pad_idx=vocab[PAD]).to(DEVICE)
# model.load_state_dict(torch.load("Data/model_weights.pth", map_location=DEVICE))
# model.eval()

# def generate_answer(model, input_text, max_len=100):
#     input_indices = text_to_indices(input_text, vocab, max_len=500)
#     story_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(DEVICE)

#     with torch.no_grad():
#         embedded = model.embedding(story_tensor)
#         encoder_outputs, (h_n, c_n) = model.encoder(embedded)

#         hidden = (h_n.transpose(0, 1).reshape(1, -1).unsqueeze(0),
#                   c_n.transpose(0, 1).reshape(1, -1).unsqueeze(0))

#         decoder_input = torch.tensor([[vocab[SOS]]], device=DEVICE)
#         output_indices = []

#         for _ in range(max_len):
#             emb_dec = model.embedding(decoder_input)

#             repeated_hidden = hidden[0].transpose(0, 1).repeat(1, encoder_outputs.size(1), 1)

#             attn_input = torch.cat((repeated_hidden, encoder_outputs), dim=2)
#             attn_energy = torch.tanh(model.attn(attn_input))
#             attn_scores = torch.sum(attn_energy, dim=2)

#             attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(1)
#             context = torch.bmm(attn_weights, encoder_outputs)

#             rnn_input = torch.cat((emb_dec, context), dim=2)
#             output, hidden = model.decoder(rnn_input, hidden)
#             pred = model.out(output.squeeze(1))
#             top1 = pred.argmax(1).item()

#             if top1 == vocab[EOS] or top1 == vocab[PAD]:
#                 break

#             output_indices.append(top1)
#             decoder_input = torch.tensor([[top1]], device=DEVICE)

#         words = [inv_vocab.get(idx, "<UNK>") for idx in output_indices]
#         return " ".join(words)

# # Loop through first few samples (e.g., 10)
# for idx, sample in enumerate(data[:10]):
#     story_text = sample["story"]
#     question_text = sample["question"]["text"]
#     true_answers = [ans["text"] for ans in sample["answers"]]

#     # Condition on story + question
#     input_text = story_text + " " + question_text
#     generated_ans = generate_answer(model, input_text)

#     print("=" * 80)
#     print(f"📖 Story snippet #{idx+1}:\n{story_text[:500]}...\n")
#     print(f"❓ Question:\n{question_text}\n")
#     print(f"✅ True answers:\n{true_answers}\n")
#     print(f"🤖 Generated answer:\n{generated_ans}\n")

# print("=" * 80)
# print("✅ QA generation complete!")











# import json
# import re
# from rank_bm25 import BM25Okapi
# import spacy
# import nltk

# # Load spacy model (only for tokenization and entity detection)
# nlp = spacy.load("en_core_web_sm")

# # ------------------------
# # Load dataset
# # ------------------------
# with open("Data/test_narrativeqa.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# # ------------------------
# # Utility functions
# # ------------------------

# def clean_text(text):
#     text = re.sub(r"<.*?>", "", text)  # remove HTML tags
#     text = re.sub(r"\b\w+\s*-\s*", "", text)  # remove speaker labels
#     return text

# def split_sentences(text):
#     return nltk.sent_tokenize(text)

# def find_entities(text):
#     doc = nlp(text)
#     return set([ent.text for ent in doc.ents])

# def find_answer(story, question):
#     story_clean = clean_text(story)
#     sentences = split_sentences(story_clean)
#     tokenized_sents = [s.lower().split() for s in sentences]
    
#     bm25 = BM25Okapi(tokenized_sents)
#     tokenized_question = question.lower().split()
#     scores = bm25.get_scores(tokenized_question)

#     question_ents = find_entities(question)

#     # Boost scores by entity overlap
#     boosted_scores = []
#     for sent, score in zip(sentences, scores):
#         sent_ents = find_entities(sent)
#         if question_ents & sent_ents:
#             score += 1.0  # boost factor
#         boosted_scores.append(score)

#     best_index = boosted_scores.index(max(boosted_scores))
#     return sentences[best_index]


# # ------------------------
# # Run on a few samples
# # ------------------------
# for sample in data[:5]:
#     story = sample["story"]
#     # Fix for question
#     question = sample["question"]["text"]
#     true_answers = sample["answers"]

#     predicted = find_answer(story, question)

#     print("📖 Story snippet:")
#     print(story[:500], "...")

#     print("\n❓ Question:")
#     print(question)

#     print("\n✅ True answers:")
#     print(true_answers)

#     print("\n🤖 Predicted answer:")
#     print(predicted)

#     print("\n" + "="*80 + "\n")
