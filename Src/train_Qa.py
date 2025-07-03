import json
import torch
from torch.utils.data import DataLoader, Dataset
from Utils.vocab import build_vocab, text_to_indices, PAD, SOS, EOS
from models.summarizer import Seq2SeqWithAttention
from Utils.vocab import save_vocab
import torch.nn as nn
import torch.optim as optim
# from google.colab import files


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------
# Load QA data /content/train_narrativeqa_qa.json
# ----------------------
with open("/content/small_narrativeqa.json", "r", encoding="utf-8") as f:

# with open("/content/train_narrativeqa_qa.json", "r", encoding="utf-8") as f:
    data = json.load(f)
all_texts = []

cleaned_data = []
for sample in data:
    story_text = sample["story"]
    q_text = sample["question"]["text"]
    if sample["answers"] and sample["answers"][0]["text"].strip():
        ans_text = sample["answers"][0]["text"]
        all_texts.append(story_text)
        all_texts.append(q_text)
        all_texts.append(ans_text)
        cleaned_data.append(sample)  # Keep track of only valid samples
    else:
        continue

vocab = build_vocab(all_texts, max_vocab_size=10000)
# save_vocab(vocab, "Data/vocab_qa.json")
inv_vocab = {idx: word for word, idx in vocab.items()}

# ----------------------
# Dataset
# ----------------------
class QADataset(Dataset):
    def __init__(self, samples, vocab):
        self.samples = samples
        self.vocab = vocab

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
      story_text = self.samples[idx]["story"]
      question_text = self.samples[idx]["question"]["text"]
    
    # Combine story and question
      combined_input = story_text + " " + question_text

    # Prepare answer text
      answers_list = self.samples[idx]["answers"]
    # Choose the first answer text (adjust if needed)
      answer_text = answers_list[0]["text"] if isinstance(answers_list, list) and len(answers_list) > 0 else ""

    # Tokenize and convert to indices
      input_idx = text_to_indices(combined_input, self.vocab, max_len=600)  # adjust max_len if needed
      answer_idx = [self.vocab[SOS]] + text_to_indices(answer_text, self.vocab, max_len=100) + [self.vocab[EOS]]


      # print("Answer text:", answer_text)
      # print("Tokens:", word_tokenize(answer_text.lower()))
      # print("Indices:", answer_idx)
      # print("Decoded:", " ".join([inv_vocab.get(idx, "<UNK>") for idx in answer_idx]))
      # print("-" * 50)
      return torch.tensor(input_idx), torch.tensor(answer_idx)


    # def __getitem__(self, idx):
    #     question = self.samples[idx]["question"]
    #     answers = self.samples[idx]["answers"]

    #     # Choose the first answer text as target
    #     answer_text = answers[0]["text"]

    #     # question_idx = text_to_indices(question, self.vocab, max_len=100)
    #     # answer_idx = [self.vocab[SOS]] + text_to_indices(answer_text, self.vocab, max_len=100) + [self.vocab[EOS]]

    #     # return torch.tensor(question_idx), torch.tensor(answer_idx)
    #     question_idx = text_to_indices(question_text, self.vocab, max_len=100)
    #     answer_idx = [self.vocab[SOS]] + text_to_indices(answer_text, self.vocab, max_len=100) + [self.vocab[EOS]]

    #     return torch.tensor(question_idx), torch.tensor(answer_idx)

def collate_fn(batch):
    questions, answers = zip(*batch)
    questions_padded = nn.utils.rnn.pad_sequence(questions, batch_first=True, padding_value=vocab[PAD])
    answers_padded = nn.utils.rnn.pad_sequence(answers, batch_first=True, padding_value=vocab[PAD])
    return questions_padded, answers_padded

dataset = QADataset(data, vocab)
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# ----------------------
# Model
# ----------------------
model = Seq2SeqWithAttention(len(vocab), embed_dim=128, hidden_dim=256, pad_idx=vocab[PAD]).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=vocab[PAD])
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for questions, answers in loader:
        questions, answers = questions.to(DEVICE), answers.to(DEVICE)

        optimizer.zero_grad()
        output = model(questions, answers, teacher_forcing_ratio=0.3)

        output = output[:, 1:].reshape(-1, len(vocab))
        target = answers[:, 1:].reshape(-1)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_loss:.4f}")

# ----------------------
# Save
# ----------------------
torch.save(model.state_dict(), "model_weights_qa.pth")
# files.download('/content/model_weights_qa.pth')
print(" Model weights saved to model_weights_qa.pth")
save_vocab(vocab, "vocab_qa.json")
# files.download('/content/vocab_qa.json')
print(" Vocab saved to Data/vocab_qa.json")
