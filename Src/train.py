import json
import torch
from torch.utils.data import DataLoader, Dataset
from Utils.vocab import build_vocab, text_to_indices, PAD, SOS, EOS
from models.summarizer import Seq2SeqWithAttention
from Utils.vocab import save_vocab, load_vocab
from Utils.vocab import word_tokenize
import torch.nn as nn
import torch.optim as optim




DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data /content/small_narrativeqa.json
with open("Data/small_narrativeqa.json", "r", encoding="utf-8") as f:
    data = json.load(f)

all_texts = []
for sample in data:
    story_text = sample["story"]
    summary_text = sample["summary"]["text"]
    all_texts.append(story_text)
    all_texts.append(summary_text)
    
vocab = build_vocab(all_texts, max_vocab_size=10000)
inv_vocab = {idx: word for word, idx in vocab.items()}


# vocab, inv_vocab = build_vocab(data, max_vocab_size=10000)
# save_vocab(vocab, "Data/vocab.json")

class StoryDataset(Dataset):
    def __init__(self, samples, vocab):
        self.samples = samples
        self.vocab = vocab

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        story = self.samples[idx]["story"]
        summary = self.samples[idx]["summary"]["text"]

        story_idx = text_to_indices(story, self.vocab, max_len=500)
        summary_idx = [self.vocab[SOS]] + text_to_indices(summary, self.vocab, max_len=100) + [self.vocab[EOS]]
        return torch.tensor(story_idx), torch.tensor(summary_idx)
    

    
#     def __getitem__(self, idx):
#         story = self.samples[idx]["story"]
#         summary = self.samples[idx]["summary"]["text"]

#         story_idx = text_to_indices(story, self.vocab, max_len=500)
#         summary_idx = [self.vocab[SOS]] + text_to_indices(summary, self.vocab, max_len=100) + [self.vocab[EOS]]
#         return torch.tensor(story_idx), torch.tensor(summary_idx)




def collate_fn(batch):
    stories, summaries = zip(*batch)
    stories_padded = nn.utils.rnn.pad_sequence(stories, batch_first=True, padding_value=vocab[PAD])
    summaries_padded = nn.utils.rnn.pad_sequence(summaries, batch_first=True, padding_value=vocab[PAD])
    return stories_padded, summaries_padded

dataset = StoryDataset(data, vocab)
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

model = Seq2SeqWithAttention(len(vocab), embed_dim=128, hidden_dim=256, pad_idx=vocab[PAD]).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=vocab[PAD])
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 15

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for stories, summaries in loader:
        stories, summaries = stories.to(DEVICE), summaries.to(DEVICE)

        optimizer.zero_grad()
        output = model(stories, summaries)

        output = output[:, 1:].reshape(-1, len(vocab))
        target = summaries[:, 1:].reshape(-1)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_loss:.4f}")


torch.save(model.state_dict(), "Data/model_weights2.pth")
# files.download('/content/model_weights.pth')
print(" Model weights saved to model_weights.pth")
save_vocab(vocab, "Data/vocab2.json")
# files.download('/content/vocab.json')


print(" Model weights saved to model_weights.pth")



