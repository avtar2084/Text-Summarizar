import json
import torch
from Utils.vocab import build_vocab, text_to_indices, PAD, SOS, EOS, load_vocab
from models.summarizer import Seq2SeqWithAttention

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------
# Load data
# ----------------------
with open("Data/small_narrativeqa.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# ----------------------
# Load vocab
# ----------------------
vocab = load_vocab("Data/vocab.json")
inv_vocab = {v: k for k, v in vocab.items()}
VOCAB_SIZE = len(vocab)

print("Vocab size (summarize.py):", VOCAB_SIZE)

# ----------------------
# Load model
# ----------------------
model = Seq2SeqWithAttention(VOCAB_SIZE, embed_dim=128, hidden_dim=256, pad_idx=vocab[PAD]).to(DEVICE)
model.load_state_dict(torch.load("Data/model_weights.pth", map_location=DEVICE))
model.eval()
# model = Seq2SeqWithAttention(VOCAB_SIZE, embed_dim=128, hidden_dim=256, pad_idx=vocab[PAD]).to(DEVICE)
#


# Generate summary function
# ----------------------
def generate_summary(model, story_tensor, max_len=60):   # 💡 Reduce max_len for shorter summaries
    with torch.no_grad():
        embedded = model.embedding(story_tensor.unsqueeze(0).to(DEVICE))
        encoder_outputs, (h_n, c_n) = model.encoder(embedded)

        hidden = (h_n.transpose(0, 1).reshape(1, -1).unsqueeze(0),
                  c_n.transpose(0, 1).reshape(1, -1).unsqueeze(0))

        decoder_input = torch.tensor([[vocab[SOS]]], device=DEVICE)
        summary_indices = []

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

            # Attention energy
#             attn_input = torch.cat((repeated_hidden, encoder_outputs), dim=2)
#             attn_energy = torch.tanh(model.attn(attn_input))
#             attn_scores = torch.sum(attn_energy, dim=2)


            if top1 == vocab[EOS] or top1 == vocab[PAD]:
                break

            summary_indices.append(top1)
            decoder_input = torch.tensor([[top1]], device=DEVICE)

        return summary_indices



for i, sample in enumerate(data[:1]):
    sample_story = sample["story"]
    story_tensor = torch.tensor(text_to_indices(sample_story, vocab, max_len=500))

    gen_indices = generate_summary(model, story_tensor)
    gen_words = [inv_vocab.get(idx, "<UNK>") for idx in gen_indices]
    generated_summary = " ".join(gen_words)

    print(f"\n Story snippet #{i+1}:")
    print(sample_story[:500], "...\n")

    print(" True summary:")
    print(sample["summary"]["text"][:500], "...\n")

    print(" Generated summary (short):")
    print(generated_summary)
    print("=" * 80)







