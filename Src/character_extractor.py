import json
import spacy
import re
from collections import defaultdict, Counter

nlp = spacy.load("en_core_web_sm")

with open("data/small_narrativeqa.json", "r", encoding="utf-8") as f:
    data = json.load(f)



# def extract_possible_characters(text, top_n=10):
    
#     # Heuristic: repeated capitalized words

#     words = re.findall(r'\b[A-Z][a-z]+\b', text)
#     word_counts = Counter(words)
#     common_names = [w for w, count in word_counts.items() if count >= 2]
#     return common_names[:top_n]



def extract_possible_characters(text, top_n=10):
    # Heuristic: repeated capitalized words
    words = re.findall(r'\b[A-Z][a-z]+\b', text)
    word_counts = Counter(words)
    common_names = [w for w, count in word_counts.items() if count >= 2]
    return common_names[:top_n]


def extract_traits(doc, character):
    traits = set()
    for sent in doc.sents:
        if character in sent.text:
            for token in sent:
                # Linking verbs
                if token.text.lower() in ["is", "was", "seems", "became", "felt", "appeared", "looked"]:
                    subj = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
                    if any(character in s.text for s in subj):
                        adj = [child.text for child in token.children if child.pos_ == "ADJ"]
                        traits.update(adj)

                
                
                # Adjective phrases or appositions
                if token.text == character:
                    for child in token.children:
                        if child.dep_ == "appos":
                            traits.update([desc.text for desc in child.subtree if desc.pos_ == "ADJ"])
                        if child.pos_ == "ADJ":
                            traits.add(child.text)
    return traits

# Process first 3 stories
for idx, sample in enumerate(data[:1]):
    story_text = sample["story"]
    doc = nlp(story_text)

    # Combine spaCy named entities and heuristic capitalized names
    spacy_characters = set(ent.text for ent in doc.ents if ent.label_ == "PERSON")
    heuristic_names = set(extract_possible_characters(story_text, top_n=10))
    all_characters = spacy_characters.union(heuristic_names)

    character_traits = defaultdict(set)


    # If no characters found, skip processing


    for character in all_characters:
        traits = extract_traits(doc, character)
        if traits:
            character_traits[character].update(traits)

    
    
    print("=" * 80)
    print(f"Story #{idx + 1} - Character Traits\n")
    if not character_traits:
        print("No character traits found.")
    for char, traits in character_traits.items():
        print(f"{char}: {', '.join(traits)}")











# import json
# import torch
# from Utils.vocab import text_to_indices, load_vocab, PAD, SOS, EOS
# from models.summarizer import Seq2SeqWithAttention

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load data
# with open("data/test_narrativeqa.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# vocab = load_vocab("Data/vocab.json")
# inv_vocab = {v: k for k, v in vocab.items()}
# VOCAB_SIZE = len(vocab)

# print("Vocab size:", VOCAB_SIZE)

# model = Seq2SeqWithAttention(VOCAB_SIZE, embed_dim=128, hidden_dim=256, pad_idx=vocab[PAD]).to(DEVICE)
# model.load_state_dict(torch.load("Data/model_weights.pth", map_location=DEVICE))
# model.eval()

# def generate_description(model, story_tensor, max_len=100):
#     with torch.no_grad():
#         embedded = model.embedding(story_tensor.unsqueeze(0).to(DEVICE))
#         encoder_outputs, (h_n, c_n) = model.encoder(embedded)

#         hidden = (h_n.transpose(0, 1).reshape(1, -1).unsqueeze(0),
#                   c_n.transpose(0, 1).reshape(1, -1).unsqueeze(0))

#         decoder_input = torch.tensor([[vocab[SOS]]], device=DEVICE)
#         desc_indices = []

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

#             desc_indices.append(top1)
#             decoder_input = torch.tensor([[top1]], device=DEVICE)

#         return desc_indices

# # Change: Instead of summary, instruct it to generate character descriptions
# instruction_text = "Describe the main characters in this story."

# for idx, story_entry in enumerate(data[:1]):
#     story_text = story_entry["story"]

#     # Optionally add instruction to story text
#     modified_story_text = instruction_text + " " + story_text

#     story_tensor = torch.tensor(text_to_indices(modified_story_text, vocab, max_len=500))
#     gen_indices = generate_description(model, story_tensor)
#     gen_words = [inv_vocab.get(idx, "<UNK>") for idx in gen_indices]
#     generated_description = " ".join(gen_words)

#     print(f"\n Story snippet #{idx+1}:")
#     print(story_text[:500], "...\n")
#     print(" Generated character descriptions:")
#     print(generated_description)
#     