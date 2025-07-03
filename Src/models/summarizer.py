import torch
import torch.nn as nn

class Seq2SeqWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(embed_dim + hidden_dim * 2, hidden_dim * 2, batch_first=True)
        self.attn = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.out = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.size()

        # Encoder
        embedded_src = self.embedding(src)
        encoder_outputs, (h_n, c_n) = self.encoder(embedded_src)

        decoder_input = tgt[:, 0].unsqueeze(1)  # SOS
        outputs = torch.zeros(batch_size, tgt_len, self.out.out_features, device=src.device)

        hidden = (h_n.transpose(0, 1).reshape(batch_size, -1).unsqueeze(0),
                  c_n.transpose(0, 1).reshape(batch_size, -1).unsqueeze(0))

        for t in range(1, tgt_len):
            embedded_dec = self.embedding(decoder_input)

            # Repeat decoder hidden state across time dimension
            repeated_hidden = hidden[0].transpose(0, 1).repeat(1, encoder_outputs.size(1), 1)

            # Calculate attention scores
            attn_input = torch.cat((repeated_hidden, encoder_outputs), dim=2)
            attn_energy = torch.tanh(self.attn(attn_input))  # shape: [batch, src_len, hidden_dim*2]
            attn_scores = torch.sum(attn_energy, dim=2)     # shape: [batch, src_len]

            attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(1)  # shape: [batch, 1, src_len]

            # Context vector
            context = torch.bmm(attn_weights, encoder_outputs)  # shape: [batch, 1, hidden_dim*2]

            # Combine embedding and context
            rnn_input = torch.cat((embedded_dec, context), dim=2)

            output, hidden = self.decoder(rnn_input, hidden)
            pred = self.out(output.squeeze(1))
            outputs[:, t, :] = pred

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = pred.argmax(1)
            decoder_input = tgt[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        return outputs
