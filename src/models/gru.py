import torch.nn as nn
from .base import InputEncoder, OutputHead

class BlendshapeGRU(nn.Module):
    def __init__(self, d_model=256, hidden_size=256, n_layers=3, dropout=0.2,
                 n_speakers=2, bidirectional=True,
                 audio_type="mfcc", use_phonemes=True):
        super().__init__()
        self.encoder = InputEncoder(
            d_model=d_model, n_speakers=n_speakers, dropout=dropout,
            audio_type=audio_type, use_phonemes=use_phonemes,
        )
        self.gru = nn.GRU(d_model, hidden_size, n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0.,
                          bidirectional=bidirectional)
        head_dim = hidden_size * (2 if bidirectional else 1)
        self.head = OutputHead(head_dim, dropout=dropout)

    def forward(self, af, pi, pt, si, lengths=None, hubert=None):
        x = self.encoder(af, pi, pt, si, hubert=hubert)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.gru(x)
        return self.head(out)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)