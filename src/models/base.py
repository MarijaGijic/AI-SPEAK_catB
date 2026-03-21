import torch, torch.nn as nn, math
from src.config import FEAT_DIM, HUBERT_DIM, N_PHONEMES, N_BLENDSHAPES

class InputEncoder(nn.Module):
    def __init__(self, d_model=256, phoneme_emb_dim=32, speaker_emb_dim=8,
                 n_phonemes=N_PHONEMES, n_speakers=2, dropout=0.1,
                 audio_type="mfcc",
                 use_phonemes=True):
        
        super().__init__()
        self.audio_type = audio_type
        self.use_phonemes = use_phonemes

        self.phoneme_emb = nn.Embedding(n_phonemes, phoneme_emb_dim, padding_idx=0)
        self.speaker_emb = nn.Embedding(n_speakers, speaker_emb_dim)

        audio_dim = HUBERT_DIM+FEAT_DIM if audio_type == "hubert" else FEAT_DIM
        ph_dim    = phoneme_emb_dim + 1 if use_phonemes else 0
        in_dim = audio_dim + ph_dim + speaker_emb_dim
        
        if audio_type == "hubert":
            self.proj = nn.Sequential(
                nn.Linear(in_dim, d_model * 2),
                nn.LayerNorm(d_model * 2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model), nn.GELU(),
            )
        else:
            self.proj = nn.Sequential(
                nn.Linear(in_dim, d_model),
                nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout),
            )

    def forward(self, af, pi, pt, si, hubert=None):
        # af is always MFCC features
        # hubert is only passed when audio_type="hubert"
        parts = [af]
        if self.audio_type == "hubert" and hubert is not None:
            parts.append(hubert)
        if self.use_phonemes:
            parts += [pt, self.phoneme_emb(pi)]
        parts.append(self.speaker_emb(si))
        return self.proj(torch.cat(parts, dim=-1))
    
class OutputHead(nn.Module):
    def __init__(self, in_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(in_dim // 2, N_BLENDSHAPES), nn.Sigmoid()
        )
    
    def forward(self, x): return self.net(x)