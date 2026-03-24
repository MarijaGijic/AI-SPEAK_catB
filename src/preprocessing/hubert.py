import numpy as np
import torch
import torch.nn.functional as F
from transformers import HubertModel, Wav2Vec2FeatureExtractor

from src.config import HUBERT_SR, HUBERT_DIM

HUBERT_MODEL:     HubertModel              = None
HUBERT_EXTRACTOR: Wav2Vec2FeatureExtractor = None
DEVICE: str = "cpu"


def load_hubert(
    device: str = "cpu",
    model_name: str = "facebook/hubert-base-ls960",
) -> None:

    global HUBERT_MODEL, HUBERT_EXTRACTOR, DEVICE

    DEVICE = device
    print(f"[HuBERT] Ucitavam {model_name} na {device} ...")
    HUBERT_EXTRACTOR = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    HUBERT_MODEL     = HubertModel.from_pretrained(model_name).to(device).eval()
    print(f"[HuBERT] Model spreman. HUBERT_DIM={HUBERT_DIM}")


def extract_hubert_features(y_16k: np.ndarray, n_frames: int) -> np.ndarray:

    if HUBERT_MODEL is None or HUBERT_EXTRACTOR is None:
        raise RuntimeError(
            "[HuBERT] Model nije ucitan. Pozovi load_hubert() prije ekstrakcije."
        )

    inputs = HUBERT_EXTRACTOR(
        y_16k, sampling_rate=HUBERT_SR, return_tensors='pt', padding=False
    )

    cpu_model = HUBERT_MODEL.cpu()
    with torch.no_grad():
        feats = cpu_model(inputs.input_values).last_hidden_state
    HUBERT_MODEL.to(DEVICE)

    feats_60 = F.interpolate(
        feats.transpose(1, 2), size=n_frames,
        mode='linear', align_corners=False
    ).transpose(1, 2)

    return feats_60.squeeze(0).numpy().astype(np.float32)