import torch
from torch.serialization import add_safe_globals

# 🔐 Allow required XTTS objects for unpickling
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, Xtts, XttsEmbeddingConfig
add_safe_globals([XttsConfig, XttsAudioConfig, Xtts, XttsEmbeddingConfig])

from TTS.api import TTS

# Load model
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

# Get speaker embedding from your voice clip
embedding = tts.get_speaker_embedding("reconstructed.wav")

print("Speaker embedding shape:", embedding.shape)
print("First 10 dims:", embedding[0][:10])
