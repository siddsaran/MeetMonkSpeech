import librosa
import whisper
import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from googletrans import Translator
import asyncio
#from sentence_transformers import SentenceTransformer
#import parselmouth
# from TTS.api import TTS

async def create_mapping(words, translator):
    map = dict()
    for word in words:
        result = await translator.translate(word, dest="hindi")
        result = result.text.strip(" .")
        map[word] = result
    return map

def create_timestamp_list(result):
    res = []
    for segment in result["segments"]:
        if "words" in segment:
            for word in segment["words"]:
                res.append({
                    "word": word["word"].strip(),
                    "start": word["start"],
                    "end": word["end"]
                })
    return res

def get_word_list(result):
    res = []
    for word in result["text"].split(" "):
        if len(word) > 0:
            res.append(word.strip(" ."))
    return res

def get_word_features(y, sr, timestamps):
    result = []
    for item in timestamps:
        start = float(item["start"])
        end = float(item["end"])
        start_sample = int(start * sr)
        end_sample = int(end * sr)

        word_audio = y[start_sample:end_sample]

        # RMS Energy
        rms = librosa.feature.rms(y=word_audio)[0]
        avg_rms = np.mean(rms)

        # Pitch (f0)
        f0, voiced_flag, _ = librosa.pyin(
            word_audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7')
        )
        avg_pitch = np.nanmean(f0)  # Mean pitch, ignoring unvoiced

        result.append({
            "word": item["word"],
            "start": start,
            "end": end,
            "rms": avg_rms,
            "pitch": avg_pitch
        })
    return result


async def main_async():
    filename = "./Input Sentences/test2.mp3"
    hindi_file = "./Hindi Audio Sentences/I am going home tomorrow.mp3"
    model = whisper.load_model("large")
    translator = Translator()

    # Original
    result = model.transcribe(filename, word_timestamps=True)
    sentence = result["text"].strip(".")
    words_en_list = get_word_list(result)
    words_en_timestamps = create_timestamp_list(result)

    # Translated
    result = model.transcribe(hindi_file, word_timestamps=True)
    words_translated_timestamps = create_timestamp_list(result)
    words_translated = get_word_list(result)
    translated = await translator.translate(sentence, dest="hindi")
    translated_text = translated.text

    # Create the map
    print(words_translated)
    print(words_en_list)
    map = await create_mapping(words_en_list, translator)
    print(map)

    # Audio Analysis
    #y_tgt, sr_tgt = librosa.load(hindi_file)
    y, sr = librosa.load(filename)
    word_energy = get_word_features(y, sr, words_en_timestamps)
    print(words_en_timestamps)
    print(word_energy)

    # Sampling
    target_sr = 8000  # target "sampled" rate for simulation
    sample_ratio = sr / target_sr
    sampled_indices = np.arange(0, len(y), sample_ratio).astype(int)
    sampled_y = y[sampled_indices]

    # Quantization
    n_bits = 8  # 8-bit quantization
    n_levels = 2 ** n_bits
    quantized_y = np.round((sampled_y + 1) / 2 * (n_levels - 1))
    quantized_y = quantized_y / (n_levels - 1) * 2 - 1

    # Reconstruction
    reconstructed_y = np.interp(np.arange(len(y)), sampled_indices, quantized_y)

    # Save
    sf.write("reconstructed1.wav", reconstructed_y, sr)
    print("Done")

    # waveform_tensor = torch.tensor(y).unsqueeze(0)
    #
    # mel_spectrogram = torchaudio.transforms.MelSpectrogram (
    #     sample_rate=sr,
    #     n_fft=1024,
    #     hop_length=256,
    #     n_mels=80
    # )
    #
    # mel_spec = mel_spectrogram(waveform_tensor)  # shape: (1, n_mels, time)
    # print(mel_spec)

    # Extract pitch and energy
    # f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    # f0_tgt, _, _ = librosa.pyin(y_tgt, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    # rms = librosa.feature.rms(y=y)[0]
    # rms_tgt = librosa.feature.rms(y=y_tgt)[0]

    # convert to PyTorch tensor from numpy waveform and add another dimension
    # 2D because many forms require form of (batch size, signal length)

    #y_torch = torch.tensor(y).unsqueeze(0)
    #spec = T.Spectrogram(n_fft=1024, hop_length=256)(y_torch)
    #recon = T.GriffinLim(n_fft=1024, hop_length=256)(spec)
    #sf.write("reconstructed.wav", recon.squeeze().numpy(), sr)

def main():
    # english = ["Can", "you", "help", "me"]
    # hindi = ["क्या", "आप", "मेरी", "मदद", "कर", "सकते", "हैं"]
    # translator = Translator()
    # print(english)
    # print(hindi)
    # print(asyncio.run(create_mapping(hindi, english, translator)))
    asyncio.run(main_async())
    

if __name__ == "__main__":
    main()