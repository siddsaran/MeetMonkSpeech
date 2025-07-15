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

        # Pitch (f0)
        f0, voiced_flag, _ = librosa.pyin(
            word_audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7')
        )
        avg_pitch = np.nanmean(f0)  # Mean pitch, ignoring unvoiced

        result.append({
            "word": item["word"].strip(" ."),
            "start": start,
            "end": end,
            "rms": rms,
            "pitch": avg_pitch
        })
    return result

def apply_rms(y_tgt, sr, eng_timestamps, hind_timestamps, map):
    y = np.copy(y_tgt)
    for i in range(len(hind_timestamps)):
        elem = hind_timestamps[i]
        hindi_word = elem['word']
        start = int(elem['start'] * sr)
        end = int(elem['end'] * sr)
        hindi_rms = elem['rms'].mean()
        target_rms = hindi_rms
        for elem in eng_timestamps:
            word = elem['word']
            translation = map[word]
            if hindi_word in translation:
                target_rms = elem['rms'].mean()
                print(word + " with translation " + translation + " map to " + hindi_word)
        y[start:end] *= target_rms / hindi_rms
    return y



async def main_async():
    filename = "./Input Sentences/test1.mp3"
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
    y_tgt, sr_tgt = librosa.load(hindi_file)
    # Load audios
    y, sr = librosa.load(filename)
    word_energy = get_word_features(y, sr, words_en_timestamps)
    word_energy_hindi = get_word_features(y_tgt, sr_tgt, words_translated_timestamps)
    print(word_energy)
    print(word_energy_hindi)
    reconstructed_y = apply_rms(y_tgt, sr, word_energy, word_energy_hindi, map)
    # Save
    sf.write("reconstructed1.wav", reconstructed_y, sr)
    print("Done")
    # y_input, sr = librosa.load(filename)
    # y_output, sr_out = librosa.load("There are three days left.mp3")
    #
    # # Extract pitch contours
    # f0_in, _, _ = librosa.pyin(y_input, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    # f0_out, _, _ = librosa.pyin(y_output, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    #
    # # Handle nan
    # mean_f0_in = np.nanmean(f0_in)
    # mean_f0_out = np.nanmean(f0_out)
    # ratio = mean_f0_in / mean_f0_out
    #
    # # Pitch shift output voice globally
    # n_steps = 12 * np.log2(ratio)
    # y_output_pitch = librosa.effects.pitch_shift(y=y_output, sr=sr_out, n_steps=n_steps)
    #
    # # Extract RMS envelopes
    # rms_in = librosa.feature.rms(y=y_input)[0]
    # rms_out = librosa.feature.rms(y=y_output_pitch)[0]
    #
    # # Align lengths for simplicity (you could stretch/compress properly here)
    # min_len = min(len(rms_in), len(rms_out))
    # rms_ratio = rms_in[:min_len] / (rms_out[:min_len] + 1e-8)
    # rms_ratio = np.clip(rms_ratio, 0.5, 2.0)  # limit gain factor range
    #
    # frame_length = 2048
    # hop_length = 512
    # y_out_energy = np.copy(y_output_pitch)
    #
    # # Apply energy shaping
    # for i, gain in enumerate(rms_ratio):
    #     start = i * hop_length
    #     end = start + frame_length
    #     y_out_energy[start:end] *= gain
    #
    # # Normalize
    # y_out_energy = y_out_energy / np.max(np.abs(y_out_energy))
    #
    # # 4️⃣ Save
    # sf.write("google_output_voice_styled.wav", y_out_energy, sr_out)
    # print("Done")

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