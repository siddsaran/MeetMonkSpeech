import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt
import whisper
from googletrans import Translator
import asyncio

class AnalogVoiceConverter:
    def __init__(self, target_voice_path):
        """
        Initialize with target voice (your voice) for parameter extraction
        """
        self.target_voice_path = target_voice_path
        self.target_params = self._extract_voice_parameters(target_voice_path)

    def _extract_voice_parameters(self, audio_path):
        """
        Extract key analog parameters from voice sample
        """
        # Load audio at standard sampling rate
        y, sr = librosa.load(audio_path, sr=22050)  # Standard rate

        # 1. AMPLITUDE/LOUDNESS CHARACTERISTICS
        # RMS energy (amplitude envelope)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]

        # Dynamic range analysis
        amplitude_stats = {
            'mean_rms': np.mean(rms),
            'std_rms': np.std(rms),
            'max_amplitude': np.max(np.abs(y)),
            'dynamic_range': np.max(rms) - np.min(rms[rms > 0])
        }

        # 2. FREQUENCY CHARACTERISTICS
        # Fundamental frequency (pitch)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
            frame_length=2048, hop_length=512
        )

        # Formant frequencies (using spectral peaks)
        formants = self._extract_formants(y, sr)

        # Spectral envelope (frequency content distribution)
        stft = librosa.stft(y, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        spectral_envelope = np.mean(magnitude, axis=1)  # Average across time

        frequency_stats = {
            'mean_f0': np.nanmean(f0),
            'std_f0': np.nanstd(f0),
            'f0_range': np.nanmax(f0) - np.nanmin(f0),
            'formants': formants,
            'spectral_envelope': spectral_envelope,
            'voiced_ratio': np.sum(voiced_flag) / len(voiced_flag)
        }

        # 3. TIMING CHARACTERISTICS
        # Speech rate and rhythm
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)

        timing_stats = {
            'speech_rate': len(onset_times) / (len(y) / sr),  # onsets per second
            'pause_intervals': self._analyze_pauses(rms),
            'rhythm_pattern': np.diff(onset_times) if len(onset_times) > 1 else [0]
        }

        return {
            'amplitude': amplitude_stats,
            'frequency': frequency_stats,
            'timing': timing_stats,
            'sample_rate': sr,
            'bit_depth_equivalent': 16  # Standard PCM
        }

    def _extract_formants(self, y, sr, num_formants=3):
        """
        Extract formant frequencies using LPC analysis
        """
        # Pre-emphasis filter
        pre_emphasis = 0.97
        emphasized = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

        # Window the signal
        windowed = emphasized * np.hamming(len(emphasized))

        # Linear Predictive Coding analysis
        # Simplified formant extraction using spectral peaks
        spectrum = np.abs(np.fft.fft(windowed))
        freqs = np.fft.fftfreq(len(spectrum), 1/sr)

        # Find peaks in positive frequency range
        pos_spectrum = spectrum[:len(spectrum)//2]
        pos_freqs = freqs[:len(freqs)//2]

        # Find formant peaks
        peaks, _ = signal.find_peaks(pos_spectrum, height=np.max(pos_spectrum)*0.1)
        formant_freqs = pos_freqs[peaks][:num_formants]

        return formant_freqs.tolist()

    def _analyze_pauses(self, rms, threshold_ratio=0.1):
        """
        Analyze pause patterns in speech
        """
        threshold = np.mean(rms) * threshold_ratio
        silence_mask = rms < threshold

        # Find silence regions
        silence_regions = []
        in_silence = False
        start_idx = 0

        for i, is_silent in enumerate(silence_mask):
            if is_silent and not in_silence:
                in_silence = True
                start_idx = i
            elif not is_silent and in_silence:
                in_silence = False
                silence_regions.append(i - start_idx)

        return silence_regions

    def convert_voice(self, source_audio_path, output_path):
        """
        Convert source voice to match target voice parameters
        """
        # Load source audio
        source_y, source_sr = librosa.load(source_audio_path, sr=self.target_params['sample_rate'])

        # Extract source parameters
        source_params = self._extract_voice_parameters(source_audio_path)

        # Apply conversions step by step
        converted_audio = source_y.copy()

        # 1. AMPLITUDE CONVERSION
        converted_audio = self._convert_amplitude(converted_audio, source_params, self.target_params)

        # 2. FREQUENCY CONVERSION
        converted_audio = self._convert_frequency(converted_audio, source_params, self.target_params)

        # 3. TIMING CONVERSION
        converted_audio = self._convert_timing(converted_audio, source_params, self.target_params)

        # 4. APPLY SPECTRAL ENVELOPE
        converted_audio = self._apply_spectral_envelope(converted_audio, source_params, self.target_params)

        # Save converted audio
        sf.write(output_path, converted_audio, self.target_params['sample_rate'])

        return converted_audio

    def _convert_amplitude(self, audio, source_params, target_params):
        """
        Convert amplitude characteristics using analog principles
        """
        # Calculate RMS of current audio
        current_rms = np.sqrt(np.mean(audio**2))

        # Scale to match target RMS
        target_rms = target_params['amplitude']['mean_rms']
        if current_rms > 0:
            scale_factor = target_rms / current_rms
            audio = audio * scale_factor

        # Apply dynamic range compression/expansion
        target_dynamic_range = target_params['amplitude']['dynamic_range']
        source_dynamic_range = source_params['amplitude']['dynamic_range']

        if source_dynamic_range > 0:
            # Simple dynamic range adjustment
            compression_ratio = target_dynamic_range / source_dynamic_range
            audio = np.sign(audio) * (np.abs(audio) ** compression_ratio)

        return audio

    def _convert_frequency(self, audio, source_params, target_params):
        """
        Convert frequency characteristics (pitch and formants)
        """
        sr = target_params['sample_rate']

        # 1. PITCH CONVERSION
        source_f0 = source_params['frequency']['mean_f0']
        target_f0 = target_params['frequency']['mean_f0']

        if not (np.isnan(source_f0) or np.isnan(target_f0)) and source_f0 > 0:
            # Calculate pitch shift in semitones
            pitch_shift_semitones = 12 * np.log2(target_f0 / source_f0)

            # Apply pitch shift using phase vocoder
            audio = librosa.effects.pitch_shift(
                audio, sr=sr, n_steps=pitch_shift_semitones
            )

        # 2. FORMANT CONVERSION (simplified)
        # This is a simplified formant shifting - in practice, you'd use more sophisticated methods
        source_formants = source_params['frequency']['formants']
        target_formants = target_params['frequency']['formants']

        if len(source_formants) > 0 and len(target_formants) > 0:
            # Simple spectral warping for formant adjustment
            audio = self._apply_formant_shift(audio, source_formants, target_formants, sr)

        return audio

    def _apply_formant_shift(self, audio, source_formants, target_formants, sr):
        """
        Apply formant shifting using spectral warping
        """
        # Get STFT
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Frequency bins
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        # Apply formant shifting (simplified)
        if len(source_formants) > 0 and len(target_formants) > 0:
            # Calculate shift ratio for first formant
            shift_ratio = target_formants[0] / source_formants[0]

            # Warp frequency axis
            warped_magnitude = np.zeros_like(magnitude)
            for i, freq in enumerate(freqs):
                target_freq = freq * shift_ratio
                if target_freq < freqs[-1]:
                    # Simple interpolation
                    target_idx = np.argmin(np.abs(freqs - target_freq))
                    warped_magnitude[target_idx] = magnitude[i]
        else:
            warped_magnitude = magnitude

        # Reconstruct audio
        modified_stft = warped_magnitude * np.exp(1j * phase)
        audio = librosa.istft(modified_stft, hop_length=512)

        return audio

    def _convert_timing(self, audio, source_params, target_params):
        """
        Convert timing characteristics (speech rate, rhythm)
        """
        # Simple time-stretching to match speech rate
        source_rate = source_params['timing']['speech_rate']
        target_rate = target_params['timing']['speech_rate']

        if source_rate > 0 and target_rate > 0:
            stretch_factor = source_rate / target_rate

            # Apply time stretching
            audio = librosa.effects.time_stretch(audio, rate=stretch_factor)

        return audio

    def _apply_spectral_envelope(self, audio, source_params, target_params):
        """
        Apply target spectral envelope to match voice timbre
        """
        # Get STFT
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Get spectral envelopes
        source_envelope = source_params['frequency']['spectral_envelope']
        target_envelope = target_params['frequency']['spectral_envelope']

        # Apply envelope transformation
        if len(source_envelope) == len(target_envelope):
            # Avoid division by zero
            envelope_ratio = target_envelope / (source_envelope + 1e-10)

            # Apply to magnitude spectrum
            modified_magnitude = magnitude * envelope_ratio[:, np.newaxis]
        else:
            modified_magnitude = magnitude

        # Reconstruct audio
        modified_stft = modified_magnitude * np.exp(1j * phase)
        audio = librosa.istft(modified_stft, hop_length=512)

        return audio

    def analyze_conversion_quality(self, original_path, converted_path):
        """
        Analyze how well the conversion matches target parameters
        """
        converted_params = self._extract_voice_parameters(converted_path)

        # Compare key parameters
        analysis = {
            'pitch_match': {
                'target': self.target_params['frequency']['mean_f0'],
                'converted': converted_params['frequency']['mean_f0'],
                'difference': abs(self.target_params['frequency']['mean_f0'] -
                                  converted_params['frequency']['mean_f0'])
            },
            'amplitude_match': {
                'target': self.target_params['amplitude']['mean_rms'],
                'converted': converted_params['amplitude']['mean_rms'],
                'difference': abs(self.target_params['amplitude']['mean_rms'] -
                                  converted_params['amplitude']['mean_rms'])
            },
            'timing_match': {
                'target': self.target_params['timing']['speech_rate'],
                'converted': converted_params['timing']['speech_rate'],
                'difference': abs(self.target_params['timing']['speech_rate'] -
                                  converted_params['timing']['speech_rate'])
            }
        }

        return analysis

# Example usage
if __name__ == "__main__":
    # Initialize converter with your voice sample
    converter = AnalogVoiceConverter("reconstructed.wav")

    # Convert source audio to sound like you
    converted_audio = converter.convert_voice(
        "hindi_adjusted1.wav",
        "converted_output.wav"
    )

    # Analyze conversion quality
    analysis = converter.analyze_conversion_quality(
        "hindi_adjusted1.wav",
        "converted_output.wav"
    )

    print("Conversion Analysis:")
    for param, values in analysis.items():
        print(f"{param}: {values}")