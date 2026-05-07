from pathlib import Path

import librosa
import numpy as np
import torch
from panns_inference import AudioTagging, labels

def get_music_index():
    for label_index, label in enumerate(labels):
        if label.lower() == "music":
            return label_index
    return None


def compute_music_windows(wav_path, window_sec=10.0, hop_sec=2.0, sample_rate=32000):
    if not Path(wav_path).exists():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    music_index = get_music_index()
    if music_index is None:
        return []

    audio = librosa.load(wav_path, sr=sample_rate, mono=True)[0]
    audio = audio.astype(np.float32)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model = AudioTagging(checkpoint_path=None, device=device)

    # sliding windows: default 10 sec window with 2 sec hop
    window_size = int(window_sec * sample_rate)
    hop_size = int(hop_sec * sample_rate)

    music_windows = []

    for start in range(0, len(audio), hop_size):
        window = audio[start:start + window_size]

        # stop when remaining audio is shorter than 1 sec
        if len(window) < sample_rate:
            break

        # pad final short window to model input size
        if len(window) < window_size:
            window = np.pad(window, (0, window_size - len(window)))

        # music probability for this window
        output = model.inference(window[None, :])[0]
        score = float(output[0][music_index])

        music_windows.append({
            "start": start / sample_rate,
            "end": min((start + window_size) / sample_rate, len(audio) / sample_rate),
            "music_score": score,
        })

    return music_windows


def mean_music_score(music_windows, start, end):
    # average music score for windows overlapping this interval
    scores = []

    for music_window in music_windows:
        if music_window["end"] >= start and music_window["start"] <= end:
            scores.append(music_window["music_score"])

    if len(scores) == 0:
        return 0.0

    return float(np.mean(scores))


def music_features_for_interval(music_windows, start, end, context_sec=60.0):
    # compare music score inside interval against surrounding context
    inside = mean_music_score(music_windows, start, end)
    before = mean_music_score(music_windows, max(0, start - context_sec), start)
    after = mean_music_score(music_windows, end, end + context_sec)

    # music delta measures how different the interval is from nearby content
    surrounding = (before + after) / 2
    delta = abs(inside - surrounding)

    return inside, surrounding, delta
