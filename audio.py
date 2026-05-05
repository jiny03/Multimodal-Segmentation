import ffmpeg
from pathlib import Path
import numpy as np
import soundfile as sf
import subprocess
import pyloudnorm as pyln
import re
from scipy.signal import find_peaks
import librosa


def extract_audio(video_path, wav_path, sample_rate=16000):
    video_path = Path(video_path)
    wav_path = Path(wav_path)

    # check if audio track exists
    data = ffmpeg.probe(str(video_path))
    audio_streams = []

    for stream in data["streams"]:
        if stream["codec_type"] == "audio":
            audio_streams.append(stream)

    if len(audio_streams) == 0:
        raise RuntimeError(f"No audio track in: {video_path}")

    # extract audio using ffmpeg and save to mono 16kHz WAV
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-acodec", "pcm_s16le",
        "-y",
        "-loglevel", "error",
        str(wav_path),
    ]

    subprocess.run(command, check=True)
    return str(wav_path)


def load_audio(wav_path):
    # load WAV as float audio
    wav_path = Path(wav_path)
    audio, sample_rate = sf.read(str(wav_path))
    audio = audio.astype(np.float32)
    return audio, sample_rate


def compute_loudness_delta(wav_path, window_sec=1.0):
    wav_path = Path(wav_path)
    audio, sample_rate = load_audio(wav_path)

    # window loudness (default: 1 sec window in 16kHz)
    audio = audio.astype(np.float32)
    window_size = int(window_sec * sample_rate)
    meter = pyln.Meter(sample_rate, block_size=window_sec)
    loudness_list = []

    for start in range(0, len(audio) - window_size + 1, window_size):
        # silence = -70 LKFS
        loudness = -70.0
        window = audio[start:start + window_size]

        if np.max(np.abs(window)) > 1e-6:
            loudness = meter.integrated_loudness(window)
            if not np.isfinite(loudness):
                loudness = -70.0
        loudness_list.append(loudness)

    loudness_array = np.array(loudness_list, dtype=np.float32)

    # difference between neighboring windows
    loudness_delta = np.abs(np.diff(loudness_array))

    # timestamps: default 1.0 sec window ([1.0, 2.0, 3.0, ...])
    times = np.arange(1, len(loudness_array)) * window_sec

    return times, loudness_array, loudness_delta


def compute_mfcc_delta(wav_path, window_sec=1.0, num_mfcc=13):
    wav_path = Path(wav_path)
    audio, sample_rate = load_audio(wav_path)

    # compute MFCC features for each 1 sec window
    window_size = int(window_sec * sample_rate)
    mfcc_windows = []

    for start in range(0, len(audio) - window_size + 1, window_size):
        window = audio[start:start + window_size]

        # silence = zero MFCC vector
        if np.max(np.abs(window)) <= 1e-6:
            mfcc_mean = np.zeros(num_mfcc, dtype=np.float32)
        else:
            mfcc = librosa.feature.mfcc(y=window.astype(np.float32), sr=sample_rate, n_mfcc=num_mfcc)
            mfcc_mean = np.mean(mfcc, axis=1).astype(np.float32)

        mfcc_windows.append(mfcc_mean)

    if len(mfcc_windows) < 2:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    # difference between neighboring MFCC windows
    mfcc_delta = []

    for window_index in range(1, len(mfcc_windows)):
        previous = mfcc_windows[window_index - 1]
        current = mfcc_windows[window_index]
        denominator = np.linalg.norm(previous) * np.linalg.norm(current)

        if denominator <= 1e-8:
            distance = 0.0
        else:
            similarity = float(np.dot(previous, current) / denominator)
            distance = 1.0 - max(-1.0, min(1.0, similarity))

        mfcc_delta.append(distance)

    # timestamps: default 1.0 sec window ([1.0, 2.0, 3.0, ...])
    times = np.arange(1, len(mfcc_windows)) * window_sec
    return times, np.array(mfcc_delta, dtype=np.float32)


def detect_silence_regions(wav_path, noise_db=-30, min_duration=0.3):
    wav_path = Path(wav_path)

    # use ffmpeg silencedetect to find quiet regions near possible boundaries
    command = [
        "ffmpeg",
        "-i", str(wav_path),
        "-af", f"silencedetect=noise={noise_db}dB:d={min_duration}",
        "-f", "null",
        "-"
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    output = result.stderr

    silence_starts = []
    silence_regions = []

    for line in output.splitlines():
        start_match = re.search(r"silence_start:\s*([0-9.]+)", line)
        end_match = re.search(r"silence_end:\s*([0-9.]+)", line)

        if start_match:
            silence_starts.append(float(start_match.group(1)))

        if end_match and len(silence_starts) > 0:
            start = silence_starts.pop(0)
            end = float(end_match.group(1))
            silence_regions.append((start, end))

    # [(silence_start, silence_end), ...]
    return silence_regions


def silence_neighbor_present(time, silence_regions, tolerance_sec=1.0):
    for silence_start, silence_end in silence_regions:
        if silence_start - tolerance_sec <= time <= silence_end + tolerance_sec:
            return True

    return False


def normalize_score(values):
    # normalize
    values = np.asarray(values, dtype=np.float32)

    if len(values) == 0:
        return values

    min_val = np.min(values)
    max_val = np.max(values)

    if max_val - min_val < 1e-8:
        return np.zeros_like(values)

    return (values - min_val) / (max_val - min_val)


def compute_boundary_score_loudness_silence(wav_path):
    times, loudness, loudness_delta = compute_loudness_delta(wav_path)
    mfcc_delta = compute_mfcc_delta(wav_path)[1]
    silence_regions = detect_silence_regions(wav_path)

    # loudness changes & nearby silence -> one boundary score
    loudness_score = normalize_score(loudness_delta)
    spectral_score = normalize_score(mfcc_delta)

    if len(spectral_score) != len(loudness_score):
        min_length = min(len(spectral_score), len(loudness_score))
        times = times[:min_length]
        loudness_score = loudness_score[:min_length]
        spectral_score = spectral_score[:min_length]

    silence_bonus = []
    for t in times:
        if silence_neighbor_present(t, silence_regions):
            silence_bonus.append(1.0)
        else:
            silence_bonus.append(0.0)

    silence_bonus = np.array(silence_bonus, dtype=np.float32)

    boundary_score = 0.45 * loudness_score + 0.35 * spectral_score + 0.20 * silence_bonus

    return times, boundary_score, loudness_score, silence_bonus


def find_boundary_peaks(wav_path, prominence=0.3, distance_sec=25):
    times, boundary_score, loudness_score, silence_bonus = compute_boundary_score_loudness_silence(wav_path)

    distance_samples = int(distance_sec)

    # finds the local maximum of boundary score which is likely a content change
    peak_indices, properties = find_peaks(boundary_score, prominence=prominence, distance=distance_samples)

    peak_times = times[peak_indices]
    peak_scores = boundary_score[peak_indices]

    return peak_times, peak_scores, peak_indices, properties


def generate_candidate_intervals(peak_times, peak_scores, min_duration=25, max_duration=140, skip_penalty_weight=0.15):
    candidates = []

    # pair boundary peaks to create possible ad intervals
    for start_peak_index in range(len(peak_times)):
        for end_peak_index in range(start_peak_index + 1, len(peak_times)):
            start = peak_times[start_peak_index]
            end = peak_times[end_peak_index]
            duration = end - start

            if min_duration <= duration <= max_duration:
                boundary_score = peak_scores[start_peak_index] + peak_scores[end_peak_index]

                # number of internal peaks between start and end
                skipped_peaks = end_peak_index - start_peak_index - 1
                skip_penalty = skip_penalty_weight * skipped_peaks

                adjusted_score = boundary_score - skip_penalty

                candidates.append({
                    "start": float(start),
                    "end": float(end),
                    "duration": float(duration),
                    "score": float(adjusted_score),
                    "raw_boundary_score": float(boundary_score),
                    "start_score": float(peak_scores[start_peak_index]),
                    "end_score": float(peak_scores[end_peak_index]),
                    "skipped_peaks": int(skipped_peaks),
                    "skip_penalty": float(skip_penalty),
                })

    return candidates




















