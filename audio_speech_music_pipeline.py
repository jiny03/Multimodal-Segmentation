from pathlib import Path
import argparse
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from audio import extract_audio, find_boundary_peaks, generate_candidate_intervals
from music import compute_music_windows, music_features_for_interval
from speech import transcribe_audio, summarize_video_theme, detect_non_content_from_full_transcript, get_interval_transcript


STRUCTURAL_AUDIO_THRESHOLD = 0.56
MAX_STRUCTURAL_AUDIO_RESULTS = 2
MIN_STRUCTURAL_AUDIO_CENTER_GAP = 90.0
TEXT_VECTORIZER = TfidfVectorizer(token_pattern=r"(?u)\b[a-z0-9]+\b")


def normalize_label(label):
    if label == "ad":
        return "ad"

    return "content"


# cache
def read_json_cache(path):
    path = Path(path)

    if not path.exists():
        return None

    with open(path) as file:
        return json.load(file)

def write_json_cache(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as file:
        json.dump(data, file, indent=2)

def get_cached_json(path, compute_fn, force_recompute=False):
    if not force_recompute:
        cached = read_json_cache(path)

        if cached is not None:
            return cached

    data = compute_fn()
    write_json_cache(path, data)
    return data

def boundary_peaks(wav_path):
    peak_times, peak_scores, peak_indices, properties = find_boundary_peaks(wav_path)

    return {
        "peak_times": peak_times.tolist(),
        "peak_scores": peak_scores.tolist(),
    }


def duration_prior_score(duration):
    common_durations = [30.0, 45.0, 60.0, 90.0, 120.0]
    best_distance = min(abs(float(duration) - d) for d in common_durations)

    return max(0.0, 1.0 - best_distance / 20.0)


def text_distance(doc_a, doc_b):
    # computes text distance of text neighbors to a candidate
    analyzer = TEXT_VECTORIZER.build_analyzer()

    if len(analyzer(doc_a)) == 0 or len(analyzer(doc_b)) == 0:
        return 0.0

    vectors = TEXT_VECTORIZER.fit_transform([doc_a, doc_b])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return float(1.0 - np.clip(similarity, 0.0, 1.0))


def transcript_shift_score(transcript_segments, start, end, context_sec=45.0):
    # compare candidate transcript text against nearby context to detect topic shifts
    before = get_interval_transcript(transcript_segments, max(0.0, start - context_sec), start)
    inside = get_interval_transcript(transcript_segments, start, end)
    after = get_interval_transcript(transcript_segments, end, end + context_sec)

    distances = []

    if len(before.strip()) > 0:
        distances.append(text_distance(inside, before))

    if len(after.strip()) > 0:
        distances.append(text_distance(inside, after))

    if len(distances) == 0:
        return 0.0

    return float(sum(distances) / len(distances))


def structural_audio_score(result):
    # rank nonverbal candidates using boundary strength, ad duration, music change, and speech gaps
    score = (
        0.30 * float(result["audio_boundary_score"]) +
        0.30 * duration_prior_score(result["duration"]) +
        0.25 * float(result["music_delta"]) +
        0.15 * float(result["low_speech_score"]) -
        0.06 * int(result.get("candidate_skipped_peaks", 0))
    )

    return clamp_score(score)


def nearest_boundary_score(start, end, peak_times, peak_scores, tolerance_sec=5.0):
    if len(peak_times) == 0:
        return 0.0

    peak_times = np.asarray(peak_times, dtype=np.float32)
    peak_scores = np.asarray(peak_scores, dtype=np.float32)
    nearby = (np.abs(peak_times - start) <= tolerance_sec) | (np.abs(peak_times - end) <= tolerance_sec)

    if not np.any(nearby):
        return 0.0

    return float(np.max(peak_scores[nearby]))


def clamp_score(score):
    if score < 0.0:
        score = 0.0
    else:
        if score > 1.0:
            score = 1.0
    return float(score)


def low_speech_score(transcript_segments, start, end):
    duration = end - start

    if duration <= 0:
        return 0.0

    speech_duration = sum(
        max(0.0, min(end, float(segment["end"])) - max(start, float(segment["start"])))
        for segment in transcript_segments
    )
    return clamp_score(1.0 - speech_duration / duration)


def overlap_ratio(start_a, end_a, start_b, end_b):
    overlap_start = max(start_a, start_b)
    overlap_end = min(end_a, end_b)

    if overlap_end <= overlap_start:
        return 0.0
    else:
        overlap_duration = overlap_end - overlap_start
        shorter_duration = min(end_a - start_a, end_b - start_b)

        if shorter_duration <= 0:
            return 0.0
        else:
            return float(overlap_duration / shorter_duration)


def overlaps_segment(results, start, end, threshold=0.5):
    # avoid adding fallback detections that duplicate stronger detections already kept.
    for result in results:
        result_start = float(result["start"])
        result_end = float(result["end"])
        current_overlap_ratio = overlap_ratio(start, end, result_start, result_end)

        if current_overlap_ratio >= threshold:
            return True

    return False


def combine_labels(label_a, label_b):
    if normalize_label(label_a) == "ad" or normalize_label(label_b) == "ad":
        return "ad"

    return "content"


def merge_source_text(source_a, source_b):
    if source_b in source_a:
        source = source_a
    else:
        source = source_a + "+" + source_b

    return source


def merge_reason_text(reason_a, reason_b):
    if len(reason_a.strip()) == 0:
        reason = reason_b
    else:
        if len(reason_b.strip()) == 0:
            reason = reason_a
        else:
            if reason_b in reason_a:
                reason = reason_a
            else:
                reason = reason_a + " " + reason_b

    return reason


def keep_speech_segment(segment, min_duration=15.0, min_score=0.75):
    # only keep speech segments with scores higher than 0.75
    start = float(segment["start"])
    end = float(segment["end"])
    duration = end - start
    label = normalize_label(segment.get("label", "content"))
    score = float(segment.get("non_content_score", 0.0))

    # filter label = content
    if label != "ad":
        return False

    # if duration is short it is very likely false unless its speech score is very high
    if duration < min_duration and score < 0.95:
        return False

    return score >= min_score


def keep_structural_audio_result(result):
    #filter audio structural results with a set threshold and duration
    if float(result["duration"]) < 25.0:
        return False

    if int(result.get("candidate_skipped_peaks", 0)) > 1:
        return False

    return float(result["structural_audio_score"]) >= STRUCTURAL_AUDIO_THRESHOLD


def segment_center(segment):
    return (float(segment["start"]) + float(segment["end"])) / 2.0


def near_structural_candidate(segment, selected_segments):
    center = segment_center(segment)

    for selected in selected_segments:
        if abs(center - segment_center(selected)) < MIN_STRUCTURAL_AUDIO_CENTER_GAP:
            return True

    return False


def keep_music_region_result(result, min_score=0.65, min_duration=25.0, min_music_delta=0.25):
    # filter music results with a set threshold
    if float(result["duration"]) < min_duration:
        return False

    if float(result["music_delta"]) < min_music_delta:
        return False

    if float(result["audio_boundary_score"]) < 0.50 and float(result["low_speech_score"]) < 0.75:
        return False

    return float(result["non_content_score"]) >= min_score


def merge_two_segments(previous, current):
    previous["end"] = max(float(previous["end"]), float(current["end"]))
    previous["duration"] = float(previous["end"]) - float(previous["start"])

    previous["label"] = combine_labels(
        previous.get("label", "possible_ad_or_non_content"),
        current.get("label", "possible_ad_or_non_content")
    )

    previous["non_content_score"] = max(
        float(previous.get("non_content_score", 0.0)),
        float(current.get("non_content_score", 0.0))
    )

    previous["source"] = merge_source_text(
        previous.get("source", ""),
        current.get("source", "")
    )

    previous["speech_score"] = max(
        float(previous.get("speech_score", 0.0)),
        float(current.get("speech_score", 0.0))
    )

    previous["audio_boundary_score"] = max(
        float(previous.get("audio_boundary_score", 0.0)),
        float(current.get("audio_boundary_score", 0.0))
    )

    previous["music_inside"] = max(
        float(previous.get("music_inside", 0.0)),
        float(current.get("music_inside", 0.0))
    )

    previous["music_surrounding"] = max(
        float(previous.get("music_surrounding", 0.0)),
        float(current.get("music_surrounding", 0.0))
    )

    previous["music_delta"] = max(
        float(previous.get("music_delta", 0.0)),
        float(current.get("music_delta", 0.0))
    )

    previous["low_speech_score"] = max(
        float(previous.get("low_speech_score", 0.0)),
        float(current.get("low_speech_score", 0.0))
    )

    previous["reason"] = merge_reason_text(
        previous.get("reason", ""),
        current.get("reason", "")
    )

    return previous


def merge_consecutive_segments(segments, max_gap=8.0, max_merged_duration=180.0):
    # if nearby segments are consecutive (ignoring small gaps) merge them into one
    sorted_segments = sorted(segments, key=lambda segment: segment["start"])
    merged_segments = []

    for segment in sorted_segments:
        current = dict(segment)

        if len(merged_segments) == 0:
            merged_segments.append(current)
        else:
            previous = merged_segments[-1]
            gap = float(current["start"]) - float(previous["end"])
            merged_duration = float(current["end"]) - float(previous["start"])

            if gap <= max_gap and merged_duration <= max_merged_duration:
                merged_segments[-1] = merge_two_segments(previous, current)
            else:
                merged_segments.append(current)

    return merged_segments


def format_ad_segment(segment):
    return {
        "start": float(segment["start"]),
        "end": float(segment["end"]),
        "duration": float(segment["duration"]),
        "non_content_score": float(segment["non_content_score"]),
        "source": segment.get("source", "audio_speech_music"),
        "speech_score": float(segment.get("speech_score", 0.0)),
        "audio_boundary_score": float(segment.get("audio_boundary_score", 0.0)),
        "music_delta": float(segment.get("music_delta", 0.0)),
        "low_speech_score": float(segment.get("low_speech_score", 0.0))
    }


def format_transcript_segments(transcript_segments):
    # for frontend video player
    return [
        {
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "duration": float(segment["end"]) - float(segment["start"]),
            "text": segment.get("text", "")
        }
        for segment in transcript_segments
    ]


def build_speech_result(segment, music_windows, peak_times, peak_scores):
    # use speech scores (generated from LLM) as primary, add audio/music scores as secondary factors
    start = float(segment["start"])
    end = float(segment["end"])

    music_inside, music_surrounding, music_delta = music_features_for_interval(music_windows, start, end)
    audio_boundary_score = nearest_boundary_score(start, end, peak_times, peak_scores)

    speech_score = float(segment.get("non_content_score", 0.0))
    music_score = clamp_score(music_delta)
    audio_score = clamp_score(audio_boundary_score)

    # not final
    combined_score = 0.75 * speech_score + 0.15 * music_score + 0.10 * audio_score

    if normalize_label(segment.get("label", "content")) == "ad" and speech_score >= 0.65:
        final_score = speech_score
    else:
        final_score = combined_score

    result = {
        "start": start,
        "end": end,
        "duration": end - start,
        "label": normalize_label(segment.get("label", "content")),
        "non_content_score": float(final_score),
        "source": "llm_full_transcript",
        "speech_score": speech_score,
        "audio_boundary_score": audio_score,
        "music_inside": music_inside,
        "music_surrounding": music_surrounding,
        "music_delta": music_delta,
        "low_speech_score": 0.0,
        "reason": segment.get("reason", "")
    }

    return result


def build_audio_boundary_result(candidate, transcript_segments, music_windows):
    start = float(candidate["start"])
    end = float(candidate["end"])

    music_inside, music_surrounding, music_delta = music_features_for_interval(music_windows, start, end)
    silence_score = low_speech_score(transcript_segments, start, end)

    audio_boundary_score = clamp_score(float(candidate["score"]) / 2.0)
    music_delta_score = clamp_score(music_delta)
    music_presence_score = clamp_score(music_inside)
    duration_prior = duration_prior_score(end - start)
    shift_score = transcript_shift_score(transcript_segments, start, end)

    combined_score = 0.35 * audio_boundary_score + 0.30 * music_delta_score + 0.20 * music_presence_score + 0.15 * silence_score

    result = {
        "start": start,
        "end": end,
        "duration": end - start,
        "label": "ad",
        "non_content_score": float(combined_score),
        "source": "audio_boundary_fallback",
        "speech_score": 0.0,
        "audio_boundary_score": audio_boundary_score,
        "music_inside": music_inside,
        "music_surrounding": music_surrounding,
        "music_delta": music_delta,
        "low_speech_score": silence_score,
        "duration_prior": duration_prior,
        "transcript_shift_score": shift_score,
        "structural_audio_score": 0.0,
        "candidate_raw_boundary_score": float(candidate["raw_boundary_score"]),
        "candidate_skipped_peaks": int(candidate["skipped_peaks"]),
        "reason": "Detected by paired audio boundary changes, music behavior, and low speech density."
    }
    result["structural_audio_score"] = structural_audio_score(result)

    return result


def generate_music_candidates(music_windows, transcript_segments, min_duration=15.0, max_duration=140.0, music_threshold=0.45, low_speech_threshold=0.45):
    candidates = []
    current_start = None
    current_end = None
    current_scores = []

    for window in music_windows:
        start = float(window["start"])
        end = float(window["end"])
        music_score = float(window["music_score"])

        silence_score = low_speech_score(transcript_segments, start, end)

        if music_score >= music_threshold and silence_score >= low_speech_threshold:
            if current_start is None:
                current_start = start
                current_end = end
                current_scores = [music_score]
            else:
                current_end = end
                current_scores.append(music_score)
        else:
            if current_start is not None:
                duration = current_end - current_start

                if min_duration <= duration <= max_duration:
                    candidates.append({
                        "start": float(current_start),
                        "end": float(current_end),
                        "duration": float(duration),
                        "score": float(sum(current_scores) / len(current_scores)),
                        "source": "music_region"
                    })

                current_start = None
                current_end = None
                current_scores = []

    if current_start is not None:
        duration = current_end - current_start

        if min_duration <= duration <= max_duration:
            candidates.append({
                "start": float(current_start),
                "end": float(current_end),
                "duration": float(duration),
                "score": float(sum(current_scores) / len(current_scores)),
                "source": "music_region"
            })

    return candidates


def build_music_region_result(candidate, transcript_segments, music_windows, peak_times, peak_scores):
    # Convert a sustained music region into a fallback candidate with nearby boundary support.
    start = float(candidate["start"])
    end = float(candidate["end"])

    music_inside, music_surrounding, music_delta = music_features_for_interval(music_windows, start, end)
    silence_score = low_speech_score(transcript_segments, start, end)

    audio_boundary_score = nearest_boundary_score(start, end, peak_times, peak_scores, tolerance_sec=8.0)
    audio_boundary_score = clamp_score(audio_boundary_score)

    music_presence_score = clamp_score(music_inside)
    music_delta_score = clamp_score(music_delta)

    combined_score = 0.40 * music_delta_score + 0.30 * audio_boundary_score + 0.20 * silence_score + 0.10 * music_presence_score

    result = {
        "start": start,
        "end": end,
        "duration": end - start,
        "label": "ad",
        "non_content_score": float(combined_score),
        "source": "music_region_fallback",
        "speech_score": 0.0,
        "audio_boundary_score": audio_boundary_score,
        "music_inside": music_inside,
        "music_surrounding": music_surrounding,
        "music_delta": music_delta,
        "low_speech_score": silence_score,
        "music_region_score": float(candidate["score"]),
        "reason": "Detected by sustained music with low speech density."
    }

    return result


def analyze_single_video(video_path, output_json_path, force_recompute=False):
    video_path = Path(video_path)
    output_json_path = Path(output_json_path)

    cache_dir = Path("cache") / video_path.stem
    wav_path = cache_dir / "audio.wav"

    if force_recompute or not wav_path.exists():
        extract_audio(video_path, wav_path)

    # transcribe the full audio
    transcript_cache_path = cache_dir / "transcript.json"
    transcript_segments = get_cached_json(
        transcript_cache_path,
        lambda: transcribe_audio(wav_path, model_size="large-v3-turbo", device="cuda", compute_type="float16"),
        force_recompute=force_recompute
    )

    theme_cache_path = cache_dir / "video_theme.json"
    theme_cache = get_cached_json(
        theme_cache_path,
        lambda: {"video_theme": summarize_video_theme(transcript_segments)},
        force_recompute=force_recompute
    )
    video_theme = theme_cache["video_theme"]

    # Use gemma4 to detect verbal non-content
    speech_cache_path = cache_dir / "speech_segments.json"
    speech_segments = get_cached_json(
        speech_cache_path,
        lambda: detect_non_content_from_full_transcript(transcript_segments, video_theme),
        force_recompute=force_recompute
    )

    # compute audio and music boundaries to use as secondary evidence
    boundary_cache_path = cache_dir / "boundary_peaks.json"
    boundary_cache = get_cached_json(
        boundary_cache_path,
        lambda: boundary_peaks(wav_path),
        force_recompute=force_recompute
    )
    peak_times = boundary_cache["peak_times"]
    peak_scores = boundary_cache["peak_scores"]

    music_cache_path = cache_dir / "music_windows.json"
    music_windows = get_cached_json(
        music_cache_path,
        lambda: compute_music_windows(wav_path, window_sec=10.0, hop_sec=2.0),
        force_recompute=force_recompute
    )

    results = []
    kept_speech_segments = 0
    kept_structural_audio_candidates = 0

    # speech detection from transcript as primary evidence
    for segment in speech_segments:
        if keep_speech_segment(segment):
            result = build_speech_result(segment, music_windows, peak_times, peak_scores)
            results.append(result)
            kept_speech_segments += 1

    # gGenerate fallback candidates by pairing strong audio boundary changes
    audio_candidates = generate_candidate_intervals(peak_times, peak_scores, min_duration=15, max_duration=140)
    structural_audio_candidates = []

    for candidate in audio_candidates:
        start = float(candidate["start"])
        end = float(candidate["end"])

        if overlaps_segment(results, start, end):
            continue
        else:
            result = build_audio_boundary_result(candidate, transcript_segments, music_windows)

            if float(result["duration"]) >= 25.0 and int(result.get("candidate_skipped_peaks", 0)) <= 1:
                structural_audio_candidates.append(result)

    filtered_structural_audio_candidates = []

    for candidate in structural_audio_candidates:
        if keep_structural_audio_result(candidate):
            filtered_structural_audio_candidates.append(candidate)

    structural_audio_candidates = filtered_structural_audio_candidates

    structural_audio_candidates = sorted(
        structural_audio_candidates,
        key=lambda segment: segment["structural_audio_score"],
        reverse=True
    )

    selected_structural_audio_candidates = []

    for proposal in structural_audio_candidates:
        if overlaps_segment(results + selected_structural_audio_candidates, float(proposal["start"]), float(proposal["end"])):
            continue

        if near_structural_candidate(proposal, selected_structural_audio_candidates):
            continue

        proposal["label"] = "ad"
        proposal["non_content_score"] = max(
            float(proposal["non_content_score"]),
            float(proposal["structural_audio_score"])
        )
        proposal["source"] = "audio_boundary_structural"
        proposal["reason"] = "Detected by structural audio score, ad-like duration, music change, transcript shift, and speech density."
        selected_structural_audio_candidates.append(proposal)
        results.append(proposal)
        kept_structural_audio_candidates += 1

        if kept_structural_audio_candidates >= MAX_STRUCTURAL_AUDIO_RESULTS:
            break

    # generate fallback candidates from sustained music regions with low speech
    music_candidates = generate_music_candidates(music_windows, transcript_segments, min_duration=15.0, max_duration=140.0, music_threshold=0.45, low_speech_threshold=0.45)

    for candidate in music_candidates:
        start = float(candidate["start"])
        end = float(candidate["end"])

        if overlaps_segment(results, start, end):
            continue
        else:
            result = build_music_region_result(candidate, transcript_segments, music_windows, peak_times, peak_scores)

            if keep_music_region_result(result):
                results.append(result)

    results = sorted(results, key=lambda segment: segment["start"])

    # merge consecutive segments
    merged_results = merge_consecutive_segments(results, max_gap=8.0, max_merged_duration=180.0)
    ad_segments = []

    for segment in merged_results:
        ad_segments.append(format_ad_segment(segment))

    formatted_transcript_segments = format_transcript_segments(transcript_segments)

    output = {
        "video_filename": video_path.name,
        "module": "audio_speech_music",
        "ad_segments": ad_segments,
        "transcript_segments": formatted_transcript_segments
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, "w") as file:
        json.dump(output, file, indent=2)

    return output


def find_video_files(input_dir):
    input_dir = Path(input_dir)
    extensions = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions)


def run_all_videos(input_dir="videos_with_ads", output_dir="audio_outputs", force_recompute=False):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    video_files = find_video_files(input_dir)
    outputs = []

    for video_path in video_files:
        output_json_path = output_dir / f"{video_path.stem}_audio.json"
        output = analyze_single_video(video_path, output_json_path, force_recompute=force_recompute)
        outputs.append(output)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_videos": len(video_files),
        "videos": []
    }

    for output in outputs:
        summary["videos"].append({
            "video_filename": output["video_filename"],
            "num_ad_segments": len(output["ad_segments"])
        })

    summary_path = output_dir / "summary_audio.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w") as file:
        json.dump(summary, file, indent=2)

    return outputs


def run_audio_speech_music(video_path, output_json_path, force_recompute=False):
    # Integration entry point for callers that want to process one video directly.
    output = analyze_single_video(video_path, output_json_path, force_recompute=force_recompute)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="videos_with_ads")
    parser.add_argument("--output-dir", default="audio_outputs")
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()

    run_all_videos(args.input_dir, args.output_dir, force_recompute=args.force_recompute)
