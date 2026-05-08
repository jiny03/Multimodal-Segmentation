from pathlib import Path
import argparse
import importlib
import importlib.util
import json
import subprocess
import sys


# Canonical paths for component modules
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
AUDIO_PIPELINE_PATH = BACKEND_DIR / "audio" / "audio_speech_music_pipeline.py"
VIDEO_ANALYZER_PATH = BACKEND_DIR / "video" / "video_analyzer.py"
TEXT_ANALYZER_PATH = BACKEND_DIR / "text" / "text_analyzer.py"

TEXT_LABEL_MAP = {
    "sponsorship/advertisement": "ad",
    "intro/outro": "intro/outro",
    "transition / intermission": "transition / intermission",
    "recap": "recap",
}
VISUAL_ONLY_MAX_DURATION = 90.0
MERGE_GAP_SECONDS = 5.0
FUSED_SCORE_THRESHOLD = 0.58
VISUAL_ONLY_SCORE_THRESHOLD = 0.75
SHORT_AUDIO_MAX_DURATION = 12.0
SHORT_AUDIO_VISUAL_LOOKAHEAD = 45.0
SHORT_AUDIO_EXTENSION_MAX_DURATION = 45.0
AUDIO_VISUAL_TRIM_MIN_LEAD = 20.0
AUDIO_VISUAL_TRIM_MIN_REMAINING = 15.0
INTRO_MIN_DURATION = 5.0
INTRO_MAX_DURATION = 75.0
INTRO_BOUNDARY_WINDOW = 20.0
INTRO_VISUAL_BOUNDARY_THRESHOLD = 0.60
INTRO_AUDIO_BOUNDARY_THRESHOLD = 0.35
OUTRO_MIN_DURATION = 5.0
OUTRO_MAX_DURATION = 20.0
OUTRO_DURATION_FRACTION = 0.022
OUTRO_BOUNDARY_SEARCH = 60.0
OUTRO_VISUAL_BOUNDARY_THRESHOLD = 0.60
SOURCE_WEIGHTS = {
    "audio_original": 0.70,
    "visual": 0.55,
    "text": 0.35,
}


def assert_component_paths():
    for path in [AUDIO_PIPELINE_PATH, VIDEO_ANALYZER_PATH, TEXT_ANALYZER_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Component script not found: {path}")


def read_json(path):
    with open(path) as file:
        return json.load(file)


def overlap_seconds(first, second):
    return max(0.0, min(first["end"], second["end"]) - max(first["start"], second["start"]))


def segments_overlap(first, second, gap_seconds=0.0):
    return first["start"] <= second["end"] + gap_seconds and second["start"] <= first["end"] + gap_seconds


def clamp_score(score):
    return max(0.0, min(1.0, float(score)))


def normalize_non_content_label(label, default_label="ad"):
    if label is None:
        return default_label

    normalized_label = str(label).strip().lower()

    if normalized_label in {"", "content", "core content"}:
        return None

    if normalized_label == "sponsorship/advertisement":
        return "ad"

    return normalized_label


def load_optional_json(path):
    if not path.exists():
        return {}

    return read_json(path)


def speech_coverage_from_audio(audio_data):
    transcript_segments = audio_data.get("transcript_segments", [])

    if len(transcript_segments) == 0:
        return 0.0

    video_duration = max(float(segment.get("end", 0.0)) for segment in transcript_segments)

    if video_duration <= 0.0:
        return 0.0

    speech_duration = 0.0

    for segment in transcript_segments:
        speech_duration += max(0.0, float(segment.get("end", 0.0)) - float(segment.get("start", 0.0)))

    return clamp_score(speech_duration / video_duration)


def music_profile_from_cache(stem):
    path = Path("cache") / stem / "music_windows.json"

    if not path.exists():
        return 0.0, 0.0, []

    music_windows = read_json(path)

    if len(music_windows) == 0:
        return 0.0, 0.0, []

    scores = [float(window.get("music_score", 0.0)) for window in music_windows]
    mean_music = sum(scores) / len(scores)
    high_music_windows = [window for window in music_windows if float(window.get("music_score", 0.0)) >= 0.30]
    high_music_fraction = len(high_music_windows) / len(music_windows)
    high_music_ranges = []
    range_in_progress = None

    for window in high_music_windows:
        start = float(window["start"])
        end = float(window["end"])

        if range_in_progress is None:
            range_in_progress = {"start": start, "end": end}
        else:
            if start <= range_in_progress["end"] + 4.0:
                range_in_progress["end"] = end
            else:
                high_music_ranges.append(range_in_progress)
                range_in_progress = {"start": start, "end": end}

    if range_in_progress is not None:
        high_music_ranges.append(range_in_progress)

    return mean_music, high_music_fraction, high_music_ranges[:5]


def audio_boundary_peaks_from_cache(stem):
    path = Path("cache") / stem / "boundary_peaks.json"

    if not path.exists():
        return []

    data = read_json(path)
    peak_times = data.get("peak_times", [])
    peak_scores = data.get("peak_scores", [])
    peaks = []

    for index, peak_time in enumerate(peak_times):
        if index >= len(peak_scores):
            continue

        peaks.append({
            "time": float(peak_time),
            "score": float(peak_scores[index]),
        })

    return peaks


def visual_reliability_from_data(visual_data):
    confidence = visual_data.get("confidence", {})
    score_spread = float(confidence.get("score_spread", 0.0))
    boundary_peakiness = float(confidence.get("boundary_peakiness", 0.0))

    if score_spread >= 0.40 and boundary_peakiness >= 0.45:
        return "high"

    if score_spread >= 0.25:
        return "medium"

    return "low"


def video_duration_from_outputs(audio_data, visual_data):
    if visual_data.get("duration"):
        return float(visual_data["duration"])

    transcript_segments = audio_data.get("transcript_segments", [])

    if len(transcript_segments) > 0:
        return max(float(segment.get("end", 0.0)) for segment in transcript_segments)

    return 0.0


def classify_video_type(speech_coverage, mean_music, high_music_fraction, visual_reliability):
    rich_speech = speech_coverage >= 0.35
    music_present = mean_music >= 0.25 or high_music_fraction >= 0.25

    if rich_speech and music_present:
        return "speech_music_mixed"

    if rich_speech:
        return "speech_heavy"

    if music_present:
        return "music_heavy"

    if visual_reliability == "high":
        return "visual_driven"

    return "unknown"


def build_video_profile(stem, audio_dir, visual_dir):
    audio_data = load_optional_json(audio_dir / f"{stem}_audio.json")
    visual_data = load_optional_json(visual_dir / f"{stem}_visual.json")
    speech_coverage = speech_coverage_from_audio(audio_data)
    mean_music, high_music_fraction, high_music_ranges = music_profile_from_cache(stem)
    visual_reliability = visual_reliability_from_data(visual_data)
    video_type = classify_video_type(speech_coverage, mean_music, high_music_fraction, visual_reliability)

    return {
        "video_type": video_type,
        "rich_speech": speech_coverage >= 0.35,
        "music_present": mean_music >= 0.25 or high_music_fraction >= 0.25,
        "speech_coverage": round(speech_coverage, 4),
        "mean_music_score": round(mean_music, 4),
        "high_music_fraction": round(high_music_fraction, 4),
        "high_music_ranges": high_music_ranges,
        "visual_reliability": visual_reliability,
    }


def source_weights_for_profile(profile):
    weights = dict(SOURCE_WEIGHTS)
    video_type = profile.get("video_type", "unknown")

    if video_type in {"speech_heavy", "speech_music_mixed"}:
        weights["audio_original"] += 0.08
        weights["text"] += 0.04
        weights["visual"] -= 0.04

    if video_type == "music_heavy":
        weights["audio_original"] += 0.04
        weights["visual"] += 0.04

    if video_type == "visual_driven":
        weights["visual"] += 0.08

    if profile.get("visual_reliability") == "low":
        weights["visual"] -= 0.12

    return weights


def make_intro_outro_segment(label, start, end, sources, score=0.55):
    return {
        "start": round(start, 2),
        "end": round(end, 2),
        "duration": round(end - start, 2),
        "label": label,
        "score": score,
        "sources": sources,
        "source_scores": {source: score for source in sources},
    }


def load_audio_segments(path):
    if not path.exists():
        return []

    data = read_json(path)
    segments = data.get("non_content_segments", data.get("ad_segments", []))
    transcript_segments = data.get("transcript_segments", [])
    results = []

    for segment in segments:
        label = normalize_non_content_label(segment.get("label"), default_label="ad")

        if label is None:
            continue

        score = clamp_score(segment.get("non_content_score", 0.85))
        speech_start, speech_end = transcript_bounds_for_segment(segment, transcript_segments)
        results.append({
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "label": label,
            "score": score,
            "sources": ["audio_original"],
            "source_scores": {"audio_original": score},
            "audio_source_label": segment.get("source", ""),
            "speech_start": speech_start,
            "speech_end": speech_end,
        })

    return results


def transcript_bounds_for_segment(segment, transcript_segments):
    matching_segments = []

    for transcript_segment in transcript_segments:
        if overlap_seconds(segment, transcript_segment) > 0.0:
            matching_segments.append(transcript_segment)

    if len(matching_segments) == 0:
        return None, None

    speech_start = min(float(transcript_segment["start"]) for transcript_segment in matching_segments)
    speech_end = max(float(transcript_segment["end"]) for transcript_segment in matching_segments)

    return speech_start, speech_end


def average_visual_score(visual_data, start, end):
    scores = []

    for window in visual_data.get("window_scores", []):
        time = float(window["time"])

        if start <= time <= end:
            scores.append(float(window.get("score", 0.0)))

    if len(scores) == 0:
        return 0.0

    return sum(scores) / len(scores)


def load_visual_segments(path):
    if not path.exists():
        return []

    data = read_json(path)
    results = []

    for segment in data.get("ad_candidates", []):
        start = float(segment["start"])
        end = float(segment["end"])
        score = clamp_score(average_visual_score(data, start, end))
        results.append({
            "start": start,
            "end": end,
            "label": "ad",
            "score": score,
            "sources": ["visual"],
            "source_scores": {"visual": score},
        })

    return results


def load_text_segments(path):
    if not path.exists():
        return []

    data = read_json(path)
    results = []

    for segment in data.get("segments", []):
        label = TEXT_LABEL_MAP.get(segment.get("label"))

        if label is None:
            continue

        results.append({
            "start": float(segment["start"]),
            "end": float(segment["end"]) + 2.0,
            "label": label,
            "score": 0.80,
            "sources": ["text"],
            "source_scores": {"text": 0.80},
        })

    return results


def extend_short_audio_segments(audio_segments, visual_segments, profile):
    if profile.get("video_type") != "music_heavy":
        return audio_segments

    extended_segments = []

    for audio_segment in audio_segments:
        start = audio_segment["start"]
        end = audio_segment["end"]
        duration = end - start
        extended_segment = dict(audio_segment)

        if duration <= SHORT_AUDIO_MAX_DURATION and audio_segment["score"] >= 0.95:
            best_visual = None

            for visual_segment in visual_segments:
                if visual_segment["start"] <= end:
                    continue

                gap = visual_segment["start"] - end

                if gap > SHORT_AUDIO_VISUAL_LOOKAHEAD:
                    continue

                if best_visual is None or visual_segment["start"] < best_visual["start"]:
                    best_visual = visual_segment

            if best_visual is not None:
                extended_end = min(best_visual["start"], start + SHORT_AUDIO_EXTENSION_MAX_DURATION)
                extended_segment["end"] = extended_end
                extended_segment["duration"] = extended_end - start
                extended_segment["boundary_refinement"] = "short_audio_extended_to_next_visual_start"

        extended_segments.append(extended_segment)

    return extended_segments


def score_from_sources(source_scores, profile):
    score = 0.0
    source_weights = source_weights_for_profile(profile)

    for source, source_score in source_scores.items():
        score += source_weights.get(source, 0.0) * clamp_score(source_score)

    if len(source_scores) == 2:
        score += 0.12
    else:
        if len(source_scores) >= 3:
            score += 0.20

    return clamp_score(score)


def label_from_cluster(cluster):
    label_scores = {}

    for segment in cluster:
        label = normalize_non_content_label(segment.get("label"))

        if label is None:
            continue

        label_scores[label] = label_scores.get(label, 0.0) + clamp_score(segment.get("score", 0.0))

    if len(label_scores) == 0:
        return "ad"

    return max(label_scores, key=label_scores.get)


def merge_cluster(cluster, profile):
    sources = []
    source_scores = {}
    audio_segments = [segment for segment in cluster if "audio_original" in segment["sources"]]
    visual_segments = [segment for segment in cluster if "visual" in segment["sources"]]

    if len(audio_segments) > 0:
        start = min(segment["start"] for segment in audio_segments)
        end = max(segment["end"] for segment in audio_segments)
        start, end = refine_audio_boundaries(start, end, audio_segments, visual_segments, profile)
    else:
        start = min(segment["start"] for segment in cluster)
        end = max(segment["end"] for segment in cluster)

    for segment in cluster:
        for source in segment["sources"]:
            if source not in sources:
                sources.append(source)

        for source, source_score in segment.get("source_scores", {}).items():
            source_scores[source] = max(source_scores.get(source, 0.0), clamp_score(source_score))

    score = score_from_sources(source_scores, profile)
    label = label_from_cluster(cluster)

    return {
        "start": round(start, 2),
        "end": round(end, 2),
        "duration": round(end - start, 2),
        "label": label,
        "score": round(score, 4),
        "sources": sources,
        "source_scores": {source: round(score, 4) for source, score in source_scores.items()},
    }


def refine_audio_boundaries(start, end, audio_segments, visual_segments, profile):
    if profile.get("video_type") not in {"speech_heavy", "speech_music_mixed"}:
        return start, end

    for audio_segment in audio_segments:
        source_label = audio_segment.get("audio_source_label", "")
        speech_start = audio_segment.get("speech_start")

        if speech_start is None:
            continue

        if "positional_intro_outro" in source_label and "llm_full_transcript" in source_label:
            if start < speech_start and speech_start < end:
                start = speech_start

    if len(visual_segments) > 0:
        visual_start = min(segment["start"] for segment in visual_segments)

        if start + AUDIO_VISUAL_TRIM_MIN_LEAD <= visual_start and visual_start < end:
            if end - visual_start >= AUDIO_VISUAL_TRIM_MIN_REMAINING:
                start = visual_start

        for audio_segment in audio_segments:
            source_label = audio_segment.get("audio_source_label", "")

            if "audio_boundary_structural" not in source_label or "llm_full_transcript" not in source_label:
                continue

            if start + 12.0 < visual_start and visual_start < end:
                start = visual_start

    return start, end


def merge_candidates(candidates, profile):
    sorted_candidates = sorted(candidates, key=lambda segment: segment["start"])
    clusters = []

    for candidate in sorted_candidates:
        if len(clusters) == 0:
            clusters.append([candidate])
            continue

        previous_cluster = clusters[-1]
        previous = merge_cluster(previous_cluster, profile)

        if segments_overlap(previous, candidate, gap_seconds=MERGE_GAP_SECONDS):
            previous_cluster.append(candidate)
        else:
            clusters.append([candidate])

    return [merge_cluster(cluster, profile) for cluster in clusters]


def has_support(candidate, candidates, source_name):
    for other in candidates:
        if source_name not in other["sources"]:
            continue

        if overlap_seconds(candidate, other) > 0.0:
            return True

    return False


def visual_only_score_threshold(profile):
    if profile.get("visual_reliability") == "high" and not profile.get("rich_speech"):
        return 0.70

    return VISUAL_ONLY_SCORE_THRESHOLD


def keep_cluster(cluster, raw_candidates, profile):
    sources = set(cluster["sources"])
    source_scores = cluster.get("source_scores", {})

    if cluster["score"] >= FUSED_SCORE_THRESHOLD:
        return True

    if sources == {"visual"} and source_scores.get("visual", 0.0) >= visual_only_score_threshold(profile) and cluster["duration"] <= VISUAL_ONLY_MAX_DURATION:
        return True

    if sources == {"text"} and cluster.get("label") == "ad" and has_support(cluster, raw_candidates, "visual"):
        return True

    return False


def strongest_audio_boundary(audio_boundary_peaks, start, end, threshold):
    best_peak = None

    for peak in audio_boundary_peaks:
        time = float(peak["time"])

        if time < start or time > end:
            continue

        score = float(peak["score"])

        if score < threshold:
            continue

        if best_peak is None or score > float(best_peak["score"]):
            best_peak = peak

    if best_peak is None:
        return None

    return float(best_peak["time"])


def estimate_intro_end(visual_data, visual_segments, audio_boundary_peaks, video_duration):
    candidate_times = [INTRO_MIN_DURATION]

    if len(visual_segments) > 0:
        first_visual_start = min(segment["start"] for segment in visual_segments)
        candidate_times.append(first_visual_start)
        search_start = max(INTRO_MIN_DURATION, first_visual_start - INTRO_BOUNDARY_WINDOW)
        search_end = min(video_duration, first_visual_start + INTRO_BOUNDARY_WINDOW)

        visual_boundary = snap_time_to_visual_boundary(first_visual_start, visual_data, INTRO_BOUNDARY_WINDOW, INTRO_VISUAL_BOUNDARY_THRESHOLD)

        if visual_boundary is not None:
            candidate_times.append(visual_boundary)

        audio_boundary = strongest_audio_boundary(audio_boundary_peaks, first_visual_start, search_end, INTRO_AUDIO_BOUNDARY_THRESHOLD)

        if audio_boundary is not None:
            candidate_times.append(audio_boundary)
    else:
        visual_boundary = snap_time_to_visual_boundary(INTRO_MIN_DURATION, visual_data, INTRO_MAX_DURATION - INTRO_MIN_DURATION, INTRO_VISUAL_BOUNDARY_THRESHOLD)

        if visual_boundary is not None:
            candidate_times.append(visual_boundary)

    return min(INTRO_MAX_DURATION, max(candidate_times))


def estimate_outro_start(visual_data, video_duration):
    if video_duration <= 0.0:
        return 0.0

    shot_cuts = []

    for shot_cut_time in visual_data.get("shot_cut_times", []):
        time = float(shot_cut_time)

        if video_duration - OUTRO_BOUNDARY_SEARCH <= time <= video_duration - OUTRO_MIN_DURATION:
            shot_cuts.append(time)

    if len(shot_cuts) > 0:
        return max(shot_cuts)

    fallback_time = max(0.0, video_duration - OUTRO_DURATION_FRACTION * video_duration)
    visual_boundary = snap_time_to_visual_boundary(fallback_time, visual_data, OUTRO_BOUNDARY_SEARCH, OUTRO_VISUAL_BOUNDARY_THRESHOLD)

    if visual_boundary is not None:
        return visual_boundary

    outro_duration = min(OUTRO_MAX_DURATION, max(OUTRO_MIN_DURATION, video_duration * OUTRO_DURATION_FRACTION))
    return max(0.0, video_duration - outro_duration)


def add_intro_outro_segments(final_segments, video_duration, visual_data, visual_segments, audio_boundary_peaks):
    if video_duration <= 0.0:
        return final_segments

    intro_end = estimate_intro_end(visual_data, visual_segments, audio_boundary_peaks, video_duration)
    outro_start = estimate_outro_start(visual_data, video_duration)
    intro_segment = make_intro_outro_segment("intro", 0.0, intro_end, ["position", "visual", "audio_boundary"])
    outro_segment = make_intro_outro_segment("outro", outro_start, video_duration, ["position", "visual"])
    trimmed_segments = trim_segments_around_intro_outro(final_segments, intro_segment, outro_segment)
    trimmed_segments.append(intro_segment)
    trimmed_segments.append(outro_segment)

    return sorted(trimmed_segments, key=lambda segment: segment["start"])


def trim_segments_around_intro_outro(final_segments, intro_segment, outro_segment):
    trimmed_segments = []

    for segment in final_segments:
        candidate = dict(segment)

        if candidate["label"] != "intro" and overlap_seconds(candidate, intro_segment) > 0.0:
            candidate["start"] = max(candidate["start"], intro_segment["end"])

        if candidate["label"] != "outro" and overlap_seconds(candidate, outro_segment) > 0.0:
            candidate["end"] = min(candidate["end"], outro_segment["start"])

        candidate["duration"] = round(candidate["end"] - candidate["start"], 2)

        if candidate["duration"] > 0.5:
            candidate["start"] = round(candidate["start"], 2)
            candidate["end"] = round(candidate["end"], 2)
            trimmed_segments.append(candidate)

    return trimmed_segments


def integrate_video(stem, audio_dir, visual_dir, text_dir):
    audio_data = load_optional_json(audio_dir / f"{stem}_audio.json")
    visual_data = load_optional_json(visual_dir / f"{stem}_visual.json")
    video_profile = build_video_profile(stem, audio_dir, visual_dir)
    audio_segments = load_audio_segments(audio_dir / f"{stem}_audio.json")
    visual_segments = load_visual_segments(visual_dir / f"{stem}_visual.json")
    text_segments = load_text_segments(text_dir / f"{stem}_text.json")
    audio_boundary_peaks = audio_boundary_peaks_from_cache(stem)
    audio_segments = extend_short_audio_segments(audio_segments, visual_segments, video_profile)
    raw_candidates = audio_segments + visual_segments + text_segments
    merged_candidates = merge_candidates(raw_candidates, video_profile)
    final_segments = []

    for candidate in merged_candidates:
        if keep_cluster(candidate, raw_candidates, video_profile):
            final_segments.append(candidate)

    final_segments = snap_boundaries_to_visual(final_segments, visual_data)
    video_duration = video_duration_from_outputs(audio_data, visual_data)
    final_segments = add_intro_outro_segments(final_segments, video_duration, visual_data, visual_segments, audio_boundary_peaks)
    ad_segments = [segment for segment in final_segments if segment.get("label") == "ad"]

    return {
        "video_id": stem,
        "module": "backend_integration",
        "audio_pipeline": str(AUDIO_PIPELINE_PATH),
        "duration": round(video_duration, 2),
        "video_profile": video_profile,
        "non_content_segments": final_segments,
        "ad_segments": ad_segments,
        "debug_counts": {
            "audio_candidates": len(audio_segments),
            "visual_candidates": len(visual_segments),
            "text_candidates": len(text_segments),
        },
    }


def snap_time_to_visual_boundary(time, visual_data, snap_window=15, min_peak=0.40):
    snap_window = int(round(snap_window))
    probe_segment = {
        "start": float(time),
        "end": float(time) + 1.0,
        "duration": 1.0,
        "label": "boundary_probe",
    }
    snapped_segments = snap_boundaries_to_visual([probe_segment], visual_data, snap_window=snap_window, min_peak=min_peak)

    if len(snapped_segments) == 0:
        return None

    snapped_time = float(snapped_segments[0]["start"])

    if snapped_time == float(time):
        return None

    return snapped_time


def snap_boundaries_to_visual(segments, visual_data, snap_window=15, min_peak=0.40):
    # the visual module outputs a boundary_change score for each second
    # it peaks right at ad transitions, so we can use it to correct the start/end times

    window_scores = visual_data.get("window_scores", [])

    if len(window_scores) == 0:
        return segments

    # build a simple list of boundary_change values indexed by second
    max_time = int(float(window_scores[-1]["time"])) + 1
    boundary_scores = [0.0] * max_time

    for window in window_scores:
        t = int(float(window["time"]))

        if t < max_time:
            boundary_scores[t] = float(window.get("boundary_change", 0.0))

    snapped_segments = []

    for segment in segments:
        segment_start = int(float(segment["start"]))
        segment_end = int(float(segment["end"]))

        # look for the strongest visual peak near the segment start
        start_lo = max(0, segment_start - snap_window)
        start_hi = min(max_time, segment_start + snap_window + 1)
        start_region = boundary_scores[start_lo:start_hi]

        if len(start_region) > 0 and max(start_region) >= min_peak:
            new_start = start_lo + start_region.index(max(start_region))
        else:
            new_start = segment_start

        # look for the strongest visual peak near the segment end
        end_lo = max(0, segment_end - snap_window)
        end_hi = min(max_time, segment_end + snap_window + 1)
        end_region = boundary_scores[end_lo:end_hi]

        if len(end_region) > 0 and max(end_region) >= min_peak:
            new_end = end_lo + end_region.index(max(end_region))
        else:
            new_end = segment_end

        # only apply if the result still makes sense
        if new_end <= new_start:
            snapped_segments.append(segment)
            continue

        snapped_segment = dict(segment)
        snapped_segment["start"] = round(float(new_start), 2)
        snapped_segment["end"] = round(float(new_end), 2)
        snapped_segment["duration"] = round(float(new_end - new_start), 2)
        snapped_segments.append(snapped_segment)

    return snapped_segments

def find_video_ids(audio_dir, visual_dir, text_dir):
    video_ids = set()

    for path in audio_dir.glob("test_*_audio.json"):
        video_ids.add(path.stem.replace("_audio", ""))

    for path in visual_dir.glob("test_*_visual.json"):
        video_ids.add(path.stem.replace("_visual", ""))

    for path in text_dir.glob("test_*_text.json"):
        video_ids.add(path.stem.replace("_text", ""))

    return sorted(video_ids)


def run_audio_pipeline(video_dir, audio_output_dir):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    audio_module = importlib.import_module("backend.audio.audio_speech_music_pipeline")
    audio_module.run_all_videos(video_dir, audio_output_dir)


def run_video_analyzer(video_path, output_path):
    subprocess.run(
        [sys.executable, str(VIDEO_ANALYZER_PATH), str(video_path), "--output", str(output_path)],
        check=True,
    )


def run_text_analyzer(video_path, output_path):
    spec = importlib.util.spec_from_file_location("text_analyzer", TEXT_ANALYZER_PATH)
    text_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(text_module)
    result = text_module.pipeline(str(video_path))

    with open(output_path, "w") as file:
        json.dump(result, file, indent=2)


def run_integration(video_dir="demo_video", output_dir="demo_backend_output", skip_analysis=False):
    assert_component_paths()

    video_path = Path(video_dir)
    output_path = Path(output_dir)
    audio_output_path = output_path / "audio_outputs"
    visual_output_path = output_path / "visual_outputs"
    text_output_path = output_path / "text_outputs"

    for path in [output_path, audio_output_path, visual_output_path, text_output_path]:
        path.mkdir(parents=True, exist_ok=True)

    if not skip_analysis:
        run_audio_pipeline(video_dir, audio_output_path)

        for video_file in sorted(video_path.glob("*.mp4")):
            stem = video_file.stem
            run_video_analyzer(video_file, visual_output_path / f"{stem}_visual.json")
            run_text_analyzer(video_file, text_output_path / f"{stem}_text.json")

    outputs = []

    for video_id in find_video_ids(audio_output_path, visual_output_path, text_output_path):
        output = integrate_video(video_id, audio_output_path, visual_output_path, text_output_path)
        outputs.append(output)

        with open(output_path / f"{video_id}_integrated.json", "w") as file:
            json.dump(output, file, indent=2)

    summary = {
        "module": "backend_integration",
        "video_dir": str(video_path),
        "audio_pipeline": str(AUDIO_PIPELINE_PATH),
        "output_dir": str(output_path),
        "videos": [
            {
                "video_id": output["video_id"],
                "num_non_content_segments": len(output["non_content_segments"]),
                "num_ad_segments": len(output["ad_segments"]),
                "debug_counts": output["debug_counts"],
            }
            for output in outputs
        ],
    }

    with open(output_path / "summary_backend_integration.json", "w") as file:
        json.dump(summary, file, indent=2)

    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", default="demo_video")
    parser.add_argument("--output-dir", default="demo_backend_output")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip running component analyzers, only integrate existing outputs")
    args = parser.parse_args()

    run_integration(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        skip_analysis=args.skip_analysis,
    )


if __name__ == "__main__":
    main()
