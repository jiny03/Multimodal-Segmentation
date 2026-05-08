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

TEXT_AD_LABELS = {"sponsorship/advertisement"}
VISUAL_ONLY_MAX_DURATION = 90.0
MERGE_GAP_SECONDS = 5.0
FUSED_SCORE_THRESHOLD = 0.58
VISUAL_ONLY_SCORE_THRESHOLD = 0.75
SHORT_AUDIO_MAX_DURATION = 12.0
SHORT_AUDIO_VISUAL_LOOKAHEAD = 45.0
SHORT_AUDIO_EXTENSION_MAX_DURATION = 45.0
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


def visual_reliability_from_data(visual_data):
    confidence = visual_data.get("confidence", {})
    score_spread = float(confidence.get("score_spread", 0.0))
    boundary_peakiness = float(confidence.get("boundary_peakiness", 0.0))

    if score_spread >= 0.40 and boundary_peakiness >= 0.45:
        return "high"

    if score_spread >= 0.25:
        return "medium"

    return "low"


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


def load_audio_segments(path):
    if not path.exists():
        return []

    data = read_json(path)
    segments = data.get("ad_segments", [])
    transcript_segments = data.get("transcript_segments", [])
    results = []

    for segment in segments:
        score = clamp_score(segment.get("non_content_score", 0.85))
        speech_start, speech_end = transcript_bounds_for_segment(segment, transcript_segments)
        results.append({
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "label": "ad",
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
        if segment.get("label") not in TEXT_AD_LABELS:
            continue

        results.append({
            "start": float(segment["start"]),
            "end": float(segment["end"]) + 2.0,
            "label": "ad",
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

    return {
        "start": round(start, 2),
        "end": round(end, 2),
        "duration": round(end - start, 2),
        "label": "ad",
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

    if sources == {"text"} and has_support(cluster, raw_candidates, "visual"):
        return True

    return False


def integrate_video(stem, audio_dir, visual_dir, text_dir):
    video_profile = build_video_profile(stem, audio_dir, visual_dir)
    audio_segments = load_audio_segments(audio_dir / f"{stem}_audio.json")
    visual_segments = load_visual_segments(visual_dir / f"{stem}_visual.json")
    text_segments = load_text_segments(text_dir / f"{stem}_text.json")
    audio_segments = extend_short_audio_segments(audio_segments, visual_segments, video_profile)
    raw_candidates = audio_segments + visual_segments + text_segments
    merged_candidates = merge_candidates(raw_candidates, video_profile)
    final_segments = []

    for candidate in merged_candidates:
        if keep_cluster(candidate, raw_candidates, video_profile):
            final_segments.append(candidate)

    return {
        "video_id": stem,
        "module": "backend_integration",
        "audio_pipeline": str(AUDIO_PIPELINE_PATH),
        "video_profile": video_profile,
        "ad_segments": final_segments,
        "debug_counts": {
            "audio_candidates": len(audio_segments),
            "visual_candidates": len(visual_segments),
            "text_candidates": len(text_segments),
        },
    }


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
