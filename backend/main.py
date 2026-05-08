import argparse
import json
import subprocess
from pathlib import Path
from backend_integration import run_integration

def get_video_duration(video_path):
    """Uses ffprobe to get the duration of the video file."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())

def convert_to_timeline(integrated_data, total_duration):
    # Extract all non-content segments (ads, intro, outro, etc.)
    raw_segments = integrated_data.get("non_content_segments", [])
    
    # Sort by start time just in case
    raw_segments.sort(key=lambda x: x["start"])

    timeline = []
    current_time = 0.0
    content_count = 1
    ad_count = 1

    for seg in raw_segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        seg_label = seg.get("label", "ad")

        # 1. If there's a gap between current_time and the start of this segment, it's content
        if seg_start > current_time:
            timeline.append({
                "title": f"Segment {content_count}",
                "type": "content",
                "start": round(current_time, 3)
            })
            content_count += 1

        # 2. Add the non-content segment
        label_title = "Ad Break" if seg_label == "ad" else seg_label.capitalize()
        timeline.append({
            "title": f"{label_title} {ad_count}",
            "type": seg_label,
            "start": round(seg_start, 3)
        })
        ad_count += 1
        
        current_time = seg_end

    # 3. If there is time left after the last segment, add a final content block
    if current_time < total_duration:
        timeline.append({
            "title": f"Segment {content_count}",
            "type": "content",
            "start": round(current_time, 3)
        })

    return timeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS576 Project - Video Analyzer")
    parser.add_argument("video", help="input video file")
    args = parser.parse_args()

    video_file = Path(args.video)
    if not video_file.exists():
        print(f"Error: Video file {video_file} not found.")
        exit(1)

    # 1. Run the existing integration pipeline
    # We point to the parent directory as run_integration expects a directory of videos
    video_dir = video_file.parent
    stem = video_file.stem
    
    run_integration(video_dir=str(video_dir), output_dir="demo_backend_output", skip_analysis=False)

    # 2. Load the resulting integrated JSON
    integrated_json_path = Path("demo_backend_output") / f"{stem}_integrated.json"
    if not integrated_json_path.exists():
        print(f"Error: Integration failed to produce {integrated_json_path}")
        exit(1)

    with open(integrated_json_path, "r") as f:
        integrated_data = json.load(f)

    # 3. Get metadata and convert
    duration = get_video_duration(video_file)
    final_timeline = convert_to_timeline(integrated_data, duration)

    # 4. Save/Output the specific format requested
    output_filename = f"{stem}_timeline.json"
    with open(output_filename, "w") as f:
        json.dump(final_timeline, f, indent=2)

    data = {}
    data['audio']=final_timeline
    data['text']=final_timeline
    data['video']=final_timeline
    print(json.dumps(data, indent=2))