from video.video_analyzer import analyze 
from text.text_analyzer import pipeline
from audio.audio_speech_music_pipeline import run_audio_speech_music
from pathlib import Path

import json
import argparse
import sys
import os

import json

import json

def convert_video_analysis(data):
    segments = data.get("segments", [])
    total_duration = data.get("duration", 0)
    
    if not segments:
        return []
    
    if not segments:
        return []

    output = []
    content_count = 1
    ad_count = 1
    last_end_time = 0.0

    for i, seg in enumerate(segments):
        start = float(seg["start"])
        end = float(seg["end"])
        seg_type = seg["type"]

        # --- Conditional Clamping ---
        # Only clamp if the specific type matches the position
        if seg_type == "intro" and i == 0:
            start = 0.0
        
        if seg_type == "outro" and i == len(segments) - 1:
            end = total_duration

        # --- Gap Handling ---
        # If there's a gap between the end of the last part and the start of this one
        if start > last_end_time:
            output.append({
                "title": f"Segment {content_count}",
                "type": "content",
                "start": round(last_end_time, 3)
            })
            content_count += 1

        # --- Segment Mapping ---
        item = {"start": round(start, 3)}
        
        if seg_type == "intro":
            item["title"] = "Intro"
            item["type"] = "intro"
        elif seg_type == "outro":
            item["title"] = "Outro"
            item["type"] = "outro"
        elif seg_type == "ad":
            item["title"] = f"Ad Break {ad_count}"
            item["type"] = "ad"
            ad_count += 1
        else:
            # Treats "low_motion" or any other undefined type as content
            item["title"] = f"Segment {content_count}"
            item["type"] = "content"
            content_count += 1
            
        output.append(item)
        last_end_time = end

    # --- Final Tail Check ---
    # If the video continues after the last defined segment
    if last_end_time < total_duration:
        output.append({
            "title": f"Segment {content_count}",
            "type": "content",
            "start": round(last_end_time, 3)
        })

    return output


# Example usage with your provided data
input_data = {
  "video": {
    "source": "test_001.mp4",
    "duration": 1458.43,
    "fps": 29.88,
    "resolution": "640x360",
    "segments": [
      {
        "start": 0.07,
        "end": 225.07,
        "type": "intro",
        "duration": 106.57
      },
      {
        "start": 225.07,
        "end": 1115.63,
        "type": "low_motion",
        "duration": 405.48
      },
      {
        "start": 1115.63,
        "end": 1450.39,
        "type": "outro",
        "duration": 342.76
      }
    ]
  }
}

def convert_text_analysis(data):
    segments = data.get("segments", [])
    total_duration = data.get("duration", 0)
    
    if not segments:
        return []

    output = []
    content_count = 1
    ad_count = 1
    last_end_time = 0.0
    
    # Track if we've already assigned the Intro
    has_intro = False

    for i, seg in enumerate(segments):
        start = float(seg["start"])
        end = float(seg["end"])
        label = seg["label"]

        # --- Gap Handling ---
        # Fill time between the previous segment's end and current segment's start
        if start > last_end_time:
            output.append({
                "title": f"Segment {content_count}",
                "type": "content",
                "start": round(last_end_time, 3)
            })
            content_count += 1

        # --- Segment Mapping ---
        item = {"start": round(start, 3)}
        
        if label == "intro/outro":
            if not has_intro:
                item["title"] = "Intro"
                item["type"] = "intro"
                item["start"] = 0.0 if i == 0 else item["start"] # Optional: clamp to 0 if first
                has_intro = True
            else:
                item["title"] = "Outro"
                item["type"] = "outro"
                # Optional: clamp to total_duration if it's the last segment
                if i == len(segments) - 1:
                    end = total_duration
                    
        elif label == "sponsorship/advertisement":
            item["title"] = f"Ad Break {ad_count}"
            item["type"] = "ad"
            ad_count += 1
            
        else:
            # Fallback for any other labels
            item["title"] = f"Segment {content_count}"
            item["type"] = "content"
            content_count += 1
            
        output.append(item)
        last_end_time = end

    # --- Final Tail Check ---
    # Catch any remaining video time after the last segment
    if last_end_time < total_duration:
        output.append({
            "title": f"Segment {content_count}",
            "type": "content",
            "start": round(last_end_time, 3)
        })

    return output
def convert_audio_analysis(data):
    ad_segments = data.get("ad_segments", [])
    # We need the total duration to fill the final segment. 
    # If not in audio data, we'll try to get it from transcript segments.
    total_duration = 0.0
    if data.get("transcript_segments"):
        total_duration = data["transcript_segments"][-1]["end"]

    output = []
    content_count = 1
    ad_count = 1
    last_end_time = 0.0

    # Sort ads by start time just in case
    sorted_ads = sorted(ad_segments, key=lambda x: x["start"])

    for ad in sorted_ads:
        start = float(ad["start"])
        end = float(ad["end"])

        # 1. If there is a gap before this ad, it's Content
        if start > last_end_time:
            output.append({
                "title": f"Segment {content_count}",
                "type": "content",
                "start": round(last_end_time, 3)
            })
            content_count += 1

        # 2. Add the Ad Break
        output.append({
            "title": f"Ad Break {ad_count}",
            "type": "ad",
            "start": round(start, 3)
        })
        ad_count += 1
        last_end_time = end

    # 3. Final Tail Check: Fill from last ad end to total duration
    if last_end_time < total_duration:
        output.append({
            "title": f"Segment {content_count}",
            "type": "content",
            "start": round(last_end_time, 3)
        })

    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS576 Project - Video Analyzer")
    parser.add_argument("video", help="input video file")
    args = parser.parse_args()

    results = {}

    video_analysis = analyze(args.video, verbose=False)
    results['video'] = convert_video_analysis(video_analysis)
    
    text_analysis = pipeline(args.video, verbose=False)
    results['text'] = convert_text_analysis(text_analysis)


    base_dir = os.path.dirname(os.path.abspath(__file__))
    temp_audio_json = os.path.join(base_dir, "audio", "audio_outputs", f"{Path(args.video).stem}_audio.json")
    
    audio_analysis = run_audio_speech_music(args.video, temp_audio_json)
    results['audio'] = convert_audio_analysis(audio_analysis)

    print(json.dumps(results))