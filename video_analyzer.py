import cv2
import numpy as np
import json
import argparse
import sys
import os
from collections import Counter

# thresholds - tuned by testing on a few sample videos
MOTION_LOW = 3.0       # basically no movement
MOTION_MED = 15.0      # low motion but something is happening
SHOT_CUT = 40.0        # big jump = scene cut
STATIC_VAR = 500.0     # low variance = uniform looking frame
DARK_THRESH = 20.0     # very dark = probably black screen
SAMPLE_N = 2           # sample every Nth frame to go faster
SMOOTH_WIN = 8.0       # smoothing window in seconds
MIN_SEG = 10.0         # ignore segments shorter than this


def get_frame_info(prev_gray, curr_gray, curr_frame):
    # compute motion by diffing consecutive grayscale frames
    diff = cv2.absdiff(curr_gray, prev_gray).astype(np.float32)
    motion = float(diff.mean())

    # use HSV to get saturation and brightness info
    hsv = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
    sat = float(hsv[:, :, 1].mean())
    bri = float(hsv[:, :, 2].mean())
    var = float(curr_gray.astype(np.float32).var())

    return motion, sat, bri, var


def label_frame(motion, bri, var):
    # black/dead screen
    if bri < DARK_THRESH and var < STATIC_VAR:
        return "dead_air"
    # static holding screen - low motion and uniform frame
    if motion < MOTION_LOW and var < STATIC_VAR:
        return "static_screen"
    # talking head or slide - something there but not much movement
    if motion < MOTION_MED:
        return "low_motion"
    return "content"


def smooth(labels, window):
    # sliding majority vote to clean up noisy labels
    result = []
    n = len(labels)
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        best = Counter(labels[lo:hi]).most_common(1)[0][0]
        result.append(best)
    return result


def make_segments(labels, times, motions):
    segs = []
    if not labels:
        return segs

    cur = labels[0]
    start = times[0]

    for i in range(1, len(labels)):
        # force a new segment on shot cuts even if label didn't change
        if labels[i] != cur or motions[i] > SHOT_CUT:
            segs.append({"start": round(start, 2), "end": round(times[i], 2), "type": cur})
            cur = labels[i]
            start = times[i]

    segs.append({"start": round(start, 2), "end": round(times[-1], 2), "type": cur})
    return segs


def drop_short(segs):
    # keep merging until no segment is shorter than MIN_SEG
    changed = True
    while changed:
        changed = False
        out = []
        i = 0
        while i < len(segs):
            dur = segs[i]["end"] - segs[i]["start"]
            if dur < MIN_SEG and len(segs) > 1:
                if i + 1 < len(segs):
                    segs[i + 1]["start"] = segs[i]["start"]
                elif out:
                    out[-1]["end"] = segs[i]["end"]
                changed = True
                i += 1
                continue
            out.append(segs[i])
            i += 1
        segs = out
    return segs


def merge_same(segs):
    # collapse adjacent segments with the same label
    if not segs:
        return segs
    out = [dict(segs[0])]
    for s in segs[1:]:
        if s["type"] == out[-1]["type"]:
            out[-1]["end"] = s["end"]
        else:
            out.append(dict(s))
    return out


def refine_labels(segs, total):
    # use position in video to upgrade some labels to intro/outro
    intro_cutoff = min(120.0, total * 0.15)
    outro_cutoff = min(120.0, total * 0.10)

    out = []
    for s in segs:
        label = s["type"]
        dur = round(s["end"] - s["start"], 2)

        if s["start"] < intro_cutoff and label in ("static_screen", "low_motion", "dead_air"):
            label = "intro"
        elif s["end"] > (total - outro_cutoff) and label in ("static_screen", "low_motion", "dead_air"):
            label = "outro"
        elif label == "dead_air" and dur < 2.0:
            label = "content"

        out.append({"start": s["start"], "end": s["end"], "type": label, "duration": dur})
    return out


def analyze(video_path, out_path=None, verbose=True, sample_n=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video:", video_path)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps
    step = sample_n if sample_n else SAMPLE_N

    if verbose:
        print(f"Video: {os.path.basename(video_path)}")
        print(f"Resolution: {w}x{h}, FPS: {fps:.1f}, Duration: {duration:.1f}s")
        print()

    labels = []
    times = []
    motions = []
    prev_gray = None
    idx = 0
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            t = idx / fps

            if prev_gray is not None:
                motion, sat, bri, var = get_frame_info(prev_gray, gray, frame)
                lbl = label_frame(motion, bri, var)
                labels.append(lbl)
                times.append(t)
                motions.append(motion)
                count += 1

                if verbose and count % 300 == 0:
                    print(f"  {100*idx/total_frames:.1f}%  t={t:.0f}s  {lbl}  motion={motion:.1f}")

            prev_gray = gray
        idx += 1

    cap.release()

    if not labels:
        print("No frames analyzed")
        sys.exit(1)

    if verbose:
        print(f"\nAnalyzed {count} frames, building segments...")

    win = max(1, int((SMOOTH_WIN * fps) / step))
    smoothed = smooth(labels, win)
    segs = make_segments(smoothed, times, motions)
    segs = drop_short(segs)
    segs = refine_labels(segs, duration)
    segs = merge_same(segs)

    # add duration to any segment missing it
    for s in segs:
        if "duration" not in s:
            s["duration"] = round(s["end"] - s["start"], 2)

    totals = {}
    for s in segs:
        totals[s["type"]] = totals.get(s["type"], 0) + s["duration"]

    output = {
        "source": os.path.basename(video_path),
        "duration": round(duration, 2),
        "fps": round(fps, 2),
        "resolution": f"{w}x{h}",
        "segments": segs,
        "summary": {k: round(v, 1) for k, v in totals.items()}
    }

    if out_path is None:
        base = os.path.splitext(video_path)[0]
        out_path = base + ".segments.json"

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    if verbose:
        print(f"\nSegments: {len(segs)}")
        print(f"Output: {out_path}\n")
        for s in segs:
            sm, ss = divmod(s["start"], 60)
            em, es = divmod(s["end"], 60)
            print(f"  {int(sm):02d}:{ss:04.1f} -> {int(em):02d}:{es:04.1f}  {s['type']}  ({s['duration']:.0f}s)")
        print()
        for t, v in sorted(totals.items(), key=lambda x: -x[1]):
            print(f"  {t}: {v:.0f}s ({100*v/duration:.1f}%)")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS576 Project - Video Analyzer")
    parser.add_argument("video", help="input video file")
    parser.add_argument("--output", default=None, help="output json path")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--sample-every", type=int, default=None, help="sample every N frames")
    args = parser.parse_args()

    analyze(args.video, args.output, verbose=not args.quiet, sample_n=args.sample_every)
