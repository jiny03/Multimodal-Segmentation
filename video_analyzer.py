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
    # use position in video to upgrade some labels to intro/outro/ad_break
    intro_cutoff = min(120.0, total * 0.15)
    outro_cutoff = min(120.0, total * 0.10)

    out = []
    for s in segs:
        label = s["type"]
        dur = round(s["end"] - s["start"], 2)
        in_intro = s["start"] < intro_cutoff
        in_outro = s["end"] > (total - outro_cutoff)
        in_middle = not in_intro and not in_outro

        if in_intro and label in ("static_screen", "low_motion", "dead_air"):
            label = "intro"
        elif in_outro and label in ("static_screen", "low_motion", "dead_air"):
            label = "outro"
        elif in_middle and label in ("static_screen", "dead_air"):
            # static or black screen in the middle of the video = likely an ad break or transition
            label = "ad_break"
        elif label == "dead_air" and dur < 2.0:
            label = "content"

        out.append({"start": s["start"], "end": s["end"], "type": label, "duration": dur})
    return out


def compute_window_scores(times, motions, bris, vars_, window_sec=1.0):
    """
    For each 1-second window, compute a non-content score between 0 and 1.
    High score = likely non-content (ad, transition, dead air, intro/outro).
    Low score = likely real content.

    Based on visual features only - meant to be combined with Jin's audio scores.
    Formula: weighted combo of low-motion, low-variance, and darkness signals.
    """
    if not times:
        return []

    total = times[-1]
    scores = []
    t = 0.0

    while t < total:
        t_end = t + window_sec
        # grab all frame samples that fall in this window
        m_vals, b_vals, v_vals = [], [], []
        for i in range(len(times)):
            if t <= times[i] < t_end:
                m_vals.append(motions[i])
                b_vals.append(bris[i])
                v_vals.append(vars_[i])

        if not m_vals:
            # no frames sampled in this window, carry forward 0
            scores.append({"time": round(t, 2), "score": 0.0})
            t += window_sec
            continue

        avg_motion = np.mean(m_vals)
        avg_bri = np.mean(b_vals)
        avg_var = np.mean(v_vals)

        # each component gives 0-1, then weighted together
        # low motion -> more likely non-content
        motion_score = max(0.0, 1.0 - avg_motion / MOTION_MED)
        # low variance -> static screen
        var_score = max(0.0, 1.0 - avg_var / STATIC_VAR)
        # dark frame
        dark_score = max(0.0, 1.0 - avg_bri / 128.0)

        score = round(0.5 * motion_score + 0.3 * var_score + 0.2 * dark_score, 4)
        scores.append({"time": round(t, 2), "score": score})
        t += window_sec

    return scores


def find_ad_candidates(window_scores, threshold=0.55, min_dur=5.0):
    # find runs of consecutive high-score windows and merge them into intervals
    # threshold: score above this = likely non-content
    # min_dur: ignore intervals shorter than this (seconds)
    candidates = []
    in_ad = False
    start = 0.0

    for entry in window_scores:
        t = entry["time"]
        s = entry["score"]
        if s >= threshold and not in_ad:
            in_ad = True
            start = t
        elif s < threshold and in_ad:
            dur = t - start
            if dur >= min_dur:
                candidates.append({"start": round(start, 2), "end": round(t, 2), "duration": round(dur, 2)})
            in_ad = False

    # close out if video ends while still in an ad region
    if in_ad and window_scores:
        t = window_scores[-1]["time"] + 1.0
        dur = t - start
        if dur >= min_dur:
            candidates.append({"start": round(start, 2), "end": round(t, 2), "duration": round(dur, 2)})

    return candidates


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
    bris = []
    vars_ = []
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
                bris.append(bri)
                vars_.append(var)
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

    # per-second window scores for integration with audio module
    window_scores = compute_window_scores(times, motions, bris, vars_, window_sec=1.0)

    # time intervals that are likely ads/non-content based on window scores
    ad_candidates = find_ad_candidates(window_scores)

    totals = {}
    for s in segs:
        totals[s["type"]] = totals.get(s["type"], 0) + s["duration"]

    output = {
        "source": os.path.basename(video_path),
        "duration": round(duration, 2),
        "fps": round(fps, 2),
        "resolution": f"{w}x{h}",
        "segments": segs,
        "ad_candidates": ad_candidates,
        "window_scores": window_scores,
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
        print(f"\nWindow scores: {len(window_scores)} entries (1s each)")
        print(f"Ad candidates: {len(ad_candidates)} intervals")
        for a in ad_candidates:
            am, as_ = divmod(a["start"], 60)
            em, es = divmod(a["end"], 60)
            print(f"  {int(am):02d}:{as_:04.1f} -> {int(em):02d}:{es:04.1f}  ({a['duration']:.0f}s)")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS576 Project - Video Analyzer")
    parser.add_argument("video", help="input video file")
    parser.add_argument("--output", default=None, help="output json path")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--sample-every", type=int, default=None, help="sample every N frames")
    args = parser.parse_args()

    analyze(args.video, args.output, verbose=not args.quiet, sample_n=args.sample_every)
