import cv2
import numpy as np
import json
import argparse
import sys
import os

# tuning params - took a while to settle on these
SAMPLE_N = 90          # sample every 90th frame, ~3s apart at 30fps
HIST_BINS = 16         # 16 bins per HSV channel, 48-dim feature total
SMOOTH_WIN = 20.0      # smooth scores over 20s window
MIN_AD_DUR = 20.0      # ignore anything shorter than 20s, all real ads are 28s+
AD_ENTER = 0.62        # score needs to cross this to count as an ad
AD_EXIT = 0.38         # score needs to drop below this to exit (hysteresis)
MERGE_GAP = 8.0        # merge two candidates if they're within 8s of each other
RESIZE_W = 160         # resize frames to 160px wide before doing anything
MAX_AD_DUR = 200.0     # if a detected region is way too long, split it up
ADJACENT_CUT_THRESH = 8.0  # pixel diff threshold for detecting a real scene cut
BOUNDARY_HALFWIN = 30  # how many seconds on each side for the boundary signal


def get_color_hist(small):
    # HSV histogram - captures color style better than RGB for this kind of comparison
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [HIST_BINS], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [HIST_BINS], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [HIST_BINS], [0, 256]).flatten()
    h_hist /= (h_hist.sum() + 1e-7)
    s_hist /= (s_hist.sum() + 1e-7)
    v_hist /= (v_hist.sum() + 1e-7)
    return np.concatenate([h_hist, s_hist, v_hist])


def chi2_dist(h1, h2):
    # chi-squared distance between two histograms, standard for comparing distributions
    denom = h1 + h2 + 1e-7
    return float(np.sum((h1 - h2) ** 2 / denom))


def smooth_scores(scores, window):
    # simple box filter - makes the score curve less noisy
    n = len(scores)
    result = np.zeros(n)
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        result[i] = np.mean(scores[lo:hi])
    return result


def build_per_window_hists(times, hists, n_windows):
    # average all sampled frames that fall into each 1-second bucket
    win_hists = np.zeros((n_windows, len(hists[0])))
    win_counts = np.zeros(n_windows)
    for i in range(len(times)):
        w = int(times[i])
        if w < n_windows:
            win_hists[w] += hists[i]
            win_counts[w] += 1
    for w in range(n_windows):
        if win_counts[w] > 0:
            win_hists[w] /= (win_hists[w].sum() + 1e-7)
        elif w > 0:
            win_hists[w] = win_hists[w - 1]  # carry forward if no frames landed here
    return win_hists


def build_local_contrast(win_hists, n_windows, gap=8, ctx=60):
    # main scoring idea: compare each second to the ~60s of content around it
    # skip 8s on each side so the ad itself doesn't bleed into its own reference
    # ads look different from the surrounding content, so the chi2 distance spikes
    scores = np.zeros(n_windows)
    for i in range(n_windows):
        left = list(range(max(0, i - ctx), max(0, i - gap)))
        right = list(range(min(n_windows, i + gap), min(n_windows, i + ctx)))
        ctx_idx = left + right
        if not ctx_idx:
            continue
        ctx_hist = np.mean(win_hists[ctx_idx], axis=0)
        ctx_hist /= (ctx_hist.sum() + 1e-7)
        scores[i] = chi2_dist(win_hists[i], ctx_hist)
    return scores


def build_boundary_change(win_hists, n_windows, halfwin=BOUNDARY_HALFWIN):
    # for each second, compare the 30s before vs 30s after
    # at an ad boundary the color style flips, so this peaks sharply right at the cut
    # useful for finding the exact timestamps where transitions happen
    scores = np.zeros(n_windows)
    for i in range(n_windows):
        before = list(range(max(0, i - halfwin), i))
        after = list(range(i, min(n_windows, i + halfwin)))
        if len(before) < halfwin // 2 or len(after) < halfwin // 2:
            continue
        h_b = np.mean(win_hists[before], axis=0)
        h_a = np.mean(win_hists[after], axis=0)
        h_b /= (h_b.sum() + 1e-7)
        h_a /= (h_a.sum() + 1e-7)
        scores[i] = chi2_dist(h_b, h_a)
    return scores


def normalize_pct(arr, lo_pct=5, hi_pct=95):
    # normalize using percentiles instead of min/max to be robust to outliers
    lo = np.percentile(arr, lo_pct)
    hi = np.percentile(arr, hi_pct)
    if hi > lo:
        return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return np.zeros(len(arr))


def find_boundary_peaks(boundary_norm, min_score=0.55, min_separation=10):
    # find the local maxima in the boundary signal above some threshold
    # min_separation stops it from finding the same peak twice
    n = len(boundary_norm)
    peaks = []
    i = 0
    while i < n:
        if boundary_norm[i] >= min_score:
            lo = max(0, i - min_separation // 2)
            hi = min(n, i + min_separation // 2 + 1)
            local_max_i = lo + int(np.argmax(boundary_norm[lo:hi]))
            local_max_v = float(boundary_norm[local_max_i])
            if not peaks or local_max_i - peaks[-1][0] >= min_separation:
                peaks.append((local_max_i, local_max_v))
            i = local_max_i + min_separation
        else:
            i += 1
    return peaks


def snap_to_boundary(region_cands, boundary_norm, snap_window=20, min_peak=0.45):
    # the region detector finds roughly where the ad is, but the edges are often off
    # here we look for a strong boundary peak near each edge and snap to it
    n = len(boundary_norm)
    tightened = []
    for c in region_cands:
        s, e = int(c["start"]), int(c["end"])
        s_lo, s_hi = max(0, s - snap_window), min(n, s + snap_window + 1)
        if s_hi > s_lo:
            s_local = boundary_norm[s_lo:s_hi]
            if s_local.max() >= min_peak:
                new_s = s_lo + int(np.argmax(s_local))
            else:
                new_s = s
        else:
            new_s = s
        e_lo, e_hi = max(0, e - snap_window), min(n, e + snap_window + 1)
        if e_hi > e_lo:
            e_local = boundary_norm[e_lo:e_hi]
            if e_local.max() >= min_peak:
                new_e = e_lo + int(np.argmax(e_local))
            else:
                new_e = e
        else:
            new_e = e
        if new_e - new_s >= MIN_AD_DUR:
            tightened.append({"start": float(new_s), "end": float(new_e),
                              "duration": float(new_e - new_s),
                              "method": "region_snapped"})
    return tightened


def find_missed_ads_from_boundary(boundary_norm, region_scores, region_cands, duration,
                                   min_dur=25.0, max_dur=120.0, min_peak=0.6):
    # sometimes the color scoring misses an ad because it blends into the content
    # but the boundary signal still shows two sharp peaks at the start and end
    # so we look for pairs of strong boundary peaks with elevated scores between them
    peaks = find_boundary_peaks(boundary_norm, min_score=min_peak, min_separation=15)
    n = len(boundary_norm)
    p65 = float(np.percentile(region_scores, 65))
    new_cands = []

    def overlaps_region(s, e, pad=10):
        for r in region_cands:
            if max(s, r["start"] - pad) < min(e, r["end"] + pad):
                return True
        return False

    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            p1_t, p1_v = peaks[i]
            p2_t, p2_v = peaks[j]
            dur = p2_t - p1_t
            if dur < min_dur:
                continue
            if dur > max_dur:
                break
            # the region between the two peaks should also have an elevated score
            mid_score = float(np.mean(region_scores[p1_t:p2_t]))
            if mid_score < p65:
                continue
            # require both peaks to be strong enough
            if p1_v + p2_v < 1.4:
                continue
            if overlaps_region(p1_t, p2_t):
                continue
            new_cands.append({
                "start": float(p1_t), "end": float(p2_t),
                "duration": float(dur),
                "method": "boundary_pair",
            })
    new_cands.sort(key=lambda x: x["start"])
    deduped = []
    for c in new_cands:
        if deduped and c["start"] < deduped[-1]["end"]:
            if c["duration"] > deduped[-1]["duration"]:
                deduped[-1] = c
        else:
            deduped.append(c)
    return deduped


def find_ad_candidates(scores, duration):
    # hysteresis thresholding - need a high score to start an ad, low score to end it
    # this prevents flickering where the score bounces back and forth around the threshold
    # if a region goes way over MAX_AD_DUR it's probably bleeding across content,
    # so split it up by looking at where the score dips within that region
    n = len(scores)
    in_ad = False
    ad_start = 0.0
    candidates = []

    def add_candidate(start, end):
        dur = end - start
        if dur < MIN_AD_DUR:
            return
        if dur > MAX_AD_DUR:
            # region is too long, break it up at the dips
            region = scores[int(start):int(end)]
            above = (region >= AD_EXIT).astype(int)
            run_start = None
            for j in range(len(above)):
                if above[j] and run_start is None:
                    run_start = j
                elif not above[j] and run_start is not None:
                    sub_s = start + float(run_start)
                    sub_e = start + float(j)
                    if sub_e - sub_s >= MIN_AD_DUR:
                        candidates.append({"start": round(sub_s, 2), "end": round(sub_e, 2),
                                           "duration": round(sub_e - sub_s, 2)})
                    run_start = None
            if run_start is not None:
                sub_s = start + float(run_start)
                sub_e = start + float(len(above))
                if sub_e - sub_s >= MIN_AD_DUR:
                    candidates.append({"start": round(sub_s, 2), "end": round(sub_e, 2),
                                       "duration": round(sub_e - sub_s, 2)})
        else:
            candidates.append({"start": round(start, 2), "end": round(end, 2),
                               "duration": round(dur, 2)})

    for i in range(n):
        s = scores[i]
        t = float(i)
        if not in_ad and s >= AD_ENTER:
            if t < 20.0 or t > duration - 20.0:
                continue  # ignore stuff right at the start/end of the video
            ad_start = t
            in_ad = True
        elif in_ad and s < AD_EXIT:
            add_candidate(ad_start, t)
            in_ad = False

    if in_ad and duration - ad_start < duration * 0.3:
        add_candidate(ad_start, duration)

    # merge anything close together into one candidate
    merged = []
    for c in sorted(candidates, key=lambda x: x["start"]):
        if merged and c["start"] - merged[-1]["end"] <= MERGE_GAP:
            merged[-1]["end"] = c["end"]
            merged[-1]["duration"] = round(merged[-1]["end"] - merged[-1]["start"], 2)
        else:
            merged.append(dict(c))

    return merged


def build_segments(ad_candidates, duration):
    # build the full timeline - alternates between video_content and ad
    segs = []
    cursor = 0.0
    for ad in ad_candidates:
        if ad["start"] > cursor + 1.0:
            segs.append({"start": round(cursor, 2), "end": round(ad["start"], 2),
                         "type": "video_content", "duration": round(ad["start"] - cursor, 2)})
        segs.append({"start": round(ad["start"], 2), "end": round(ad["end"], 2),
                     "type": "ad", "duration": round(ad["duration"], 2)})
        cursor = ad["end"]
    if cursor < duration - 1.0:
        segs.append({"start": round(cursor, 2), "end": round(duration, 2),
                     "type": "video_content", "duration": round(duration - cursor, 2)})
    return segs


def analyze(video_path, out_path=None, verbose=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video:", video_path)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps
    resize_h = int(RESIZE_W * h / w) if w > 0 else 90

    if verbose:
        print(f"Video: {os.path.basename(video_path)}  {w}x{h}  {fps:.1f}fps  {duration:.1f}s")

    times = []
    hists = []
    adjacent_cut_times = []

    frame_indices = list(range(0, total_frames, SAMPLE_N))
    for fi, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (RESIZE_W, resize_h), interpolation=cv2.INTER_AREA)
        t = frame_idx / fps
        times.append(t)
        hists.append(get_color_hist(small))

        # also read the very next frame to detect hard cuts (frame N vs frame N+1)
        # this is way more reliable than comparing frames 3 seconds apart
        ret2, frame2 = cap.read()
        if ret2:
            small2 = cv2.resize(frame2, (RESIZE_W, resize_h), interpolation=cv2.INTER_AREA)
            g1 = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY)
            adj = float(cv2.absdiff(g1, g2).astype(np.float32).mean())
            if adj > ADJACENT_CUT_THRESH:
                adjacent_cut_times.append(round(t, 2))

        if verbose and fi > 0 and fi % 200 == 0:
            print(f"  {100*fi/len(frame_indices):.0f}%  t={t:.0f}s")

    cap.release()

    if not times:
        print("No frames analyzed")
        sys.exit(1)

    n_windows = int(np.ceil(duration))
    win_hists = build_per_window_hists(times, hists, n_windows)

    if verbose:
        print(f"Analyzed {len(times)} frames, {len(adjacent_cut_times)} shot cuts detected")
        print("Computing scores...")

    # score 1: local contrast - how different does each second look from what's around it
    local_raw = build_local_contrast(win_hists, n_windows)
    local_smooth = smooth_scores(local_raw, int(SMOOTH_WIN))
    local_norm = normalize_pct(local_smooth)

    # score 2: boundary change - peaks at the exact moments the visual style flips
    boundary_raw = build_boundary_change(win_hists, n_windows)
    boundary_norm = normalize_pct(boundary_raw)

    # find ad regions using the local contrast score
    raw_region_cands = find_ad_candidates(local_norm, duration)

    # tighten up the edges using the boundary signal
    snapped_cands = snap_to_boundary(raw_region_cands, boundary_norm)

    # try to recover any ads the region detector missed
    missed_cands = find_missed_ads_from_boundary(boundary_norm, local_norm,
                                                  snapped_cands, duration)

    # combine and merge everything
    all_c = snapped_cands + missed_cands
    all_c.sort(key=lambda x: x["start"])
    ad_candidates = []
    for c in all_c:
        if ad_candidates and c["start"] - ad_candidates[-1]["end"] <= MERGE_GAP:
            ad_candidates[-1]["end"] = max(ad_candidates[-1]["end"], c["end"])
            ad_candidates[-1]["duration"] = round(
                ad_candidates[-1]["end"] - ad_candidates[-1]["start"], 2)
        else:
            ad_candidates.append({
                "start": round(c["start"], 2), "end": round(c["end"], 2),
                "duration": round(c["duration"], 2),
                "method": c.get("method", "region"),
            })

    segments = build_segments(ad_candidates, duration)

    # confidence block - tells the fusion module how much to trust our visual score
    # if the score curve is flat (low spread), the visual signal isn't very useful
    score_spread = float(np.percentile(local_norm, 95) - np.percentile(local_norm, 5))
    boundary_peakiness = float(np.percentile(boundary_norm, 99) - np.percentile(boundary_norm, 50))
    cuts_per_min = 60.0 * len(adjacent_cut_times) / duration if duration > 0 else 0
    interp = ("high (visual signal usable)" if score_spread > 0.4 else
              "medium" if score_spread > 0.25 else
              "low (fusion should down-weight visual)")

    # per-second output for the fusion module
    window_scores = []
    for i in range(n_windows):
        window_scores.append({
            "time": float(i),
            "score": round(float(local_norm[i]), 4),              # how likely this second is an ad
            "boundary_change": round(float(boundary_norm[i]), 4), # peaks at ad start/end timestamps
        })

    output = {
        "source": os.path.basename(video_path),
        "duration": round(duration, 2),
        "fps": round(fps, 2),
        "resolution": f"{w}x{h}",
        "segments": segments,
        "ad_candidates": ad_candidates,
        "window_scores": window_scores,
        "shot_cut_times": adjacent_cut_times,
        "confidence": {
            "score_spread": round(score_spread, 4),
            "boundary_peakiness": round(boundary_peakiness, 4),
            "shot_cuts_per_minute": round(cuts_per_min, 2),
            "interpretation": interp,
        },
    }

    if out_path is None:
        out_path = os.path.splitext(video_path)[0] + ".segments.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    if verbose:
        print(f"\nSegments ({len(segments)}):")
        for s in segments:
            sm, ss = divmod(s["start"], 60); em, es = divmod(s["end"], 60)
            print(f"  {int(sm):02d}:{ss:05.2f} -> {int(em):02d}:{es:05.2f}  {s['type']}  ({s['duration']:.1f}s)")
        print(f"\nConfidence: spread={score_spread:.3f}  boundary_peakiness={boundary_peakiness:.3f}")
        print(f"  => {interp}")
        print(f"Output: {out_path}")

    return output


def eval_against_gt(pred_output, gt_json_path):
    with open(gt_json_path) as f:
        gt = json.load(f)
    gt_ads = [s for s in gt["timeline_segments"] if s["type"] == "ad"]
    pred_ads = pred_output.get("ad_candidates", [])
    total_dur = gt["output_duration_seconds"]

    print(f"\n=== Eval vs {os.path.basename(gt_json_path)} ===")
    print(f"GT ads: {len(gt_ads)}, Predicted: {len(pred_ads)}")

    total_overlap = 0.0
    total_gt_dur = 0.0
    for ga in gt_ads:
        gs, ge = ga["final_video_start_seconds"], ga["final_video_end_seconds"]
        total_gt_dur += (ge - gs)
        best = max((max(0.0, min(ge, pa["end"]) - max(gs, pa["start"])) for pa in pred_ads), default=0.0)
        total_overlap += best

    overall = total_overlap / total_gt_dur if total_gt_dur > 0 else 0
    total_pred = sum(a["duration"] for a in pred_ads)
    fp = total_pred - total_overlap

    scores = np.array([w["score"] for w in pred_output["window_scores"]])
    boundary = np.array([w["boundary_change"] for w in pred_output["window_scores"]])
    n = len(scores)
    mask = np.zeros(n, dtype=bool)
    for ga in gt_ads:
        a, b = int(ga["final_video_start_seconds"]), int(ga["final_video_end_seconds"])
        mask[a:min(b, n)] = True
    score_gap = scores[mask].mean() - scores[~mask].mean() if mask.any() and (~mask).any() else 0.0

    # check how strong the boundary signal is at the true ad edges vs random spots
    edge_peaks = []
    for ga in gt_ads:
        for tt in [int(ga["final_video_start_seconds"]), int(ga["final_video_end_seconds"])]:
            edge_peaks.append(boundary[max(0, tt - 5):min(n, tt + 5)].max())
    edges_set = set()
    for ga in gt_ads:
        for tt in [int(ga["final_video_start_seconds"]), int(ga["final_video_end_seconds"])]:
            edges_set.update(range(max(0, tt - 15), min(n, tt + 15)))
    nonedge = [i for i in range(n) if i not in edges_set]
    np.random.seed(42)
    if nonedge:
        rs = np.random.choice(nonedge, size=min(20, len(nonedge)), replace=False)
        rand_peaks = [boundary[max(0, p - 5):min(n, p + 5)].max() for p in rs]
    else:
        rand_peaks = [0]

    print(f"  Recall:                {total_overlap:.1f}s / {total_gt_dur:.1f}s = {overall:.0%}")
    print(f"  False positive:        {fp:.1f}s ({100*fp/total_dur:.1f}% of video)")
    print(f"  Primary score gap:     {score_gap:+.3f}  (ad-mean - non-ad-mean)")
    print(f"  Boundary at GT edges:  {np.mean(edge_peaks):.3f}  vs random points: {np.mean(rand_peaks):.3f}")
    return {"recall": overall, "fp_pct": 100*fp/total_dur, "score_gap": score_gap,
            "edge_boundary": float(np.mean(edge_peaks)), "rand_boundary": float(np.mean(rand_peaks))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS576 Project - Video Analyzer")
    parser.add_argument("video", help="input video file")
    parser.add_argument("--output", default=None, help="output json path")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--sample-every", type=int, default=None, help="sample every N frames")
    args = parser.parse_args()

    analyze(args.video, args.output, verbose=not args.quiet, sample_n=args.sample_every)
