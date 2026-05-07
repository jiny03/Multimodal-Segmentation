# Video Analysis
CS576 Final Project - Hemil Bhavsar

My part is the video/visual analysis. Analyzes a video and detects ad segments vs real content using color histogram comparison. Outputs a JSON file the player and audio module can use.

## How to run

install dependencies:
```
pip install -r requirements.txt
```

basic usage:
```
python video_analyzer.py myvideo.mp4
```

saves a `.segments.json` file in the same directory as the video.

to evaluate against the provided ground truth files:
```
python video_analyzer.py test_001.mp4 --eval video_info/test_001.json
```

custom output path:
```
python video_analyzer.py myvideo.mp4 --output result.json
```

## How it works

**Core idea:** Ads are filmed and produced differently from the main video content — different color palette, different lighting, different visual style. We detect this using color histograms.

**Algorithm:**
1. Sample every 90th frame (~3s intervals at 30fps). Resize each frame to 160px wide before processing (fast, still enough color detail).
2. For each frame, compute a normalized HSV color histogram (H, S, V channels, 16 bins each = 48-bin feature vector).
3. Accumulate per-second histogram buckets across the video.
4. **Primary signal (local contrast):** For each second t, compute chi-squared histogram distance from the LOCAL context window: [t-60s to t-8s] union [t+8s to t+60s]. The 8s gap prevents the ad itself from diluting the reference. Sections that look suddenly different from what's immediately around them get a high score.
5. **New signal (boundary change):** For each second t, compare the average histogram of [t-30s to t] vs [t to t+30s]. Sharp peaks = visual style transitions = ad start/end timestamps. This is cleaner for fusion than the smeared region signal.
6. Smooth the local-contrast scores with a 20s box filter. Normalize to 0-1.
7. Use hysteresis thresholding: score must cross 0.62 to enter an ad region, drop below 0.38 to exit.
8. Cap regions longer than 200s by splitting on score dips (prevents runaway hysteresis).
9. Snap detected boundaries to nearby boundary-change peaks (tightens loose edges).
10. Recovery pass: find ads the region detector missed by looking for paired boundary peaks around an elevated-score interval.
11. Adjacent-frame cut detection: reads frame N and N+1 for each sampled frame to get real shot cuts (not the 3s-apart noise from earlier).

## Output format

```json
{
  "source": "test_001.mp4",
  "duration": 1458.4,
  "fps": 29.9,
  "resolution": "640x360",
  "segments": [
    {"start": 0.0,   "end": 81.3,  "type": "video_content", "duration": 81.3},
    {"start": 81.3,  "end": 259.1, "type": "ad",            "duration": 177.8},
    {"start": 259.1, "end": 605.5, "type": "video_content", "duration": 346.4}
  ],
  "ad_candidates": [
    {"start": 81.3, "end": 259.1, "duration": 177.8, "method": "region_snapped"}
  ],
  "window_scores": [
    {"time": 0.0,  "score": 0.12, "boundary_change": 0.08},
    {"time": 1.0,  "score": 0.14, "boundary_change": 0.09},
    {"time": 81.0, "score": 0.81, "boundary_change": 0.94}
  ],
  "shot_cut_times": [81.3, 259.1, 605.5],
  "confidence": {
    "score_spread": 0.612,
    "boundary_peakiness": 0.834,
    "shot_cuts_per_minute": 3.2,
    "interpretation": "high (visual signal usable)"
  },
  "summary": {"video_content": 1178.2, "ad": 280.2}
}
```

- `segments` — full alternating timeline of `video_content` and `ad`. Matches the ground truth JSON format exactly.
- `ad_candidates` — just the detected ad intervals. `method` is either `region_snapped` (found by local-contrast scoring) or `boundary_pair` (found by recovery pass using boundary peaks).
- `window_scores` — per-second entries with two signals:
  - `score` — 0–1 ad likelihood from local-contrast (high = likely ad region)
  - `boundary_change` — 0–1 style transition signal (sharp peaks = ad start/end timestamps)
- `shot_cut_times` — list of timestamps where real scene cuts were detected (adjacent-frame comparison).
- `confidence` — tells the fusion module how trustworthy the visual signal is for this video. If `score_spread` is low, down-weight the visual score in fusion.

## Integration with audio module (Jin)

- `window_scores[i].score` — primary per-second ad likelihood, 1s resolution
- `window_scores[i].boundary_change` — use this to localize exact ad edges. High peaks align with true ad start/end timestamps
- `confidence.interpretation` — if "low", down-weight visual in fusion
- Suggested fusion: `final_score = 0.4 * video_score + 0.4 * audio_score + 0.2 * text_score`
- Or use `ad_candidates` directly as the visual prediction and let audio/text override

## Accuracy on test videos (visual-only)

| Video | Recall | False Positive |
|-------|--------|----------------|
| test_001 | 100% | 5.4% |
| test_002 | 40% | 34.3% |
| test_003 | 52% | 29.2% |
| test_004 | 99% | 36.3% |
| test_005 | 57% | 13.2% |
| **Average** | **70%** | **23.7%** |

test_002 is the hardest — the ads are visually indistinguishable from the main content (same color style). Audio/text modules should compensate.

## Segment types — matches ground truth format
- `video_content` — main video content
- `ad` — detected ad segment
