# Video Analysis
CS576 Final Project - Hemil Bhavsar

My part is the video analysis piece. It goes through a video and figures out which parts are actual content vs non-content (intro, outro, ad breaks, dead air, static screens etc.) and saves the result as a json file that the player can use.

## How to run

install dependencies first:
```
pip install -r requirements.txt
```

then just run it on a video:
```
python video_analyzer.py myvideo.mp4
```

it saves a .segments.json file in the same folder as the video. you can also pass --output if you want it somewhere else.

for long videos use --sample-every to make it faster (trades a bit of accuracy for speed):
```
python video_analyzer.py myvideo.mp4 --sample-every 25
```

## How it works

basically goes frame by frame and computes:
- motion between frames (absdiff on grayscale)
- brightness and variance of each frame
- uses those to label each frame as content, low_motion, static_screen, or dead_air
- smooths out the labels so it doesnt flip every second
- merges consecutive same-label frames into segments
- uses position in the video to figure out if something is intro/outro
- static or black screens in the middle of the video get flagged as ad_break

## Segment types

- content - main video
- intro - start of video before the real content
- outro - end of video after main content
- ad_break - static/black screen in the middle, likely an ad or transition
- low_motion - still frames, slides, talking head with no movement
- static_screen - title cards, transition screens
- dead_air - black screen

## Window scores (for audio integration)

the output also includes per-second window scores. each entry is a 1 second window with a score 0-1 where high score = likely non-content (ad, transition, etc.) and low score = likely real content. this is computed from visual features only (motion, variance, brightness) and is meant to be combined with the audio scores from Jin's module.

## Output json

```json
{
  "source": "video.mp4",
  "duration": 1292.3,
  "fps": 25.0,
  "resolution": "640x360",
  "segments": [
    {"start": 0.0, "end": 12.0, "type": "intro", "duration": 12.0},
    {"start": 12.0, "end": 900.0, "type": "content", "duration": 888.0},
    {"start": 900.0, "end": 950.0, "type": "ad_break", "duration": 50.0}
  ],
  "window_scores": [
    {"time": 0.0, "score": 0.82},
    {"time": 1.0, "score": 0.75},
    {"time": 2.0, "score": 0.21}
  ],
  "summary": {"content": 900.0, "intro": 12.0, "ad_break": 50.0}
}
```

times are in seconds. frontend reads this to draw the timeline. window_scores can be merged with audio scores for better combined classification.
