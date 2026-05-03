# Video Analysis
CS576 Final Project - Hemil Bhavsar

My part is the video analysis piece. It goes through a video and figures out which parts are actual content vs non-content (intro, outro, dead air, static screens etc.) and saves the result as a json file that the player can use.

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
- uses position in the video to figure out if something is an intro or outro

## Segment types

- content - main video
- intro - start of video before the real content
- outro - end of video after main content
- low_motion - still frames, slides, talking head with no movement
- static_screen - title cards, transition screens
- dead_air - black screen

## Output json

```json
{
  "source": "video.mp4",
  "duration": 1292.3,
  "fps": 25.0,
  "resolution": "640x360",
  "segments": [
    {"start": 0.0, "end": 12.0, "type": "intro", "duration": 12.0},
    {"start": 12.0, "end": 900.0, "type": "content", "duration": 888.0}
  ],
  "summary": {"content": 900.0, "intro": 12.0}
}
```

times are in seconds. frontend reads this to draw the timeline.
