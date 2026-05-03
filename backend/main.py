from video_analyzer import analyze 
import json
import argparse
import sys
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS576 Project - Video Analyzer")
    parser.add_argument("video", help="input video file")
    args = parser.parse_args()
    print(analyze(args.video,verbose=False))