from pathlib import Path
import json
from faster_whisper import WhisperModel
import requests


def transcribe_audio(wav_path, model_size="large-v3-turbo", device="cuda", compute_type="float16"):
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    # transcribe using whisper
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments = model.transcribe(
        str(wav_path),
        vad_filter=True,
        word_timestamps=False,
        condition_on_previous_text=False,
        beam_size=5
    )[0]
    transcript_segments = []

    for segment in segments:
        # save transcript segment with timestamp and confidence metadata
        transcript_segments.append({
            "start": float(segment.start),
            "end": float(segment.end),
            "text": segment.text.strip(),
            "avg_logprob": float(segment.avg_logprob),
            "no_speech_prob": float(segment.no_speech_prob),
        })

    return transcript_segments


def get_interval_transcript(transcript_segments, start, end):
    texts = []

    for segment in transcript_segments:
        segment_start = segment["start"]
        segment_end = segment["end"]

        # Keep transcript segment if it overlaps the interval
        if segment_end >= start and segment_start <= end:
            texts.append(segment["text"])

    return " ".join(texts)


def summarize_video_theme(transcript_segments):
    full_text = ""

    # combine transcript text into one context string
    for segment in transcript_segments:
        full_text += segment["text"] + " "

    prompt = f"""
    Summarize the main topic and core purpose of this video in 2 sentences.
    Focus on what should count as core content.
    
    Transcript:
    {full_text[:6000]}
    """

    request_body = {
        "model": "gemma",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0
    }

    response = requests.post("http://127.0.0.1:8080/v1/chat/completions", json=request_body)

    data = response.json()
    theme = data["choices"][0]["message"]["content"]

    return theme.strip()


def detect_non_content_from_full_transcript(transcript_segments, video_theme):
    timestamped_text = ""

    # build timestamped transcript for the LLM
    for segment in transcript_segments:
        timestamped_text += f"[{segment['start']:.1f}-{segment['end']:.1f}] {segment['text']}\n"

    prompt = f"""
            You are detecting non-content intervals in a long-form video transcript.
            
            Overall video theme:
            {video_theme}
            
            Task:
            Find intervals that are likely inserted advertisements or promotional interruptions.
            
            Important rules:
            Return only intervals that interrupt or do not support the main video theme because they are commercial or promotional.
            If a brand or product is only used as a brief example in the main explanation, do not mark it as an ad.
            If there is a commercial-style product promotion, sponsor read, sales pitch, discount code, or call to action, mark it.
            If there is an unrelated inserted clip that behaves like a commercial break, mark it.
            Do not mark normal intros, outros, scene transitions, dead air, jokes, filler remarks, or production chatter unless they contain explicit ad/promotional language.
            Do not return clips shorter than 15 seconds unless they contain explicit ad/promotional language.
            
            Allowed labels:
            ad, content
            
            Return only valid JSON:
            {{
              "segments": [
                {{
                  "start": 0.0,
                  "end": 0.0,
                  "label": "ad",
                  "non_content_score": 0.0,
                  "reason": "short reason"
                }}
              ]
            }}
            
            Transcript with timestamps:
            {timestamped_text}
            """

    request_body = {
        "model": "gemma",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0
    }

    response = requests.post("http://127.0.0.1:8080/v1/chat/completions", json=request_body)

    data = response.json()

    if "choices" not in data:
        return []

    content = data["choices"][0]["message"]["content"]

    # extract JSON object from LLM response
    start_index = content.find("{")
    end_index = content.rfind("}") + 1

    if start_index == -1 or end_index == 0:
        return []

    json_text = content[start_index:end_index]
    result = json.loads(json_text)

    # return detected ad segments
    return result.get("segments", [])
