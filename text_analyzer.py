import cv2
import easyocr
import json
import numpy as np

# Initialize EasyOCR for English
# GPU is recommended for multimodal reasoning efficiency 
reader = easyocr.Reader(['en'], gpu=True) 

# Taxonomy based on project requirements 
TAXONOMY_MAP = {
    'sponsorship/advertisement': ['sponsored','iphone', 'phone', 'check out', 'discount','apple.com','pepsi','zero sugar','sports','sport','barbecue','salt','vinegar','chips','Onion','Cheddar'], 
    'intro/outro': ['ted.com','ted talk','tedtalk','ideas','episode','welcome back', 'starting soon','thanks for watching', 'subscribe', 'copyright'], 
    'transition / intermission': ['break', 'intermission', 'stay tuned'], 
    'recap': ['previously', 'last time', 'recap'] 
}

def preprocess_for_ocr(img):
    """
    Enhances low-contrast text (like white text on tan backgrounds) 
    to improve detection accuracy.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_img = clahe.apply(gray)
    
    # Thresholding to isolate text: creates a binary black/white image
    _, thresh = cv2.threshold(contrast_img, 200, 255, cv2.THRESH_BINARY_INV)
    
    return thresh

def classify_text(detected_text):
    combined_text = " ".join(detected_text).lower()
    for label, keywords in TAXONOMY_MAP.items():
        if any(word in combined_text for word in keywords):
            return label
    return "core content" 

def process_video(video_path, interval_sec=2):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval_frames = int(fps * interval_sec)
    
    metadata = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval_frames == 0:
            timestamp = frame_count / fps
            
            # Focused ROI: Expand to middle-bottom to catch center-screen ads 
            h, w, _ = frame.shape
            roi = frame[int(h*0.4):int(h*0.9), 0:w] 
            
            # --- UPDATED: Preprocessing Step ---
            processed_roi = preprocess_for_ocr(roi)
            
            # --- UPDATED: Advanced OCR Parameters ---
            # Using paragraph=True helps group 'your' and 'phone' 
            results = reader.readtext(
                processed_roi, 
                detail=0, 
                paragraph=True, 
                contrast_ths=0.1,
                adjust_contrast=0.7
            )
            
            label = classify_text(results)
            
            metadata.append({
                "timestamp": round(timestamp, 2),
                "detected_text": results,
                "label": label
            })
            if results:
                print(f"Time: {int(timestamp//60)}:{int(timestamp%60):02d} | Label: {label} | OCR: {results}")

        frame_count += 1

    cap.release()
    
    # Save metadata to generate the required "content map" 
    with open('text_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    return metadata

def group_metadata_segments(metadata, gap_threshold=60):
    """
    Groups consecutive detections of the same label into intervals, 
    ignoring 'core content'.
    """
    # 1. Filter out core content
    filtered_data = [m for m in metadata if m['label'] != 'core content']
    
    if not filtered_data:
        print("No non-core segments detected.")
        return []

    # 2. Sort by timestamp
    filtered_data.sort(key=lambda x: x['timestamp'])

    grouped_segments = []
    
    if filtered_data:
        # Initialize the first segment
        current_segment = {
            "label": filtered_data[0]['label'],
            "start": filtered_data[0]['timestamp'],
            "end": filtered_data[0]['timestamp']
        }

        for i in range(1, len(filtered_data)):
            entry = filtered_data[i]
            label = entry['label']
            timestamp = entry['timestamp']

            # Check if it's the same category AND within the 10s gap
            if label == current_segment['label'] and (timestamp - current_segment['end']) <= gap_threshold:
                current_segment['end'] = timestamp
            else:
                # Close the current segment and start a new one
                grouped_segments.append(current_segment)
                current_segment = {
                    "label": label,
                    "start": timestamp,
                    "end": timestamp
                }
        
        # Append the final segment
        grouped_segments.append(current_segment)

    print(f"{'LABEL':<30} | {'START':<10} | {'END':<10}")
    print("-" * 55)
    for seg in grouped_segments:
        print(f"{seg['label']:<30} | {format_time(seg['start']):<10} | {format_time(seg['end']):<10}")

    return grouped_segments

def format_time(seconds):
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"

if __name__ == "__main__":
    # Ensure this runs offline as a practical tool 

    path = "../../assets/video/test_003.mp4"

    metadata = process_video(path)

    
    final_segments = group_metadata_segments(metadata)