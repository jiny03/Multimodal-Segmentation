import { useState, ChangeEvent } from 'react';
import Chapter from './types';

export const useFileHandlers = () => {
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [trueChapters, setTrueChapters] = useState<Chapter[]>([]); 
  const [videoPredicted, setVideoPredicted] = useState<Chapter[]>([]);
  const [textPredicted, setTextPredicted] = useState<Chapter[]>([]);
  const [audioPredicted, setAudioPredicted] = useState<Chapter[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleVideoUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsAnalyzing(true);
    const url = URL.createObjectURL(file);
    setVideoSrc(url);

    const absolutePath = (window as any).electronAPI.getFilePath(file);

    try {
      const result = await (window as any).electronAPI.runAnalysis(absolutePath);
      // Assuming result is the JSON object: { video: [...], text: [...] }
      console.log(result.audio);
      if (result.video) setVideoPredicted(result.video);
      if (result.text) setTextPredicted(result.text);
      if (result.audio) setAudioPredicted(result.audio);
    } catch (err) {
      console.error("Analysis failed", err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleJsonUpload = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        try {
          const json = JSON.parse(event.target?.result as string);
          setTrueChapters(json);
        } catch (err) {
          alert("Invalid JSON");
        }
      };
      reader.readAsText(file);
    }
  };

  return {
    videoSrc,
    trueChapters,
    videoPredicted,
    textPredicted,
    audioPredicted,
    isAnalyzing,
    handleVideoUpload,
    handleJsonUpload
  };
};