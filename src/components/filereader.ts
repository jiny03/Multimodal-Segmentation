import { useState, ChangeEvent } from 'react';
import Chapter from './types';

export const useFileHandlers = () => {
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [chaptersData, setChaptersData] = useState<Chapter[]>([]); // For manual JSON
  const [analysisData, setAnalysisData] = useState<any>(null);    // For Python results
  const [isAnalyzing, setIsAnalyzing] = useState(false);

    const handleVideoUpload = async (e: ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        setIsAnalyzing(true);
        setAnalysisData(null);

        const url = URL.createObjectURL(file);
        setVideoSrc(url);

        const absolutePath = (window as any).electronAPI.getFilePath(file);

        if (!absolutePath || absolutePath === "undefined") {
            console.error("Could not resolve file path. Check Electron contextIsolation settings.");
            setAnalysisData({ error: "File path resolution failed." });
            setIsAnalyzing(false);
            return;
        }

        try {
            const result = await (window as any).electronAPI.runAnalysis(absolutePath);
            setAnalysisData(result);
        } catch (err) {
            setAnalysisData({ error: "Python execution failed", details: err });
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
          setChaptersData(json);
        } catch (err) {
          alert("Invalid Chapter JSON file.");
        }
      };
      reader.readAsText(file);
    }
  };

  return {
    videoSrc,
    chaptersData,
    analysisData,
    isAnalyzing,
    handleVideoUpload,
    handleJsonUpload
  };
};