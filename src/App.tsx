import React, { useEffect, useRef, useState } from 'react';
import { VideoPlayer } from '@streamspark/react-video-player';
import ChapterBar from './components/ChapterBar';
import { useFileHandlers } from './components/filereader';

import "../node_modules/@streamspark/react-video-player/dist/index.css";

const App = () => {
  const { 
    videoSrc, 
    chaptersData, 
    analysisData, 
    isAnalyzing, 
    handleVideoUpload, 
    handleJsonUpload 
  } = useFileHandlers();

  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [skipAds, setSkipAds] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const videoElement = containerRef.current?.querySelector('video');
    if (!videoElement) return;

    const handleTimeUpdate = () => {
      const time = videoElement.currentTime;
      setCurrentTime(time);

      if (skipAds && chaptersData.length > 0) {
        chaptersData.forEach((chapter, index) => {
          if (chapter.color === "#f44336") { // Ad logic
            const next = chaptersData[index + 1];
            const end = next ? next.start : videoElement.duration;
            if (time >= chapter.start && time < end) videoElement.currentTime = end;
          }
        });
      }
    };

    const updateDuration = () => setDuration(videoElement.duration || 0);

    videoElement.addEventListener('loadedmetadata', updateDuration);
    videoElement.addEventListener('timeupdate', handleTimeUpdate);
    return () => {
      videoElement.removeEventListener('loadedmetadata', updateDuration);
      videoElement.removeEventListener('timeupdate', handleTimeUpdate);
    };
  }, [skipAds, chaptersData, videoSrc]);

  return (
    <div ref={containerRef} style={{ width: '100%', maxWidth: '900px', margin: '20px auto', color: 'white', fontFamily: 'sans-serif' }}>
      
      {/* Upload Controls */}
      <div style={{ backgroundColor: '#222', padding: '20px', borderRadius: '8px', marginBottom: '20px', display: 'flex', gap: '20px' }}>
        <div>
          <label style={{ display: 'block', fontSize: '12px', color: '#aaa' }}>Video File</label>
          <input type="file" accept="video/*" onChange={handleVideoUpload} />
        </div>
        <div>
          <label style={{ display: 'block', fontSize: '12px', color: '#aaa' }}>Chapter Logic (JSON)</label>
          <input type="file" accept=".json" onChange={handleJsonUpload} />
        </div>
      </div>

      {videoSrc ? (
        <>
          <VideoPlayer key={videoSrc} src={videoSrc} title="Video Preview" />
          
          <div style={{ padding: '10px 0', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <input type="checkbox" id="skip" checked={skipAds} onChange={e => setSkipAds(e.target.checked)} />
            <label htmlFor="skip">Skip Ads (Red Segments)</label>
          </div>
          
          <ChapterBar 
            chapters={chaptersData} 
            duration={duration} 
            currentTime={currentTime}
            onChapterClick={(t) => {
              const v = containerRef.current?.querySelector('video');
              if (v) v.currentTime = t;
            }} 
          />

          {/* Python Analysis Output */}
          <div style={{ marginTop: '30px', borderTop: '1px solid #444', paddingTop: '20px' }}>
            <h3 style={{ marginBottom: '10px' }}>Python Analysis Output</h3>
            {isAnalyzing ? (
              <div style={{ color: '#ffca28' }}>Running analyzer.py...</div>
            ) : (
              <pre style={{ 
                backgroundColor: '#000', 
                padding: '15px', 
                borderRadius: '4px', 
                fontSize: '13px',
                maxHeight: '300px',
                overflowY: 'auto',
                border: '1px solid #333'
              }}>
                {analysisData ? JSON.stringify(analysisData, null, 2) : "Upload a video to see analysis."}
              </pre>
            )}
          </div>
        </>
      ) : (
        <div style={{ textAlign: 'center', padding: '50px', border: '2px dashed #444', color: '#666' }}>
          Select a video to begin analysis and playback.
        </div>
      )}
    </div>
  );
};

export default App;