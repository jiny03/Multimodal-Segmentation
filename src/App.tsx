import React, { useEffect, useRef, useState } from 'react';
import { VideoPlayer } from '@streamspark/react-video-player';
import ChapterBar from './components/ChapterBar';
import { useFileHandlers } from './components/filereader';
import Chapter from './components/types';

// Ensure styles are loaded
import "../node_modules/@streamspark/react-video-player/dist/index.css";

const App = () => {
  const { 
    videoSrc, 
    trueChapters,      
    videoPredicted,   
    textPredicted,    
    audioPredicted,   // Added audio state
    isAnalyzing, 
    handleVideoUpload, 
    handleJsonUpload 
  } = useFileHandlers();

  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  // Added 'audio' as a valid option for the skip engine
  const [skipEngine, setSkipEngine] = useState<'none' | 'true' | 'text' | 'video' | 'audio'>('none');
  const containerRef = useRef<HTMLDivElement>(null);

  // --- CORE LOGIC: Time Updates & Auto-Skip Engine ---
  useEffect(() => {
    const videoElement = containerRef.current?.querySelector('video');
    if (!videoElement) return;

    const handleTimeUpdate = () => {
      const time = videoElement.currentTime;
      setCurrentTime(time);

      // Determine which dataset to use for skipping
      let activeChapters: Chapter[] = [];
      if (skipEngine === 'true') activeChapters = trueChapters;
      else if (skipEngine === 'text') activeChapters = textPredicted;
      else if (skipEngine === 'video') activeChapters = videoPredicted;
      else if (skipEngine === 'audio') activeChapters = audioPredicted;

      // Logic: If the engine is active and the current segment is an 'ad', jump to the end of it
      if (skipEngine !== 'none' && activeChapters.length > 0) {
        activeChapters.forEach((chapter, index) => {
          if (chapter.type === "ad") { 
            const next = activeChapters[index + 1];
            // If there's a next segment, skip to its start; otherwise skip to end of video
            const end = next ? next.start : videoElement.duration;
            
            if (time >= chapter.start && time < end) {
              videoElement.currentTime = end;
            }
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
    // Added audioPredicted to dependency array
  }, [skipEngine, trueChapters, textPredicted, videoPredicted, audioPredicted, videoSrc]);

  const handleSeek = (timestamp: number) => {
    const video = containerRef.current?.querySelector('video');
    if (video) video.currentTime = timestamp;
  };

  return (
    <div ref={containerRef} style={{ 
      width: '100%', 
      maxWidth: '1000px', 
      margin: '0 auto', 
      color: 'white', 
      fontFamily: 'system-ui, -apple-system, sans-serif',
      padding: '40px 20px', 
      minHeight: '100vh',
      boxSizing: 'border-box',
      overflowY: 'visible'
    }}>
      
      <header style={{ marginBottom: '30px' }}>
        <h1 style={{ fontSize: '24px', fontWeight: 600, margin: '0 0 8px 0' }}>CS 576 Project</h1>
        <p style={{ color: '#888', fontSize: '14px' }}>Christopher Straw, Jin Yang, Hemil Bhavsar</p>
      </header>

      {/* --- UPLOAD CONTROLS --- */}
      <div style={{ 
        backgroundColor: '#1a1a1a', 
        padding: '24px', 
        borderRadius: '12px', 
        marginBottom: '24px', 
        display: 'grid', 
        gridTemplateColumns: '1fr 1fr', 
        gap: '24px',
        border: '1px solid #333'
      }}>
        <div>
          <label style={{ display: 'block', fontSize: '11px', fontWeight: 'bold', color: '#4CAF50', marginBottom: '8px', textTransform: 'uppercase' }}>
            1. Source Video (Run Analysis)
          </label>
          <input type="file" accept="video/*" onChange={handleVideoUpload} style={{ fontSize: '13px', color: '#ccc' }} />
        </div>
        <div>
          <label style={{ display: 'block', fontSize: '11px', fontWeight: 'bold', color: '#2196F3', marginBottom: '8px', textTransform: 'uppercase' }}>
            2. Ground Truth (Manual JSON)
          </label>
          <input type="file" accept=".json" onChange={handleJsonUpload} style={{ fontSize: '13px', color: '#ccc' }} />
        </div>
      </div>

      {videoSrc ? (
        <>
          {/* --- PLAYER AREA --- */}
          <div style={{ borderRadius: '12px', overflow: 'hidden', backgroundColor: '#000', border: '1px solid #333' }}>
            <VideoPlayer key={videoSrc} src={videoSrc} title="Analysis Workspace" />
          </div>
          
          {/* --- SKIP ENGINE SELECTOR --- */}
          <div style={{ 
            padding: '15px 20px', 
            backgroundColor: '#1a1a1a', 
            borderRadius: '8px', 
            margin: '20px 0', 
            display: 'flex', 
            alignItems: 'center', 
            gap: '15px',
            border: '1px solid #333'
          }}>
            <label htmlFor="skip-select" style={{ fontSize: '12px', fontWeight: 'bold', color: '#888', textTransform: 'uppercase' }}>
              Auto-Skip Non-Content:
            </label>
            <select 
              id="skip-select"
              value={skipEngine}
              onChange={(e) => setSkipEngine(e.target.value as any)}
              style={{
                backgroundColor: '#333',
                color: 'white',
                border: '1px solid #444',
                padding: '8px 12px',
                borderRadius: '4px',
                fontSize: '13px',
                cursor: 'pointer',
                outline: 'none'
              }}
            >
              <option value="none">None</option>
              <option value="true">Ground Truth</option>
              <option value="text">Text Predicted</option>
              <option value="video">Video Predicted</option>
              <option value="audio">Audio Predicted</option>
            </select>
            <div style={{ fontSize: '11px', color: skipEngine === 'none' ? '#555' : '#4CAF50', display: 'flex', alignItems: 'center', gap: '5px' }}>
              <span style={{ fontSize: '14px' }}>●</span> {skipEngine === 'none' ? 'OFF' : 'ACTIVE'}
            </div>
          </div>

          {/* --- MULTI-TRACK CHAPTER BARS --- */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '40px', marginTop: '20px' }}>
            
            {/* TRACK: TRUE SEGMENTS */}
            <section>
              <TrackHeader title="TRUE Segments" subtitle="Manual markers from ground-truth JSON" color="#4CAF50" />
              {trueChapters.length > 0 ? (
                <ChapterBar chapters={trueChapters} duration={duration} currentTime={currentTime} onChapterClick={handleSeek} />
              ) : (
                <EmptyState message="Please upload a JSON file to see manual segments." />
              )}
            </section>

            {/* AI RESULTS SECTION */}
            <div style={{ borderTop: '1px solid #333', paddingTop: '40px' }}>
                <h3 style={{ fontSize: '13px', color: '#555', marginBottom: '25px', textTransform: 'uppercase', letterSpacing: '1.5px', textAlign: 'center' }}>
                    Multimodal Analysis Results
                </h3>

                {isAnalyzing ? (
                    <div style={{ padding: '60px', textAlign: 'center', backgroundColor: '#111', borderRadius: '8px', border: '1px dashed #333' }}>
                        <span style={{ color: '#ffca28', fontSize: '14px', letterSpacing: '0.5px' }}>
                          Running Python script... Processing frames and audio...
                        </span>
                    </div>
                ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '40px' }}>
                        {/* TRACK: AUDIO PREDICTED */}
                        <section>
                            <TrackHeader title="Multimodal Prediction" subtitle="Segments inferred from text, audio, and video processing" color="#ff9800" />
                            {audioPredicted.length > 0 ? (
                                <ChapterBar chapters={audioPredicted} duration={duration} currentTime={currentTime} onChapterClick={handleSeek} />
                            ) : <EmptyState message="Awaiting analysis..." />}
                        </section>

                    </div>
                )}
            </div>
          </div>
        </>
      ) : (
        <div style={{ 
            textAlign: 'center', 
            padding: '100px 20px', 
            border: '2px dashed #222', 
            borderRadius: '16px', 
            marginTop: '40px',
            color: '#444' 
        }}>
          <h2 style={{ fontSize: '18px', fontWeight: 500, color: '#666', marginBottom: '10px' }}>Dashboard Ready</h2>
          <p style={{ fontSize: '14px' }}>Upload a video file to begin analysis.</p>
        </div>
      )}
    </div>
  );
};

/* --- UI HELPERS --- */

const TrackHeader = ({ title, subtitle, color }: { title: string; subtitle: string; color: string }) => (
    <div style={{ marginBottom: '14px', borderLeft: `4px solid ${color}`, paddingLeft: '12px' }}>
        <h2 style={{ fontSize: '15px', fontWeight: 600, margin: '0 0 2px 0' }}>{title}</h2>
        <p style={{ fontSize: '11px', color: '#666', margin: 0, textTransform: 'uppercase', letterSpacing: '0.5px' }}>{subtitle}</p>
    </div>
);

const EmptyState = ({ message }: { message: string }) => (
    <div style={{ 
        padding: '20px', 
        backgroundColor: '#111', 
        borderRadius: '6px', 
        fontSize: '12px', 
        color: '#444', 
        textAlign: 'center',
        border: '1px solid #1a1a1a' 
    }}>
        {message}
    </div>
);

export default App;