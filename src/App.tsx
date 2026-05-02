import React, { useEffect, useRef, useState } from 'react';
import { VideoPlayer } from '@streamspark/react-video-player';
import ChapterBar from './components/ChapterBar';

import videoSrc from '../assets/video/test_001.mp4';
import chaptersData from '../assets/chapters/test_001.json';
import "../node_modules/@streamspark/react-video-player/dist/index.css";

const App = () => {
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0); // Track playhead position
  const [skipAds, setSkipAds] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const videoElement = containerRef.current?.querySelector('video');
    
    if (videoElement) {
      const handleTimeUpdate = () => {
        const time = videoElement.currentTime;
        setCurrentTime(time); // Update state for the UI

        if (skipAds) {
          chaptersData.forEach((chapter, index) => {
            const isAd = chapter.color === "#f44336";
            const nextChapter = chaptersData[index + 1];
            const chapterEnd = nextChapter ? nextChapter.start : videoElement.duration;

            if (isAd && time >= chapter.start && time < chapterEnd) {
              videoElement.currentTime = chapterEnd;
            }
          });
        }
      };

      const updateDuration = () => {
        if (videoElement.duration) setDuration(videoElement.duration);
      };

      videoElement.addEventListener('loadedmetadata', updateDuration);
      videoElement.addEventListener('timeupdate', handleTimeUpdate);

      return () => {
        videoElement.removeEventListener('loadedmetadata', updateDuration);
        videoElement.removeEventListener('timeupdate', handleTimeUpdate);
      };
    }
  }, [skipAds]);

  return (
    <div ref={containerRef} style={{ width: '100%', maxWidth: '900px', margin: 'auto', color: 'white' }}>
      <VideoPlayer src={videoSrc} title="Tech Podcast Episode 12" />
      
      {/* Skip Ads Checkbox logic here... */}
      <div style={{ padding: '0 10px', display: 'flex', alignItems: 'center', gap: '8px' }}>
        <input 
          type="checkbox" 
          id="skipAds" 
          checked={skipAds} 
          onChange={(e) => setSkipAds(e.target.checked)} 
          style={{ cursor: 'pointer' }}
        />
        <label htmlFor="skipAds" style={{ fontSize: '14px', cursor: 'pointer', userSelect: 'none' }}>
          Skip Ads
        </label>
      </div>
      
      <ChapterBar 
        chapters={chaptersData} 
        duration={duration} 
        currentTime={currentTime} // Pass current time to the bar
        onChapterClick={(time) => {
          const video = containerRef.current?.querySelector('video');
          if (video) video.currentTime = time;
        }} 
      />
    </div>
  );
};

export default App;