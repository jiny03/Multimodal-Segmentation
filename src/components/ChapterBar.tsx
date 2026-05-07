import React from 'react';
import Chapter from './types';

interface ChapterBarProps {
  chapters: Chapter[];
  duration: number;
  onChapterClick: (startTime: number) => void;
}

interface ChapterBarProps {
  chapters: Chapter[];
  duration: number;
  currentTime: number; // New Prop
  onChapterClick: (startTime: number) => void;
}

const ChapterBar: React.FC<ChapterBarProps> = ({ chapters, duration, currentTime, onChapterClick }) => {
  const totalTime = duration || 1;

  return (
    <div style={{ width: '100%', boxSizing: 'border-box', padding: '10px' }}>
      {/* BAR CONTAINER */}
      <div style={{ display: 'flex', width: '100%', height: '12px', gap: '2px' }}>
        {chapters.map((chapter, index) => {
          const nextStart = chapters[index + 1] ? chapters[index + 1].start : totalTime;
          const segmentDuration = nextStart - chapter.start;
          
          // Determine if this segment is active for the bar highlight (optional)
          const isActive = currentTime >= chapter.start && currentTime < nextStart;

          return (
            <div
              key={`bar-${index}`}
              style={{
                flex: `${segmentDuration} 1 0%`, 
                backgroundColor: chapter.color,
                height: '100%',
                borderRadius: '1px',
                opacity: isActive ? 1 : 0.6, // Dim inactive segments slightly
                transition: 'opacity 0.2s'
              }}
            />
          );
        })}
      </div>

      {/* BUTTON CONTAINER */}
      <div style={{ display: 'flex', width: '100%', marginTop: '8px', gap: '2px' }}>
        {chapters.map((chapter, index) => {
          const nextStart = chapters[index + 1] ? chapters[index + 1].start : totalTime;
          const segmentDuration = nextStart - chapter.start;
          
          // LOGIC: Is the playhead currently inside this chapter?
          const isActive = currentTime >= chapter.start && currentTime < nextStart;

          return (
            <button 
              key={`btn-${index}`}
              onClick={() => onChapterClick(chapter.start)}
              style={{ 
                flex: `${segmentDuration} 1 0%`,
                background: isActive ? '#444' : '#2a2a2a', // Subtle background change
                color: isActive ? '#fff' : '#ccc',         // Brighten text
                fontWeight: isActive ? 'bold' : 'normal',   // BOLD ACTIVE TEXT
                border: isActive ? '1px solid #666' : 'none',
                padding: '10px 2px', 
                fontSize: '11px', 
                borderRadius: '4px', 
                cursor: 'pointer',
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                minWidth: 0,
                boxSizing: 'border-box',
                transition: 'all 0.2s ease' // Smooth transition for the bold/color change
              }}
              title={chapter.title}
            >
              {chapter.title}
            </button>
          );
        })}
      </div>
    </div>
  );
};

export default ChapterBar;