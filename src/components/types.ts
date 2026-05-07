interface Chapter {
  title: string;
  type: 'content' | 'ad' | 'intro' | 'outro'; // Replaces color: string
  start: number;
}

export default Chapter;