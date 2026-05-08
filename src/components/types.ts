interface Chapter {
  title: string;
  type: 'content' | 'ad' | 'intro' | 'outro' | 'intro/outro' | 'transition / intermission' | 'recap';
  start: number;
}

export default Chapter;