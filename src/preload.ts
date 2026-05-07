// See the Electron documentation for details on how to use preload scripts:
// https://www.electronjs.org/docs/latest/tutorial/process-model#preload-scripts
const { contextBridge, ipcRenderer, webUtils} = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  runAnalysis: (path: any) => ipcRenderer.invoke('run-analysis', path),
  // This helper ensures the path is extracted correctly from the File object
  getFilePath: (file: any) => webUtils.getPathForFile(file)
});