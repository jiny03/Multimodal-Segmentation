"use strict";
const { contextBridge, ipcRenderer, webUtils } = require("electron");
contextBridge.exposeInMainWorld("electronAPI", {
  runAnalysis: (path) => ipcRenderer.invoke("run-analysis", path),
  // This helper ensures the path is extracted correctly from the File object
  getFilePath: (file) => webUtils.getPathForFile(file)
});
