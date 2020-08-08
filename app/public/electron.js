const electron = require('electron');
const app = electron.app;
const BrowserWindow = electron.BrowserWindow;
const path = require('path');
const url = require('url');
const isDev = require('electron-is-dev');

let mainWindow;
// console.log(__dirname)
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 800, 
    height: 480,
    fullscreen: false,
    frame: false,
    resizable: true,
    icon: __dirname +  "/visualAI.png",
    webPreferences: {
      nodeIntegration: true
    },
    minimizable: false,
  });
  mainWindow.maximize();
  // mainWindow.removeMenu()
  // mainWindow.setMinimizable(false);
  // mainWindow.setMenu(CustomTitlebar);
  mainWindow.loadURL(isDev ? 'http://localhost:3000' : `file://${path.join(__dirname, '../build/index.html')}`);
  mainWindow.on('closed', () => mainWindow = null);
}


app.on('ready', createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});