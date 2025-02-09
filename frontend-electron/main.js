// const { app, BrowserWindow } = require('electron');
//
// let mainWindow;
//
// app.whenReady().then(() => {
//   mainWindow = new BrowserWindow({
//     width: 1280,
//     height: 720,
//     // frame: false,  // Removes border and title bar
//     transparent: true, // Optional: Make the window transparent
//     webPreferences: {
//       nodeIntegration: true
//     }
//   });
//
//   mainWindow.webContents.openDevTools()
//   mainWindow.loadFile('index.html');
//
//   app.on('window-all-closed', () => {
//     if (process.platform !== 'darwin') {
//       app.quit();
//     }
// });
// });

const { app, BrowserWindow, ipcMain, desktopCapturer, globalShortcut } = require('electron');
const { spawn } = require("child_process");
const path = require('path');
const { ExplainInfo } = require('./explaininfo.js');
require('@electron/remote/main').initialize();

function createWindow() {
  // Log the Electron module and check desktopCapturer in the main process
  //
  const electronModule = require('electron');
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true,      // if you need Node integration in your web code
      contextIsolation: false,
      enableRemoteModule: true         // Enable remote module so we can require from the main process
    }
  });
  // Load your HTML file (ensure it's in your project directory)
  require("@electron/remote/main").enable(win.webContents);
  win.webContents.openDevTools()
  win.loadFile('index.html');
  globalShortcut.register("Control+Shift+J", async () => {
    console.log("Ctrl+Shift+J pressed");
    console.log(process.platform)

    if (process.platform == "darwin") {
      let spawn = require("child_process").spawn;
      await spawn("./simulate_keys.sh");
      const explainer = new ExplainInfo();
      // console.log("wtf is going on?")
      explainer.explainThis();
    }

  })



  // Manually close the Python process after 100 seconds (100,000 milliseconds)
  // setTimeout(() => {
  //   // The default signal is 'SIGTERM'
  //   pythonProcess.kill();
  //   console.log("Python process terminated after 100 seconds.");
  // }, 100000);
}

ipcMain.handle('get-screen-sources', async (event, options) => {
  try {
    // options could be something like: { types: ['screen'] }
    const sources = await desktopCapturer.getSources(options);
    return sources;
  } catch (error) {
    return [];
  }
});
app.on("will-quit", () => {
  globalShortcut.unregisterAll();
  console.log("Shortcuts unregistered.");
});

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
  globalShortcut.unregisterAll();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

setInterval(async () => {
  try {
    // Launch Python script
    const pythonProcess = spawn("python", ["../Focus-App/gazemapping.py"]);

    pythonProcess.stdout.on("data", (data) => {
      console.log(`Python Output: ${data}`);
    });

    pythonProcess.stderr.on("data", (data) => {
      console.error(`Python Error: ${data}`);
    });

    pythonProcess.on("close", (code) => {
      console.log(`Python script exited with code ${code}`);
    });
  } catch (error) {
    console.error('Error in automatic message sending:', error);
  }
}, 10000);
