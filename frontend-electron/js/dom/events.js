import elements from './elements.js';
import settingsManager from '../settings/settings-manager.js';
const fs = require('fs');
const { spawn } = require("child_process");

let geminiTaskAwarenessPromptTemplate = `Format users prompt so that it is in this form: User is working on `
let geminiTaskAwarenessPrompt = ``
const geminiDistractedPrompt = `User is distracted by something on left side of screen. Tell user in this form what they are being distracted by and remind them of their task: "Hey you are getting distracted by X on the left side of the screen. I saw you looking. Don't forget you wanted to work on Y". Please replace X with specific details of whatever is on the left side of the screen and replace Y with the task user provided earlier. Remember, the form that I gave you is just an example. Do a different variation each time you tried reminding the user`

// Persistent variable that holds the latest timestamp from distracted.json
let latestTimestamp = null;
let numberOfDistracted = 0;

/**
 * Updates UI to show disconnect button and hide connect button
 */
const showDisconnectButton = () => {
  elements.connectBtn.style.display = 'none';
  elements.disconnectBtn.style.display = 'block';
};

/**
 * Updates UI to show connect button and hide disconnect button
 */
const showConnectButton = () => {
  elements.disconnectBtn.style.display = 'none';
  elements.connectBtn.style.display = 'block';
};

let isCameraActive = false;

/**
 * Ensures the agent is connected and initialized
 * @param {GeminiAgent} agent - The main application agent instance
 * @returns {Promise<void>}
 */
const ensureAgentReady = async (agent) => {
  if (!agent.connected) {
    await agent.connect();
    showDisconnectButton();
  }
  if (!agent.initialized) {
    await agent.initialize();
  }
};

/**
 * Sets up event listeners for the application's UI elements
 * @param {GeminiAgent} agent - The main application agent instance
 */
export function setupEventListeners(agent) {
  // Disconnect handler
  elements.disconnectBtn.addEventListener('click', async () => {
    try {
      await agent.disconnect();
      showConnectButton();
      [elements.cameraBtn, elements.screenBtn, elements.micBtn].forEach(btn => btn.classList.remove('active'));
      isCameraActive = false;
    } catch (error) {
      console.error('Error disconnecting:', error);
    }
  });

  // Connect handler
  elements.connectBtn.addEventListener('click', async () => {
    try {
      await ensureAgentReady(agent);
    } catch (error) {
      console.error('Error connecting:', error);
    }
  });

  // Microphone toggle handler
  elements.micBtn.addEventListener('click', async () => {
    try {
      await ensureAgentReady(agent);
      await agent.toggleMic();
      elements.micBtn.classList.toggle('active');
    } catch (error) {
      console.error('Error toggling microphone:', error);
      elements.micBtn.classList.remove('active');
    }
  });

  // Camera toggle handler
  elements.cameraBtn.addEventListener('click', async () => {
    try {
      await ensureAgentReady(agent);

      if (!isCameraActive) {
        await agent.startCameraCapture();
        elements.cameraBtn.classList.add('active');
      } else {
        await agent.stopCameraCapture();
        elements.cameraBtn.classList.remove('active');
      }
      isCameraActive = !isCameraActive;
    } catch (error) {
      console.error('Error toggling camera:', error);
      elements.cameraBtn.classList.remove('active');
      isCameraActive = false;
    }
  });

  // Screen sharing handler
  let isScreenShareActive = false;

  // Listen for screen share stopped events (from native browser controls)
  agent.on('screenshare_stopped', () => {
    elements.screenBtn.classList.remove('active');
    isScreenShareActive = false;
    console.info('Screen share stopped');
  });

  elements.screenBtn.addEventListener('click', async () => {
    try {
      await ensureAgentReady(agent);

      if (!isScreenShareActive) {
        await agent.startScreenShare();
        elements.screenBtn.classList.add('active');
      } else {
        await agent.stopScreenShare();
        elements.screenBtn.classList.remove('active');
      }
      isScreenShareActive = !isScreenShareActive;
    } catch (error) {
      console.error('Error toggling screen share:', error);
      elements.screenBtn.classList.remove('active');
      isScreenShareActive = false;
    }
  });

  // Message sending handlers
  const sendMessage = async () => {
    try {
      await ensureAgentReady(agent);
      const text = elements.messageInput.value.trim();
      geminiTaskAwarenessPrompt = geminiTaskAwarenessPromptTemplate + text


      const distractionFreePromptStart = geminiTaskAwarenessPrompt + ` Your task is helping the user stay focused in their task. Please just give a short confirmation stating that you will help the user (use the pronoun 'You' everytime you talk to the user)`
      await agent.sendText(distractionFreePromptStart);
      await agent.startScreenShare();
      elements.screenBtn.classList.add('active');
      elements.messageInput.value = '';

      try {
        // Launch Python script
        const pythonProcess = spawn("python", ["../../../testing/Focus-App/gazemapping.py"]);

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

      // Distracted button click
      setInterval(async () => {
        try {
          // Synchronously read the file 'distracted.json' in UTF-8 encoding
          const data = fs.readFileSync('distracted.json', 'utf8');

          // Parse the file content as JSON
          const jsonData = JSON.parse(data);

          // Log the parsed JSON data
          console.log("Parsed JSON data:", jsonData);

          // Compare the JSON timestamp with the persistent latestTimestamp
          const fileTimestamp = jsonData.timestamp;

          // Check if latestTimestamp is null (first run) or if the new timestamp is newer
          if (latestTimestamp === null || fileTimestamp > latestTimestamp) {
            numberOfDistracted++;
            latestTimestamp = fileTimestamp;
            const geminiDistractedPromptAggresive = geminiDistractedPrompt + `This is the ${numberOfDistracted} time the user is getting distracted. Be more aggresive if the number is bigger. AND YOU DON'T NEED TO SAY YOU UNDERSTAND, I ALREADY KNOW. JUST NOTIFY THE USER IMMEDIATELY`
            console.log("Updated latestTimestamp to:", latestTimestamp);
            await ensureAgentReady(agent);
            await agent.sendTextForContext(geminiTaskAwarenessPrompt);
            await agent.sendText(geminiDistractedPromptAggresive);
          } else {
            console.log("latestTimestamp remains unchanged:", latestTimestamp);
          }
        } catch (error) {
          console.error('Error reading or parsing distracted.json:', error);
        }
      }, 5000);

    } catch (error) {
      console.error('Error sending message:', error);
    }
  };

  elements.sendBtn.addEventListener('click', sendMessage);
  elements.messageInput.addEventListener('keypress', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      sendMessage();
    }
  });

  // Settings button click
  elements.settingsBtn.addEventListener('click', () => settingsManager.show());



}


// Initialize settings
settingsManager;
