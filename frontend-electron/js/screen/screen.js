// ScreenManager.js
// This version uses IPC to request the screen sources from the main process.

const { ipcRenderer } = require('electron');  // Use ipcRenderer to communicate with the main process

export class ScreenManager {
  /**
   * @param {Object} config
   * @param {number} config.width - Target width for resizing captured images.
   * @param {number} config.quality - JPEG quality (0-1).
   * @param {Function} [config.onStop] - Callback when screen sharing stops.
   */
  constructor(config) {
    this.config = {
      width: config.width || 1280,
      quality: config.quality || 0.8,
      onStop: config.onStop
    };

    this.stream = null;
    this.videoElement = null;
    this.canvas = null;
    this.ctx = null;
    this.isInitialized = false;
    this.aspectRatio = null;
    this.previewContainer = null;
  }

  /**
   * Displays the preview container if it exists.
   */
  showPreview() {
    if (this.previewContainer) {
      this.previewContainer.style.display = 'block';
    }
  }

  /**
   * Hides the preview container if it exists.
   */
  hidePreview() {
    if (this.previewContainer) {
      this.previewContainer.style.display = 'none';
    }
  }

  /**
   * Uses IPC to request screen sources from the main process and then obtains a media stream.
   * @returns {Promise<MediaStream>}
   */
  async getScreenStream() {
    // Request screen sources from the main process via IPC.
    const inputSources = await ipcRenderer.invoke('get-screen-sources', { types: ['screen'] });
    if (!inputSources || inputSources.length === 0) {
      throw new Error("No screens available for capture.");
    }
    // For simplicity, select the first available screen.
    const selectedSource = inputSources[0];

    const constraints = {
      audio: false,
      video: {
        mandatory: {
          chromeMediaSource: 'desktop',
          chromeMediaSourceId: selectedSource.id,
          minWidth: 640,
          maxWidth: 1920,
          minHeight: 480,
          maxHeight: 1080,
        }
      }
    };

    return await navigator.mediaDevices.getUserMedia(constraints);
  }

  /**
   * Initializes the screen capture, video element, and canvas.
   * @returns {Promise<void>}
   */
  async initialize() {
    if (this.isInitialized) return;

    try {
      // Get the screen stream using IPC and the main process.
      this.stream = await this.getScreenStream();

      // Create a video element to play the stream.
      this.videoElement = document.createElement('video');
      this.videoElement.srcObject = this.stream;
      this.videoElement.playsInline = true;

      // If a preview container exists in your HTML (with id "screenPreview"), append the video.
      const previewContainer = document.getElementById('screenPreview');
      if (previewContainer) {
        previewContainer.appendChild(this.videoElement);
        this.previewContainer = previewContainer;
        this.showPreview();
      }

      // Start playing the video stream.
      await this.videoElement.play();

      // Wait for metadata to load to obtain valid dimensions.
      await new Promise((resolve) => {
        if (this.videoElement.videoWidth && this.videoElement.videoHeight) {
          resolve();
        } else {
          this.videoElement.onloadedmetadata = resolve;
        }
      });

      // Calculate the aspect ratio from the video dimensions.
      const videoWidth = this.videoElement.videoWidth;
      const videoHeight = this.videoElement.videoHeight;
      this.aspectRatio = videoHeight / videoWidth;

      // Set up a canvas element for capturing images.
      const canvasWidth = this.config.width;
      const canvasHeight = Math.round(canvasWidth * this.aspectRatio);
      this.canvas = document.createElement('canvas');
      this.canvas.width = canvasWidth;
      this.canvas.height = canvasHeight;
      this.ctx = this.canvas.getContext('2d');

      // Listen for when the screen capture is stopped (e.g., when the user stops sharing).
      this.stream.getVideoTracks()[0].addEventListener('ended', () => {
        this.dispose();
        if (this.config.onStop) {
          this.config.onStop();
        }
      });

      this.isInitialized = true;
    } catch (error) {
      throw new Error(`Failed to initialize screen capture: ${error.message}`);
    }
  }

  /**
   * Returns the dimensions of the canvas.
   * @returns {{width: number, height: number}}
   */
  getDimensions() {
    if (!this.isInitialized) {
      throw new Error('Screen capture not initialized. Call initialize() first.');
    }
    return {
      width: this.canvas.width,
      height: this.canvas.height
    };
  }

  /**
   * Captures a screenshot by drawing the current video frame onto the canvas.
   * @returns {Promise<string>} Base64 encoded JPEG image.
   */
  async capture() {
    if (!this.isInitialized) {
      throw new Error('Screen capture not initialized. Call initialize() first.');
    }

    // Draw the current frame from the video onto the canvas.
    this.ctx.drawImage(
      this.videoElement,
      0,
      0,
      this.canvas.width,
      this.canvas.height
    );

    // Convert the canvas image to a base64 encoded JPEG.
    return this.canvas.toDataURL('image/jpeg', this.config.quality).split(',')[1];
  }

  /**
   * Stops the screen capture and cleans up resources.
   */
  dispose() {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }

    if (this.videoElement) {
      this.videoElement.srcObject = null;
      this.videoElement.remove();
      this.videoElement = null;
    }

    if (this.previewContainer) {
      this.hidePreview();
      this.previewContainer.innerHTML = ''; // Clear any preview content.
      this.previewContainer = null;
    }

    this.canvas = null;
    this.ctx = null;
    this.isInitialized = false;
    this.aspectRatio = null;
  }
}
