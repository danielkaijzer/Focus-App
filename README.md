# Focus Monsters - Focal

## Description
Focal is a productivity app designed to accurately monitor focus during a work session using accurate gaze to screen coordinate mapping and screen capture analysis via VLM. The application has context-awareness on which part of the screen the user is focusing by looking at user's eyegazes through the webcam.

## Toolchain used:
- OpenFace library: For eyetracking functionality (we did our own edits on the library to fulfill the requirements of our app)
- Electron: The UI framework we use

## Project Setup
To set up the project environment:
1. Clone the repository
2. Navigate to the project directory
3. Make sure npm is installed in your system
   
## Running the application:
1. Install the necessary library by running `npm install`
2. Run `npm start` to start the application

## Notes
Inside main repo: folder called `external_libs`, this contains external libraries like `openFace`


`export QT_QPA_PLATFORM_PLUGIN_PATH=/opt/anaconda3/envs/focus_env/lib/python3.10/site-packages/PyQt6/Qt6/plugins/platforms` 

`python gazemapping.py`

## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
