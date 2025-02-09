#!/bin/bash

# Use AppleScript to get selected text from the active application
osascript -e 'tell application "System Events" to keystroke "c" using {command down}'
