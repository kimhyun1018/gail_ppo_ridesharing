#!/bin/bash
set -e


Xvfb :99 -screen 0 1024x768x24 &

# Wait a moment to ensure Xvnc starts
sleep 2

# Export DISPLAY variable
export DISPLAY=:99

# Execute the command passed to the container
exec "$@"

