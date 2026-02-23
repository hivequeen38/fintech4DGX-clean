#!/usr/bin/env bash

# Move to the script directory (important for VSCode + Docker)
cd "$(dirname "$0")"

# Ensure venv exists
if [ ! -f ".venv/bin/activate" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    # Activate the virtual environment
    source .venv/bin/activate
fi

# Main retry loop
while true; do
    echo "Running MultiSentimentMain.py..."
    
    python MultiSentimentMain.py
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "Python script ran successfully, exiting loop."
        break
    else
        echo "Python script failed with exit code $exit_code, rerunning..."
        sleep 1
    fi
done
