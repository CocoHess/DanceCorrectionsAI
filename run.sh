#!/bin/bash

# Check for Python version
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "Error: Python is not installed"
    exit 1
fi

# Check for virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install requirements if needed
if [ ! -f "venv/.requirements_installed" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
    touch venv/.requirements_installed
fi

# Check for .env file
if [ ! -f ".env" ]; then
    echo "Error: .env file not found."
    echo "Please create a .env file with your OpenAI API key:"
    echo "OPENAI_API_KEY=your_openai_api_key_here"
    exit 1
fi

# Run the application
echo "Starting the DanceCorrect AI application..."
python app.py 