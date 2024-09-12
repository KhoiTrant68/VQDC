#!/bin/bash

# Find all __pycache__ directories and remove them
find . -type d -name "__pycache__" -exec rm -r {} +

# Find all Python files
PYPATH=$(find . -type f -name "*.py")

# Run Black to format the code
python3 -m black $PYPATH

# Run isort to sort imports
python3 -m isort $PYPATH
