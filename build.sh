#!/usr/bin/env bash
# exit on error
set -o errexit

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Creating and activating a 1GB swap file in /tmp..."
# MODIFIED PATH: Using /tmp which is a writable directory in Render's build environment
fallocate -l 1G /tmp/swapfile
chmod 600 /tmp/swapfile
mkswap /tmp/swapfile
swapon /tmp/swapfile
echo "Swap file is active."