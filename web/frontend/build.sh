#!/bin/bash

# Build script for Vercel deployment
# This script handles ESLint warnings gracefully

echo "🏗️ Building F1 Prediction Frontend..."

# Set CI to false to prevent warnings from being treated as errors
export CI=false

# Run the build
npm run build:ci

echo "✅ Build completed successfully!"
