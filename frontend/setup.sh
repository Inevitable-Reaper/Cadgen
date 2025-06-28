#!/bin/bash

echo "🔧 Setting up AI CAD Generator Frontend..."

# Check if we're in the frontend directory
if [ ! -f "package.json" ]; then
    echo "❌ Please run this script from the frontend directory"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Initialize Tailwind CSS if needed
if [ ! -f "src/index.css" ]; then
    echo "🎨 Setting up Tailwind CSS..."
    npx tailwindcss init -p
fi

echo "✅ Frontend setup complete!"
echo "🚀 You can now run: npm start"
