import React, { useState } from 'react';
import ModelGenerator from './components/ModelGenerator';
import EnhancedCADViewer from './components/EnhancedCADViewer';
import ModelLibrary from './components/ModelLibrary';
import SystemStatus from './components/SystemStatus';
import './App.css';

function App() {
  const [selectedModel, setSelectedModel] = useState(null);
  const [isNewModel, setIsNewModel] = useState(false);

  const handleModelGenerated = (model) => {
    setSelectedModel(model);
    setIsNewModel(true);
    
    // Remove "new" flag after 5 seconds
    setTimeout(() => {
      setIsNewModel(false);
    }, 5000);
  };

  const handleModelSelect = (model) => {
    setSelectedModel(model);
    setIsNewModel(false);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-sm">🛠️</span>
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">AI CAD Generator</h1>
                <p className="text-xs text-gray-500">Professional 3D Model Generation</p>
              </div>
            </div>
            
            <div className="text-sm text-gray-500">
              React + FastAPI Architecture
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* System Status */}
        <SystemStatus />

        {/* Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Generator and Library */}
          <div className="lg:col-span-1 space-y-6">
            <ModelGenerator onModelGenerated={handleModelGenerated} />
            <ModelLibrary 
              onModelSelect={handleModelSelect} 
              selectedModel={selectedModel}
            />
          </div>

          {/* Right Column - Viewer */}
          <div className="lg:col-span-2">
            <EnhancedCADViewer 
              model={selectedModel} 
              isNewModel={isNewModel}
            />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="text-center text-gray-500 text-sm">
            <p className="mb-2">
              🎓 AI-Powered CAD Generator • College Project • Computer Science Engineering
            </p>
            <p className="text-xs">
              Powered by CrewAI • CADQuery • Plotly • Google Gemini • React • FastAPI
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
