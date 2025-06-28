import React, { useRef, useEffect, useState } from 'react';
import { ExternalLink, Download, RotateCcw, Maximize } from 'lucide-react';
import { cadApi } from '../services/api';

const EnhancedCADViewer = ({ model, isNewModel = false }) => {
  const iframeRef = useRef(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (model?.id) {
      setIsLoading(true);
      setError(null);
      
      // Use the preview endpoint
      const previewUrl = cadApi.getModelPreviewUrl(model.id);
      
      if (iframeRef.current) {
        iframeRef.current.src = previewUrl;
      }
    }
  }, [model]);

  const handleIframeLoad = () => {
    setIsLoading(false);
  };

  const handleIframeError = () => {
    setIsLoading(false);
    setError('Failed to load 3D model');
  };

  const handleFullscreen = () => {
    if (model?.id) {
      const previewUrl = cadApi.getModelPreviewUrl(model.id);
      window.open(previewUrl, '_blank');
    }
  };

  const handleMaximize = () => {
    setIsFullscreen(!isFullscreen);
  };

  const handleRefresh = () => {
    if (iframeRef.current && model?.id) {
      setIsLoading(true);
      setError(null);
      iframeRef.current.src = cadApi.getModelPreviewUrl(model.id);
    }
  };

  const formatFileSize = (sizeKb) => {
    if (sizeKb < 1024) return `${sizeKb} KB`;
    return `${(sizeKb / 1024).toFixed(1)} MB`;
  };

  const formatTime = (seconds) => {
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = (seconds % 60).toFixed(0);
    return `${minutes}m ${remainingSeconds}s`;
  };

  if (!model) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-8 text-center">
        <div className="text-gray-400 mb-4">
          <div className="w-16 h-16 mx-auto mb-4 bg-gray-100 rounded-full flex items-center justify-center">
            🎯
          </div>
        </div>
        <h3 className="text-xl font-semibold text-gray-700 mb-2">
          No Model Selected
        </h3>
        <p className="text-gray-500">
          Generate a new model or select one from your library to view it here
        </p>
      </div>
    );
  }

  // Much bigger iframe - 700px default, fullscreen available
  const viewerHeight = isFullscreen ? 'calc(100vh - 200px)' : '700px';

  return (
    <div className={`bg-white rounded-xl shadow-lg overflow-hidden ${isFullscreen ? 'fixed inset-4 z-50' : ''}`}>
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-4 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold">{model.name}</h3>
            <p className="text-blue-100 text-sm">{model.description}</p>
          </div>
          
          <div className="flex items-center space-x-2">
            {isNewModel && (
              <div className="bg-green-500 px-3 py-1 rounded-full text-xs font-medium">
                ✨ New Model
              </div>
            )}
            <div className="bg-blue-500 px-3 py-1 rounded-full text-xs font-medium">
              📺 Enhanced Iframe
            </div>
          </div>
        </div>
        
        {/* Model Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4 text-xs">
          {model.vertex_count && (
            <div className="bg-white/10 rounded-lg p-2">
              <div className="text-blue-100">Vertices</div>
              <div className="font-semibold">{model.vertex_count.toLocaleString()}</div>
            </div>
          )}
          
          {model.generation_time && (
            <div className="bg-white/10 rounded-lg p-2">
              <div className="text-blue-100">Generated</div>
              <div className="font-semibold">{formatTime(model.generation_time)}</div>
            </div>
          )}
          
          {model.file_size_kb && (
            <div className="bg-white/10 rounded-lg p-2">
              <div className="text-blue-100">File Size</div>
              <div className="font-semibold">{formatFileSize(model.file_size_kb)}</div>
            </div>
          )}
          
          <div className="bg-white/10 rounded-lg p-2">
            <div className="text-blue-100">Created</div>
            <div className="font-semibold">
              {new Date(model.created_at).toLocaleDateString()}
            </div>
          </div>
        </div>
      </div>

      {/* Viewer Controls */}
      <div className="bg-gray-50 p-3 border-b">
        <div className="flex items-center justify-between">
          <div className="text-sm text-gray-600">
            🎮 Drag to rotate • Scroll to zoom • Right-click to pan • Full 3D controls
          </div>
          
          <div className="flex space-x-2">
            <button
              onClick={handleRefresh}
              className="p-2 bg-white rounded-lg shadow-sm hover:bg-gray-50 transition-colors"
              title="Refresh viewer"
            >
              <RotateCcw className="w-4 h-4 text-gray-600" />
            </button>
            
            <button
              onClick={handleMaximize}
              className="p-2 bg-white rounded-lg shadow-sm hover:bg-gray-50 transition-colors"
              title={isFullscreen ? "Exit fullscreen" : "Fullscreen mode"}
            >
              <Maximize className="w-4 h-4 text-gray-600" />
            </button>
            
            <button
              onClick={handleFullscreen}
              className="p-2 bg-white rounded-lg shadow-sm hover:bg-gray-50 transition-colors"
              title="Open in new tab"
            >
              <ExternalLink className="w-4 h-4 text-gray-600" />
            </button>
          </div>
        </div>
      </div>

      {/* Enhanced 3D Viewer - Much Bigger! */}
      <div className="relative" style={{ height: viewerHeight }}>
        {isLoading && (
          <div className="absolute inset-0 bg-gray-100 flex items-center justify-center z-10">
            <div className="text-center">
              <div className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-2"></div>
              <p className="text-sm text-gray-600">Loading 3D model...</p>
            </div>
          </div>
        )}
        
        {error && (
          <div className="absolute inset-0 bg-red-50 flex items-center justify-center z-10">
            <div className="text-center text-red-600">
              <div className="text-2xl mb-2">⚠️</div>
              <p className="text-sm">{error}</p>
              <button
                onClick={handleRefresh}
                className="mt-2 px-4 py-2 bg-red-100 hover:bg-red-200 rounded-lg text-xs transition-colors"
              >
                Try Again
              </button>
            </div>
          </div>
        )}
        
        {/* Much Bigger Iframe - 700px vs 500px! */}
        <iframe
          ref={iframeRef}
          title="3D CAD Model Viewer"
          className="w-full h-full border-0"
          onLoad={handleIframeLoad}
          onError={handleIframeError}
          sandbox="allow-scripts allow-same-origin"
          style={{ 
            minHeight: '600px',
            borderRadius: '0 0 12px 12px'
          }}
        />
      </div>

      {/* Download Section */}
      <div className="p-4 bg-gray-50 border-t">
        <h4 className="text-sm font-medium text-gray-700 mb-3">📥 Download Files</h4>
        
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
          {model.html_url && (
            <a
              href={cadApi.getDownloadUrl(model.html_url)}
              download
              className="flex items-center justify-center space-x-2 p-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors text-sm"
            >
              <Download className="w-4 h-4" />
              <span>HTML Viewer</span>
            </a>
          )}
          
          {model.stl_url && (
            <a
              href={cadApi.getDownloadUrl(model.stl_url)}
              download
              className="flex items-center justify-center space-x-2 p-3 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors text-sm"
            >
              <Download className="w-4 h-4" />
              <span>STL File</span>
            </a>
          )}
          
          {model.step_url && (
            <a
              href={cadApi.getDownloadUrl(model.step_url)}
              download
              className="flex items-center justify-center space-x-2 p-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors text-sm"
            >
              <Download className="w-4 h-4" />
              <span>STEP File</span>
            </a>
          )}
        </div>
        
        <p className="text-xs text-gray-500 mt-2">
          💡 STL files for 3D printing • STEP files for CAD software • Enhanced iframe viewer
        </p>
      </div>
      
      {/* Fullscreen overlay */}
      {isFullscreen && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-40"
          onClick={handleMaximize}
        />
      )}
    </div>
  );
};

export default EnhancedCADViewer;