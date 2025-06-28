import React, { useState, useEffect } from 'react';
import { Trash2, Eye, Calendar, HardDrive, Box } from 'lucide-react';
import { cadApi } from '../services/api';

const ModelLibrary = ({ onModelSelect, selectedModel }) => {
  const [models, setModels] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      setIsLoading(true);
      const modelList = await cadApi.getAllModels();
      setModels(modelList);
      setError(null);
    } catch (err) {
      setError('Failed to load models');
      console.error('Error loading models:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteModel = async (modelId, event) => {
    event.stopPropagation(); // Prevent triggering model selection
    
    if (window.confirm('Are you sure you want to delete this model?')) {
      try {
        await cadApi.deleteModel(modelId);
        setModels(models.filter(model => model.id !== modelId));
        
        // If deleted model was selected, clear selection
        if (selectedModel?.id === modelId) {
          onModelSelect(null);
        }
      } catch (err) {
        alert('Failed to delete model: ' + err.message);
      }
    }
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = now - date;
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays} days ago`;
    
    return date.toLocaleDateString();
  };

  const formatFileSize = (sizeKb) => {
    if (sizeKb < 1024) return `${sizeKb} KB`;
    return `${(sizeKb / 1024).toFixed(1)} MB`;
  };

  const getComplexityColor = (vertexCount) => {
    if (!vertexCount) return 'bg-gray-100 text-gray-800';
    if (vertexCount < 500) return 'bg-green-100 text-green-800';
    if (vertexCount < 2000) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  };

  const getComplexityLabel = (vertexCount) => {
    if (!vertexCount) return 'Unknown';
    if (vertexCount < 500) return 'Simple';
    if (vertexCount < 2000) return 'Medium';
    return 'Complex';
  };

  if (isLoading) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-4">📚 Model Library</h2>
        <div className="flex items-center justify-center py-8">
          <div className="text-center">
            <div className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-2"></div>
            <p className="text-sm text-gray-600">Loading models...</p>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-4">📚 Model Library</h2>
        <div className="text-center py-8">
          <div className="text-red-500 text-2xl mb-2">⚠️</div>
          <p className="text-red-600 text-sm">{error}</p>
          <button
            onClick={loadModels}
            className="mt-2 px-4 py-2 bg-red-100 hover:bg-red-200 rounded-lg text-xs transition-colors"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl shadow-lg">
      {/* Header */}
      <div className="p-6 border-b">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-bold text-gray-900">📚 Model Library</h2>
          <div className="text-sm text-gray-500">
            {models.length} model{models.length !== 1 ? 's' : ''}
          </div>
        </div>
      </div>

      {/* Models List */}
      <div className="max-h-96 overflow-y-auto">
        {models.length === 0 ? (
          <div className="p-8 text-center">
            <div className="text-gray-400 text-4xl mb-4">📦</div>
            <h3 className="text-lg font-medium text-gray-700 mb-2">No Models Yet</h3>
            <p className="text-gray-500 text-sm">
              Generate your first 3D model to see it here
            </p>
          </div>
        ) : (
          <div className="divide-y divide-gray-100">
            {models.map((model) => (
              <div
                key={model.id}
                onClick={() => onModelSelect(model)}
                className={`p-4 cursor-pointer transition-colors hover:bg-gray-50 ${
                  selectedModel?.id === model.id ? 'bg-blue-50 border-r-4 border-blue-500' : ''
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1 min-w-0">
                    {/* Model Name and Description */}
                    <div className="flex items-center space-x-3 mb-2">
                      <div className="flex-shrink-0 w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center text-white font-bold text-sm">
                        {model.name.charAt(0)}
                      </div>
                      
                      <div className="flex-1 min-w-0">
                        <h3 className="text-sm font-medium text-gray-900 truncate">
                          {model.name}
                        </h3>
                        <p className="text-xs text-gray-500 truncate">
                          {model.description}
                        </p>
                      </div>
                    </div>

                    {/* Model Stats */}
                    <div className="flex items-center space-x-4 text-xs text-gray-500">
                      <div className="flex items-center space-x-1">
                        <Calendar className="w-3 h-3" />
                        <span>{formatDate(model.created_at)}</span>
                      </div>
                      
                      {model.file_size_kb && (
                        <div className="flex items-center space-x-1">
                          <HardDrive className="w-3 h-3" />
                          <span>{formatFileSize(model.file_size_kb)}</span>
                        </div>
                      )}
                      
                      {model.vertex_count && (
                        <div className="flex items-center space-x-1">
                          <Box className="w-3 h-3" />
                          <span>{model.vertex_count.toLocaleString()} vertices</span>
                        </div>
                      )}
                    </div>

                    {/* Complexity Badge */}
                    {model.vertex_count && (
                      <div className="mt-2">
                        <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getComplexityColor(model.vertex_count)}`}>
                          {getComplexityLabel(model.vertex_count)}
                        </span>
                      </div>
                    )}
                  </div>

                  {/* Actions */}
                  <div className="flex items-center space-x-2 ml-4">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onModelSelect(model);
                      }}
                      className="p-2 text-gray-400 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                      title="View model"
                    >
                      <Eye className="w-4 h-4" />
                    </button>
                    
                    <button
                      onClick={(e) => handleDeleteModel(model.id, e)}
                      className="p-2 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                      title="Delete model"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      {models.length > 0 && (
        <div className="p-4 bg-gray-50 text-center border-t">
          <button
            onClick={loadModels}
            className="text-sm text-blue-600 hover:text-blue-700 font-medium"
          >
            🔄 Refresh Library
          </button>
        </div>
      )}
    </div>
  );
};

export default ModelLibrary;
