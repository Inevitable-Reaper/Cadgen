import React, { useState, useEffect } from 'react';
import { Loader2, AlertCircle, CheckCircle, Clock } from 'lucide-react';
import { cadApi } from '../services/api';

const ModelGenerator = ({ onModelGenerated }) => {
  const [description, setDescription] = useState('');
  const [complexity, setComplexity] = useState('medium');
  const [manufacturingProcess, setManufacturingProcess] = useState('general');
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentJob, setCurrentJob] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);

  // Poll job status
  useEffect(() => {
    let interval;
    if (currentJob && jobStatus?.status !== 'completed' && jobStatus?.status !== 'failed') {
      interval = setInterval(async () => {
        try {
          const status = await cadApi.getJobStatus(currentJob);
          setJobStatus(status);
          
          if (status.status === 'completed') {
            const model = await cadApi.getModel(currentJob);
            onModelGenerated(model);
            setIsGenerating(false);
            setCurrentJob(null);
          } else if (status.status === 'failed') {
            setIsGenerating(false);
            setCurrentJob(null);
          }
        } catch (error) {
          console.error('Error polling job status:', error);
        }
      }, 2000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [currentJob, jobStatus, onModelGenerated]);

  const handleGenerate = async () => {
    if (!description.trim()) {
      alert('Please enter a model description');
      return;
    }

    setIsGenerating(true);
    setJobStatus(null);

    try {
      const response = await cadApi.generateModel(description, complexity, manufacturingProcess);
      setCurrentJob(response.job_id);
      setJobStatus({ status: 'pending', progress: 0, message: 'Job started...' });
    } catch (error) {
      console.error('Generation failed:', error);
      alert('Failed to start generation: ' + error.message);
      setIsGenerating(false);
    }
  };

  const examples = [
    'Create a baseball bat 32 inches long with a grip handle',
    'Design a hex bolt M8 x 25mm with proper threads',
    'Make a spur gear with 20 teeth and 50mm diameter',
    'Build an L-bracket 60x40x5mm with mounting holes',
    'Create a coffee mug with handle, 250ml capacity',
    'Design a simple canoe hull 3 meters long'
  ];

  const getStatusIcon = () => {
    if (!jobStatus) return null;
    
    switch (jobStatus.status) {
      case 'pending':
      case 'processing':
        return <Loader2 className="w-5 h-5 animate-spin text-blue-500" />;
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'failed':
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      default:
        return <Clock className="w-5 h-5 text-gray-500" />;
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          🛠️ Generate CAD Model
        </h2>
        <p className="text-gray-600">
          Describe your 3D model and our AI will generate it for you
        </p>
      </div>

      {/* Description Input */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Model Description *
        </label>
        <textarea
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Describe your CAD model in detail..."
          rows={4}
          className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
          disabled={isGenerating}
        />
      </div>

      {/* Settings */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Complexity Level
          </label>
          <select
            value={complexity}
            onChange={(e) => setComplexity(e.target.value)}
            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            disabled={isGenerating}
          >
            <option value="simple">Simple</option>
            <option value="medium">Medium</option>
            <option value="complex">Complex</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Manufacturing Process
          </label>
          <select
            value={manufacturingProcess}
            onChange={(e) => setManufacturingProcess(e.target.value)}
            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            disabled={isGenerating}
          >
            <option value="general">General</option>
            <option value="3d_printing">3D Printing</option>
            <option value="cnc_milling">CNC Milling</option>
            <option value="casting">Casting</option>
          </select>
        </div>
      </div>

      {/* Generate Button */}
      <button
        onClick={handleGenerate}
        disabled={isGenerating || !description.trim()}
        className={`w-full py-3 px-6 rounded-lg font-semibold text-white transition-all duration-200 ${
          isGenerating || !description.trim()
            ? 'bg-gray-400 cursor-not-allowed'
            : 'bg-blue-600 hover:bg-blue-700 active:bg-blue-800'
        }`}
      >
        {isGenerating ? (
          <div className="flex items-center justify-center space-x-2">
            <Loader2 className="w-5 h-5 animate-spin" />
            <span>Generating...</span>
          </div>
        ) : (
          '🚀 Generate 3D Model'
        )}
      </button>

      {/* Job Status */}
      {jobStatus && (
        <div className="mt-4 p-4 bg-gray-50 rounded-lg">
          <div className="flex items-center space-x-3 mb-2">
            {getStatusIcon()}
            <span className="font-medium text-gray-900">
              Status: {jobStatus.status}
            </span>
          </div>
          
          <div className="mb-2">
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                style={{ width: `${jobStatus.progress}%` }}
              />
            </div>
          </div>
          
          <p className="text-sm text-gray-600">{jobStatus.message}</p>
        </div>
      )}

      {/* Examples */}
      <div className="mt-6">
        <h3 className="text-sm font-medium text-gray-700 mb-3">
          💡 Quick Examples (Click to use)
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {examples.map((example, index) => (
            <button
              key={index}
              onClick={() => !isGenerating && setDescription(example)}
              className="text-left p-3 text-sm bg-gray-50 hover:bg-gray-100 rounded-lg transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={isGenerating}
            >
              {example}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ModelGenerator;
