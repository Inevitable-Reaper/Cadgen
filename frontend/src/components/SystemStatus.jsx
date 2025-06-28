import React, { useState, useEffect } from 'react';
import { AlertCircle, CheckCircle, Wifi, WifiOff } from 'lucide-react';
import { cadApi } from '../services/api';

const SystemStatus = () => {
  const [status, setStatus] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    checkSystemHealth();
    const interval = setInterval(checkSystemHealth, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const checkSystemHealth = async () => {
    try {
      const health = await cadApi.checkHealth();
      setStatus(health);
      setIsLoading(false);
    } catch (error) {
      setStatus({ status: 'error', message: 'Cannot connect to backend' });
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 mb-4">
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 border-2 border-yellow-500 border-t-transparent rounded-full animate-spin"></div>
          <span className="text-sm text-yellow-700">Checking system status...</span>
        </div>
      </div>
    );
  }

  const isHealthy = status?.status === 'healthy' && status?.cad_system_ready && status?.api_key_configured;

  return (
    <div className={`border rounded-lg p-3 mb-4 ${
      isHealthy 
        ? 'bg-green-50 border-green-200' 
        : 'bg-red-50 border-red-200'
    }`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          {isHealthy ? (
            <CheckCircle className="w-4 h-4 text-green-600" />
          ) : (
            <AlertCircle className="w-4 h-4 text-red-600" />
          )}
          
          <span className={`text-sm font-medium ${
            isHealthy ? 'text-green-700' : 'text-red-700'
          }`}>
            {isHealthy ? '✅ System Ready' : '❌ System Issues'}
          </span>
        </div>

        <div className="flex items-center space-x-4 text-xs">
          <div className="flex items-center space-x-1">
            {status?.cad_system_ready ? (
              <Wifi className="w-3 h-3 text-green-600" />
            ) : (
              <WifiOff className="w-3 h-3 text-red-600" />
            )}
            <span className={status?.cad_system_ready ? 'text-green-600' : 'text-red-600'}>
              CAD System
            </span>
          </div>

          <div className="flex items-center space-x-1">
            {status?.api_key_configured ? (
              <CheckCircle className="w-3 h-3 text-green-600" />
            ) : (
              <AlertCircle className="w-3 h-3 text-red-600" />
            )}
            <span className={status?.api_key_configured ? 'text-green-600' : 'text-red-600'}>
              API Key
            </span>
          </div>
        </div>
      </div>

      {!isHealthy && (
        <div className="mt-2 text-xs text-red-600">
          {!status?.api_key_configured && 'Please configure GEMINI_API_KEY in backend/.env file'}
          {!status?.cad_system_ready && 'CAD system initialization failed'}
        </div>
      )}
    </div>
  );
};

export default SystemStatus;
