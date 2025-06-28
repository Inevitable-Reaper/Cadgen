import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 2 minutes for CAD generation
});

// API service functions
export const cadApi = {
  // Health check
  async checkHealth() {
    const response = await api.get('/health');
    return response.data;
  },

  // Generate CAD model
  async generateModel(description, complexity = 'medium', manufacturingProcess = 'general') {
    const response = await api.post('/generate', {
      description,
      complexity,
      manufacturing_process: manufacturingProcess
    });
    return response.data;
  },

  // Get job status
  async getJobStatus(jobId) {
    const response = await api.get(`/jobs/${jobId}`);
    return response.data;
  },

  // Get all models
  async getAllModels() {
    const response = await api.get('/models');
    return response.data;
  },

  // Get specific model
  async getModel(modelId) {
    const response = await api.get(`/models/${modelId}`);
    return response.data;
  },

  // Delete model
  async deleteModel(modelId) {
    const response = await api.delete(`/models/${modelId}`);
    return response.data;
  },

  // Get model preview URL
  getModelPreviewUrl(modelId) {
    return `${API_BASE_URL}/models/${modelId}/preview`;
  },

  // Get download URLs
  getDownloadUrl(path) {
    return `http://localhost:8000${path}`;
  }
};

export default cadApi;
