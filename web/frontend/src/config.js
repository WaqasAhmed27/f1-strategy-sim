// API Configuration
const config = {
  // API Base URL
  API_URL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  
  // WebSocket URL
  WS_URL: process.env.REACT_APP_WS_URL || 'ws://localhost:8000',
  
  // API Endpoints
  ENDPOINTS: {
    HEALTH: '/health',
    PREDICT: '/predict',
    MODELS_INFO: '/models/info',
    LIVE_PREDICT: '/live/predict',
    WS: '/ws'
  }
};

// Debug logging
console.log('🔧 F1 Prediction Config:', {
  API_URL: config.API_URL,
  WS_URL: config.WS_URL,
  NODE_ENV: process.env.NODE_ENV
});

// Helper function to get full API URL
export const getApiUrl = (endpoint) => {
  return `${config.API_URL}${endpoint}`;
};

// Helper function to get WebSocket URL
export const getWsUrl = () => {
  return `${config.WS_URL}${config.ENDPOINTS.WS}`;
};

export default config;
