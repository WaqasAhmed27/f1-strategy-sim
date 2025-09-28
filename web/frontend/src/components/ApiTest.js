import React, { useState, useEffect } from 'react';
import { Card, Button, Alert, Typography, Space } from 'antd';
import { getApiUrl } from '../config';
import axios from 'axios';

const { Title, Text } = Typography;

const ApiTest = () => {
  const [apiStatus, setApiStatus] = useState('unknown');
  const [apiUrl, setApiUrl] = useState('');
  const [error, setError] = useState(null);

  useEffect(() => {
    setApiUrl(getApiUrl('/health'));
  }, []);

  const testApi = async () => {
    setApiStatus('testing');
    setError(null);
    
    try {
      console.log('🧪 Testing API connection to:', apiUrl);
      const response = await axios.get(apiUrl);
      console.log('✅ API Response:', response.data);
      setApiStatus('connected');
    } catch (err) {
      console.error('❌ API Test Failed:', err);
      setError(err.message);
      setApiStatus('failed');
    }
  };

  return (
    <Card title="🔧 API Connection Test" style={{ marginBottom: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }}>
        <div>
          <Text strong>API URL: </Text>
          <Text code>{apiUrl}</Text>
        </div>
        
        <div>
          <Text strong>Status: </Text>
          <Text 
            style={{ 
              color: apiStatus === 'connected' ? 'green' : 
                     apiStatus === 'failed' ? 'red' : 'orange' 
            }}
          >
            {apiStatus.toUpperCase()}
          </Text>
        </div>

        <Button 
          type="primary" 
          onClick={testApi}
          loading={apiStatus === 'testing'}
        >
          Test API Connection
        </Button>

        {error && (
          <Alert
            message="Connection Error"
            description={error}
            type="error"
            showIcon
          />
        )}

        {apiStatus === 'connected' && (
          <Alert
            message="API Connected Successfully!"
            description="The backend API is responding correctly."
            type="success"
            showIcon
          />
        )}
      </Space>
    </Card>
  );
};

export default ApiTest;
