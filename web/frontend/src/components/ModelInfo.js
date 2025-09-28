import React, { useState, useEffect } from 'react';
import { Card, Typography, Table, Tag, Space, Alert } from 'antd';
import axios from 'axios';

const { Title, Text } = Typography;

const ModelInfo = () => {
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchModelInfo();
  }, []);

  const fetchModelInfo = async () => {
    try {
      const response = await axios.get('/models/info');
      setModelInfo(response.data);
    } catch (err) {
      setError('Failed to fetch model information');
      console.error('Model info error:', err);
    } finally {
      setLoading(false);
    }
  };

  const modelColumns = [
    {
      title: 'Model Name',
      dataIndex: 'name',
      key: 'name',
      render: (name) => <Tag color="blue">{name}</Tag>,
    },
    {
      title: 'Weight',
      dataIndex: 'weight',
      key: 'weight',
      render: (weight) => `${(weight * 100).toFixed(1)}%`,
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status) => (
        <Tag color={status ? 'green' : 'red'}>
          {status ? 'Ready' : 'Not Ready'}
        </Tag>
      ),
    },
  ];

  if (loading) {
    return <div>Loading model information...</div>;
  }

  if (error) {
    return <Alert message="Error" description={error} type="error" />;
  }

  const modelData = modelInfo?.models?.models?.map((model, index) => ({
    key: index,
    name: model,
    weight: modelInfo.models.normalized_weights[model],
    status: modelInfo.models.is_fitted,
  })) || [];

  return (
    <div>
      <Title level={2}>🤖 Model Information</Title>
      
      <Card title="Ensemble Models" style={{ marginBottom: 24 }}>
        <Table
          columns={modelColumns}
          dataSource={modelData}
          pagination={false}
          size="small"
        />
      </Card>

      <Card title="System Details">
        <Space direction="vertical" style={{ width: '100%' }}>
          <div>
            <Text strong>Total Models: </Text>
            <Text>{modelInfo?.models?.model_count || 0}</Text>
          </div>
          <div>
            <Text strong>Ensemble Status: </Text>
            <Tag color={modelInfo?.models?.is_fitted ? 'green' : 'red'}>
              {modelInfo?.models?.is_fitted ? 'Fitted' : 'Not Fitted'}
            </Tag>
          </div>
          <div>
            <Text strong>Last Updated: </Text>
            <Text>{modelInfo?.last_updated ? new Date(modelInfo.last_updated).toLocaleString() : 'Unknown'}</Text>
          </div>
        </Space>
      </Card>
    </div>
  );
};

export default ModelInfo;

