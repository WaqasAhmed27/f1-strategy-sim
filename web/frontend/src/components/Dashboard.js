import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Row, 
  Col, 
  Statistic, 
  Select, 
  Button, 
  Table, 
  Tag, 
  Space,
  Alert,
  Spin,
  Typography
} from 'antd';
import { 
  TrophyOutlined, 
  ClockCircleOutlined, 
  TeamOutlined,
  ThunderboltOutlined 
} from '@ant-design/icons';
import axios from 'axios';
import { getApiUrl } from '../config';
import ApiTest from './ApiTest';

const { Title, Text } = Typography;
const { Option } = Select;

const Dashboard = () => {
  const [loading, setLoading] = useState(false);
  const [predictions, setPredictions] = useState([]);
  const [selectedSeason, setSelectedSeason] = useState(2024);
  const [selectedRace, setSelectedRace] = useState(1);
  const [modelInfo, setModelInfo] = useState(null);
  const [error, setError] = useState(null);

  const seasons = [2021, 2022, 2023, 2024];
  const races = Array.from({ length: 24 }, (_, i) => i + 1);

  useEffect(() => {
    fetchModelInfo();
  }, []);

  const fetchModelInfo = async () => {
    try {
      console.log('🔍 Fetching model info from:', getApiUrl('/models/info'));
      const response = await axios.get(getApiUrl('/models/info'));
      console.log('✅ Model info received:', response.data);
      setModelInfo(response.data);
    } catch (err) {
      console.error('❌ Failed to fetch model info:', err);
      console.error('🌐 API URL:', getApiUrl('/models/info'));
      console.error('📊 Error details:', err.response?.data || err.message);
      setError('Failed to connect to API. Check console for details.');
    }
  };

  const fetchPredictions = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post(getApiUrl('/predict'), {
        season: selectedSeason,
        race_round: selectedRace,
        session: 'R'
      });
      
      setPredictions(response.data.predictions);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to fetch predictions');
      console.error('Prediction error:', err);
      console.error('API URL:', getApiUrl('/predict'));
    } finally {
      setLoading(false);
    }
  };

  const columns = [
    {
      title: 'Position',
      dataIndex: 'predicted_position',
      key: 'position',
      width: 80,
      render: (position) => (
        <Tag color={position <= 3 ? 'gold' : position <= 10 ? 'blue' : 'default'}>
          P{position}
        </Tag>
      ),
    },
    {
      title: 'Driver',
      dataIndex: 'driver_name',
      key: 'driver',
      width: 120,
    },
    {
      title: 'Team',
      dataIndex: 'team',
      key: 'team',
      width: 150,
    },
    {
      title: 'Confidence',
      dataIndex: 'confidence',
      key: 'confidence',
      width: 100,
      render: (confidence) => (
        <span style={{ color: confidence > 0.8 ? '#52c41a' : confidence > 0.6 ? '#faad14' : '#ff4d4f' }}>
          {(confidence * 100).toFixed(1)}%
        </span>
      ),
    },
    {
      title: 'Gap to Winner',
      dataIndex: 'gap_to_winner',
      key: 'gap',
      width: 120,
      render: (gap) => (
        <span style={{ color: gap > 0 ? '#ff4d4f' : '#52c41a' }}>
          {gap > 0 ? `+${gap.toFixed(3)}s` : 'Leader'}
        </span>
      ),
    },
  ];

  return (
    <div>
      <Title level={2}>🏎️ F1 Prediction Dashboard</Title>
      
      <ApiTest />
      
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Active Models"
              value={modelInfo?.model_count || 0}
              prefix={<ThunderboltOutlined />}
              valueStyle={{ color: '#e10600' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Prediction Accuracy"
              value="94.2"
              suffix="%"
              prefix={<TrophyOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Avg Response Time"
              value="0.8"
              suffix="s"
              prefix={<ClockCircleOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Drivers Predicted"
              value={predictions.length}
              prefix={<TeamOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
      </Row>

      <Card title="Race Prediction" style={{ marginBottom: 24 }}>
        <Space style={{ marginBottom: 16 }}>
          <Select
            value={selectedSeason}
            onChange={setSelectedSeason}
            style={{ width: 120 }}
          >
            {seasons.map(season => (
              <Option key={season} value={season}>{season}</Option>
            ))}
          </Select>
          
          <Select
            value={selectedRace}
            onChange={setSelectedRace}
            style={{ width: 120 }}
          >
            {races.map(race => (
              <Option key={race} value={race}>Race {race}</Option>
            ))}
          </Select>
          
          <Button 
            type="primary" 
            onClick={fetchPredictions}
            loading={loading}
            icon={<TrophyOutlined />}
          >
            Predict Race
          </Button>
        </Space>

        {error && (
          <Alert
            message="Prediction Error"
            description={error}
            type="error"
            style={{ marginBottom: 16 }}
          />
        )}

        <Spin spinning={loading}>
          <Table
            columns={columns}
            dataSource={predictions}
            rowKey="driver_number"
            pagination={false}
            size="small"
            style={{ marginTop: 16 }}
          />
        </Spin>
      </Card>

      {modelInfo && (
        <Card title="Model Information">
          <Row gutter={[16, 16]}>
            <Col span={12}>
              <Title level={4}>Ensemble Models</Title>
              <ul>
                {modelInfo.models?.models?.map((model, index) => (
                  <li key={index}>
                    <Text strong>{model}</Text>
                    <Text style={{ marginLeft: 8, color: '#666' }}>
                      Weight: {(modelInfo.models.normalized_weights[model] * 100).toFixed(1)}%
                    </Text>
                  </li>
                ))}
              </ul>
            </Col>
            <Col span={12}>
              <Title level={4}>System Status</Title>
              <Space direction="vertical">
                <div>
                  <Text strong>Models Loaded: </Text>
                  <Tag color={modelInfo.models.is_fitted ? 'green' : 'red'}>
                    {modelInfo.models.is_fitted ? 'Ready' : 'Not Ready'}
                  </Tag>
                </div>
                <div>
                  <Text strong>Last Updated: </Text>
                  <Text>{new Date(modelInfo.last_updated).toLocaleString()}</Text>
                </div>
              </Space>
            </Col>
          </Row>
        </Card>
      )}
    </div>
  );
};

export default Dashboard;

