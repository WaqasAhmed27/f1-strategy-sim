import React, { useState, useEffect, useRef } from 'react';
import { 
  Card, 
  Button, 
  Space, 
  Alert, 
  Typography, 
  Row, 
  Col,
  Statistic,
  Tag,
  List,
  Avatar
} from 'antd';
import { 
  PlayCircleOutlined, 
  PauseCircleOutlined,
  ReloadOutlined,
  ClockCircleOutlined
} from '@ant-design/icons';
import { getApiUrl, getWsUrl } from '../config';

const { Title, Text } = Typography;

const LiveRace = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [isLive, setIsLive] = useState(false);
  const [predictions, setPredictions] = useState([]);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [error, setError] = useState(null);
  const [selectedSeason] = useState(2024);
  const [selectedRace] = useState(1);
  
  const wsRef = useRef(null);
  const intervalRef = useRef(null);

  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  const connectWebSocket = () => {
    const ws = new WebSocket(getWsUrl());
    
    ws.onopen = () => {
      setIsConnected(true);
      setError(null);
      console.log('WebSocket connected');
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.type === 'prediction') {
          setPredictions(data.data.predictions);
          setLastUpdate(new Date(data.timestamp));
        } else if (data.type === 'pong') {
          console.log('Pong received');
        }
      } catch (err) {
        console.error('WebSocket message error:', err);
      }
    };
    
    ws.onclose = () => {
      setIsConnected(false);
      console.log('WebSocket disconnected');
    };
    
    ws.onerror = (error) => {
      setError('WebSocket connection error');
      console.error('WebSocket error:', error);
    };
    
    wsRef.current = ws;
  };

  const disconnectWebSocket = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsConnected(false);
  };

  const startLiveUpdates = async () => {
    if (!isConnected) {
      connectWebSocket();
      return;
    }
    
    setIsLive(true);
    
    // Send subscription message
    if (wsRef.current) {
      wsRef.current.send(JSON.stringify({
        type: 'subscribe',
        race: `${selectedSeason}_${selectedRace}_R`
      }));
    }
    
    // Start periodic updates
    intervalRef.current = setInterval(async () => {
      try {
        const response = await fetch(getApiUrl(`/live/predict/${selectedSeason}/${selectedRace}`), {
          method: 'POST'
        });
        
        if (!response.ok) {
          throw new Error('Live prediction failed');
        }
      } catch (err) {
        setError('Live prediction error: ' + err.message);
        console.error('Live prediction error:', err);
      }
    }, 30000); // Update every 30 seconds
  };

  const stopLiveUpdates = () => {
    setIsLive(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  const refreshPredictions = async () => {
    try {
      const response = await fetch(getApiUrl(`/live/predict/${selectedSeason}/${selectedRace}`), {
        method: 'POST'
      });
      
      if (!response.ok) {
        throw new Error('Prediction failed');
      }
      
      setError(null);
    } catch (err) {
      setError('Prediction error: ' + err.message);
    }
  };

  const getPositionColor = (position) => {
    if (position <= 3) return '#faad14'; // Gold
    if (position <= 10) return '#1890ff'; // Blue
    return '#d9d9d9'; // Gray
  };

  return (
    <div>
      <Title level={2}>🏁 Live Race Predictions</Title>
      
      <Card title="Live Control Panel" style={{ marginBottom: 24 }}>
        <Space>
          <Button
            type={isConnected ? 'default' : 'primary'}
            icon={isConnected ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
            onClick={isConnected ? disconnectWebSocket : connectWebSocket}
          >
            {isConnected ? 'Disconnect' : 'Connect'}
          </Button>
          
          <Button
            type={isLive ? 'default' : 'primary'}
            icon={isLive ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
            onClick={isLive ? stopLiveUpdates : startLiveUpdates}
            disabled={!isConnected}
          >
            {isLive ? 'Stop Live' : 'Start Live'}
          </Button>
          
          <Button
            icon={<ReloadOutlined />}
            onClick={refreshPredictions}
          >
            Refresh
          </Button>
        </Space>
        
        <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
          <Col span={8}>
            <Statistic
              title="Connection Status"
              value={isConnected ? 'Connected' : 'Disconnected'}
              valueStyle={{ color: isConnected ? '#52c41a' : '#ff4d4f' }}
            />
          </Col>
          <Col span={8}>
            <Statistic
              title="Live Updates"
              value={isLive ? 'Active' : 'Inactive'}
              valueStyle={{ color: isLive ? '#52c41a' : '#d9d9d9' }}
            />
          </Col>
          <Col span={8}>
            <Statistic
              title="Last Update"
              value={lastUpdate ? lastUpdate.toLocaleTimeString() : 'Never'}
              prefix={<ClockCircleOutlined />}
            />
          </Col>
        </Row>
      </Card>

      {error && (
        <Alert
          message="Error"
          description={error}
          type="error"
          style={{ marginBottom: 24 }}
        />
      )}

      <Row gutter={[16, 16]}>
        <Col span={16}>
          <Card title="Live Predictions" extra={<Tag color="red">LIVE</Tag>}>
            <List
              dataSource={predictions}
              renderItem={item => (
                <List.Item>
                  <List.Item.Meta
                    avatar={
                      <Avatar 
                        style={{ 
                          backgroundColor: getPositionColor(item.predicted_position),
                          color: 'white',
                          fontWeight: 'bold'
                        }}
                      >
                        {item.predicted_position}
                      </Avatar>
                    }
                    title={
                      <Space>
                        <Text strong>{item.driver_name}</Text>
                        <Tag color="blue">{item.team}</Tag>
                      </Space>
                    }
                    description={
                      <Space>
                        <Text>Confidence: {(item.confidence * 100).toFixed(1)}%</Text>
                        <Text>Gap: {item.gap_to_winner > 0 ? `+${item.gap_to_winner.toFixed(3)}s` : 'Leader'}</Text>
                      </Space>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>
        </Col>
        
        <Col span={8}>
          <Card title="Race Information">
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <Text strong>Season: </Text>
                <Tag color="red">{selectedSeason}</Tag>
              </div>
              <div>
                <Text strong>Race: </Text>
                <Tag color="blue">Round {selectedRace}</Tag>
              </div>
              <div>
                <Text strong>Session: </Text>
                <Tag color="green">Race</Tag>
              </div>
              <div>
                <Text strong>Drivers: </Text>
                <Text>{predictions.length}</Text>
              </div>
            </Space>
          </Card>
          
          <Card title="System Status" style={{ marginTop: 16 }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <Text strong>WebSocket: </Text>
                <Tag color={isConnected ? 'green' : 'red'}>
                  {isConnected ? 'Connected' : 'Disconnected'}
                </Tag>
              </div>
              <div>
                <Text strong>Live Updates: </Text>
                <Tag color={isLive ? 'green' : 'default'}>
                  {isLive ? 'Active' : 'Inactive'}
                </Tag>
              </div>
              <div>
                <Text strong>Last Update: </Text>
                <Text>{lastUpdate ? lastUpdate.toLocaleString() : 'Never'}</Text>
              </div>
            </Space>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default LiveRace;

