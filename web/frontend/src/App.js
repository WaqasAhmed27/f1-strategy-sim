import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Layout, Menu, theme, ConfigProvider } from 'antd';
import { 
  DashboardOutlined, 
  BarChartOutlined, 
  SettingOutlined,
  TrophyOutlined,
  TeamOutlined
} from '@ant-design/icons';
import Dashboard from './components/Dashboard';
import Predictions from './components/Predictions';
import LiveRace from './components/LiveRace';
import ModelInfo from './components/ModelInfo';
import ErrorBoundary from './components/ErrorBoundary';
import './App.css';

const { Header, Content, Sider } = Layout;

function App() {
  const [collapsed, setCollapsed] = useState(false);
  const [selectedKey, setSelectedKey] = useState('dashboard');
  
  const {
    token: { colorBgContainer, borderRadiusLG },
  } = theme.useToken();

  const menuItems = [
    {
      key: 'dashboard',
      icon: <DashboardOutlined />,
      label: 'Dashboard',
    },
    {
      key: 'predictions',
      icon: <TrophyOutlined />,
      label: 'Race Predictions',
    },
    {
      key: 'live',
      icon: <TeamOutlined />,
      label: 'Live Race',
    },
    {
      key: 'analytics',
      icon: <BarChartOutlined />,
      label: 'Analytics',
    },
    {
      key: 'models',
      icon: <SettingOutlined />,
      label: 'Model Info',
    },
  ];

  const handleMenuClick = ({ key }) => {
    setSelectedKey(key);
  };

  return (
    <ErrorBoundary>
      <ConfigProvider
        theme={{
          token: {
            colorPrimary: '#e10600', // F1 Red
            colorSuccess: '#00d2be', // F1 Teal
            colorWarning: '#ff8700', // F1 Orange
          },
        }}
      >
        <Router>
          <Layout style={{ minHeight: '100vh' }}>
            <Sider 
              collapsible 
              collapsed={collapsed} 
              onCollapse={setCollapsed}
              theme="dark"
            >
              <div className="logo" style={{ 
                height: 32, 
                margin: 16, 
                background: 'rgba(255, 255, 255, 0.2)',
                borderRadius: 6,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white',
                fontWeight: 'bold'
              }}>
                {collapsed ? 'F1' : 'F1 Predict'}
              </div>
              <Menu
                theme="dark"
                selectedKeys={[selectedKey]}
                mode="inline"
                items={menuItems}
                onClick={handleMenuClick}
              />
            </Sider>
            <Layout>
              <Header style={{ 
                padding: 0, 
                background: colorBgContainer,
                borderBottom: '1px solid #f0f0f0'
              }}>
                <div style={{ 
                  padding: '0 24px',
                  fontSize: '18px',
                  fontWeight: 'bold',
                  color: '#e10600'
                }}>
                  🏎️ F1 Real-Time Prediction System
                </div>
              </Header>
              <Content style={{ 
                margin: '24px 16px', 
                padding: 24, 
                minHeight: 280, 
                background: colorBgContainer, 
                borderRadius: borderRadiusLG 
              }}>
                <Routes>
                  <Route path="/" element={<Dashboard />} />
                  <Route path="/dashboard" element={<Dashboard />} />
                  <Route path="/predictions" element={<Predictions />} />
                  <Route path="/live" element={<LiveRace />} />
                  <Route path="/analytics" element={<div>Analytics Coming Soon...</div>} />
                  <Route path="/models" element={<ModelInfo />} />
                </Routes>
              </Content>
            </Layout>
          </Layout>
        </Router>
      </ConfigProvider>
    </ErrorBoundary>
  );
}

export default App;

