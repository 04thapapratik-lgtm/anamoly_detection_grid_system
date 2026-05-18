import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, ScatterChart, Scatter, BarChart, Bar, Legend } from 'recharts';
import { Activity, Zap, AlertTriangle, ShieldCheck, Cpu, Target } from 'lucide-react';

export default function App() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('diagnostics');
  const [selectedModel, setSelectedModel] = useState('rf_prediction');

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data.json`)
      .then(res => res.json())
      .then(json => {
        setData(json);
        setLoading(false);
      })
      .catch(err => console.error("Failed to load data", err));
  }, []);

  if (loading) {
    return (
      <div className="dashboard-container" style={{ alignItems: 'center', justifyContent: 'center', color: 'var(--accent-cyan)' }}>
        <Activity size={48} className="status-badge" />
        <h2 style={{ marginLeft: '1rem' }}>Initializing Grid Systems...</h2>
      </div>
    );
  }

  // --- Model Definitions ---
  const models = {
    'rf_prediction': { name: 'Random Forest (SOTA)', key: 'rf_prediction', color: 'var(--accent-magenta)' },
    'is_z_anomaly': { name: 'Z-Score Screening', key: 'is_z_anomaly', color: '#ff9900' },
    'is_iso_anomaly': { name: 'Isolation Forest', key: 'is_iso_anomaly', color: '#00ffaa' },
    'is_km_anomaly': { name: 'K-Means Clustering', key: 'is_km_anomaly', color: 'var(--accent-yellow)' },
  };

  const activeModel = models[selectedModel];

  // --- Calculations for Overview ---
  const totalAnomalies = data.filter(d => d[activeModel.key] > 0).length;
  const anomalyRate = ((totalAnomalies / data.length) * 100).toFixed(1);
  const maxPower = Math.max(...data.map(d => d.power)).toFixed(2);

  // --- Calculations for ML Diagnostics ---
  const calculateMetrics = (predictKey) => {
    let tp = 0, fp = 0, fn = 0, tn = 0;
    data.forEach(d => {
      const isTrueAnomaly = d.fault_indicator > 0;
      const isPredAnomaly = d[predictKey] > 0;
      if (isTrueAnomaly && isPredAnomaly) tp++;
      if (!isTrueAnomaly && isPredAnomaly) fp++;
      if (isTrueAnomaly && !isPredAnomaly) fn++;
      if (!isTrueAnomaly && !isPredAnomaly) tn++;
    });
    const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
    const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
    const f1 = precision + recall === 0 ? 0 : 2 * (precision * recall) / (precision + recall);
    const accuracy = (tp + tn) / data.length;
    return { precision: precision.toFixed(2), recall: recall.toFixed(2), f1: f1.toFixed(2), accuracy: accuracy.toFixed(2), tp, fp, fn, tn };
  };

  const currentMetrics = calculateMetrics(activeModel.key);

  const rfMetrics = calculateMetrics('rf_prediction');
  const isoMetrics = calculateMetrics('is_iso_anomaly');
  const kmMetrics = calculateMetrics('is_km_anomaly');
  const zMetrics = calculateMetrics('is_z_anomaly');

  const comparisonData = [
    { name: 'RF', F1: parseFloat(rfMetrics.f1), Precision: parseFloat(rfMetrics.precision), Recall: parseFloat(rfMetrics.recall) },
    { name: 'IF', F1: parseFloat(isoMetrics.f1), Precision: parseFloat(isoMetrics.precision), Recall: parseFloat(isoMetrics.recall) },
    { name: 'K-Means', F1: parseFloat(kmMetrics.f1), Precision: parseFloat(kmMetrics.precision), Recall: parseFloat(kmMetrics.recall) },
    { name: 'Z-Score', F1: parseFloat(zMetrics.f1), Precision: parseFloat(zMetrics.precision), Recall: parseFloat(zMetrics.recall) },
  ];

  // --- Fault Logs Data ---
  const recentFaults = data.filter(d => d[activeModel.key] > 0).slice(-20).reverse();

  return (
    <div className="dashboard-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="brand">
          <Zap size={24} color="var(--accent-cyan)" />
          AURA-SENTINEL
        </div>
        <div className={`nav-item ${activeTab === 'overview' ? 'active' : ''}`} onClick={() => setActiveTab('overview')}>
          <Activity size={18} />
          <span>Grid Overview</span>
        </div>
        <div className={`nav-item ${activeTab === 'diagnostics' ? 'active' : ''}`} onClick={() => setActiveTab('diagnostics')}>
          <Cpu size={18} />
          <span>ML Diagnostics</span>
        </div>
        <div className={`nav-item ${activeTab === 'logs' ? 'active' : ''}`} onClick={() => setActiveTab('logs')}>
          <AlertTriangle size={18} />
          <span>Fault Logs</span>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <header className="header" style={{ marginBottom: '1rem' }}>
          <div>
            <h1>Smart Grid Command Center</h1>
            <div style={{ color: 'var(--text-secondary)', marginTop: '4px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Target size={16} /> Currently Evaluating: <strong style={{ color: activeModel.color }}>{activeModel.name}</strong>
            </div>
          </div>
          <div className="status-badge">
            <ShieldCheck size={16} />
            <span>SYSTEM ACTIVE</span>
          </div>
        </header>

        {activeTab === 'overview' && (
          <div className="tab-fade-in">
            {/* KPI Grid */}
            <div className="kpi-grid">
              <div className="glass-panel">
                <div className="kpi-label">Monitored Nodes</div>
                <div className="kpi-value">{data.length}</div>
                <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Live Telemetry</div>
              </div>
              <div className="glass-panel">
                <div className="kpi-label">Detected Anomalies</div>
                <div className="kpi-value" style={{ color: activeModel.color }}>{totalAnomalies}</div>
                <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Via {activeModel.name}</div>
              </div>
              <div className="glass-panel">
                <div className="kpi-label">Peak Power Usage</div>
                <div className="kpi-value">{maxPower} kW</div>
                <div style={{ color: 'var(--accent-cyan)', fontSize: '0.8rem' }}>Last 24 Hours</div>
              </div>
            </div>

            {/* Charts Grid */}
            <div className="chart-grid">
              {/* Main Telemetry Chart */}
              <div className="glass-panel">
                <div className="panel-title">
                  <Activity size={20} color="var(--accent-cyan)" />
                  Power Telemetry & Fault Detection
                </div>
                <div className="chart-container">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={data}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                      <XAxis 
                        dataKey="timestamp" 
                        stroke="var(--text-secondary)" 
                        tick={{fill: 'var(--text-secondary)'}}
                        tickFormatter={(val) => val.split(' ')[1]}
                      />
                      <YAxis stroke="var(--text-secondary)" tick={{fill: 'var(--text-secondary)'}} />
                      <RechartsTooltip 
                        contentStyle={{ backgroundColor: 'rgba(11, 12, 16, 0.9)', border: '1px solid var(--accent-cyan)' }}
                        itemStyle={{ color: '#fff' }}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="power" 
                        stroke="var(--accent-cyan)" 
                        strokeWidth={2}
                        dot={false}
                        name="Power (kW)"
                      />
                      <Scatter 
                        data={data.filter(d => d[activeModel.key] > 0)} 
                        dataKey="power" 
                        fill={activeModel.color} 
                        name="Detected Fault"
                        className="anomaly-dot"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* PCA Clustering Scatter */}
              <div className="glass-panel">
                <div className="panel-title">
                  <Cpu size={20} color={activeModel.color} />
                  {activeModel.name} Space
                </div>
                <div className="chart-container">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis type="number" dataKey="pca1" name="PCA 1" stroke="var(--text-secondary)" tick={false} />
                      <YAxis type="number" dataKey="pca2" name="PCA 2" stroke="var(--text-secondary)" tick={false} />
                      <RechartsTooltip 
                        cursor={{ strokeDasharray: '3 3' }}
                        contentStyle={{ backgroundColor: 'rgba(11, 12, 16, 0.9)', border: `1px solid ${activeModel.color}` }}
                      />
                      <Scatter 
                        name="Normal Operation" 
                        data={data.filter(d => d[activeModel.key] == 0)} 
                        fill="var(--accent-cyan)" 
                        opacity={0.6}
                      />
                      <Scatter 
                        name="Outlier Behavior" 
                        data={data.filter(d => d[activeModel.key] > 0)} 
                        fill={activeModel.color} 
                      />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'diagnostics' && (
          <div className="tab-fade-in">
            {/* Model Switcher */}
            <div style={{ display: 'flex', gap: '1rem', marginBottom: '2rem' }}>
              {Object.values(models).map(model => (
                <button 
                  key={model.key}
                  onClick={() => setSelectedModel(model.key)}
                  style={{
                    padding: '12px 24px',
                    background: selectedModel === model.key ? 'rgba(0, 240, 255, 0.1)' : 'var(--panel-bg)',
                    border: `1px solid ${selectedModel === model.key ? model.color : 'rgba(255,255,255,0.1)'}`,
                    color: selectedModel === model.key ? model.color : 'var(--text-secondary)',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    fontWeight: 'bold',
                    transition: 'all 0.3s ease',
                    backdropFilter: 'blur(10px)'
                  }}
                >
                  {model.name}
                </button>
              ))}
            </div>

            <div className="glass-panel" style={{ marginBottom: '2rem', border: `1px solid ${activeModel.color}40` }}>
              <div className="panel-title"><Cpu size={20} color={activeModel.color} /> {activeModel.name} Performance Metrics</div>
              <div className="kpi-grid" style={{ marginBottom: 0 }}>
                <div>
                  <div className="kpi-label">F1-Score (Faults)</div>
                  <div className="kpi-value">{currentMetrics.f1}</div>
                </div>
                <div>
                  <div className="kpi-label">Precision</div>
                  <div className="kpi-value">{currentMetrics.precision}</div>
                </div>
                <div>
                  <div className="kpi-label">Recall</div>
                  <div className="kpi-value">{currentMetrics.recall}</div>
                </div>
                <div>
                  <div className="kpi-label">Overall Accuracy</div>
                  <div className="kpi-value" style={{color: activeModel.color}}>{currentMetrics.accuracy}</div>
                </div>
              </div>
            </div>

            <div className="chart-grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
              <div className="glass-panel">
                <div className="panel-title">Cross-Model Comparison</div>
                <div className="chart-container">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={comparisonData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                      <XAxis dataKey="name" stroke="var(--text-secondary)" />
                      <YAxis stroke="var(--text-secondary)" domain={[0, 1]} />
                      <RechartsTooltip contentStyle={{ backgroundColor: '#0b0c10', borderColor: 'var(--text-secondary)' }} />
                      <Legend />
                      <Bar dataKey="F1" fill="var(--accent-cyan)" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="Precision" fill="var(--accent-yellow)" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="Recall" fill="var(--accent-magenta)" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
              
              <div className="glass-panel">
                <div className="panel-title">Confusion Matrix ({activeModel.name})</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '2rem' }}>
                  <div style={{ padding: '1rem', background: 'rgba(0, 240, 255, 0.1)', borderRadius: '8px', border: '1px solid rgba(0, 240, 255, 0.3)', textAlign: 'center' }}>
                    <div style={{ fontSize: '2rem', color: 'var(--accent-cyan)', fontWeight: 'bold' }}>{currentMetrics.tn}</div>
                    <div style={{ color: 'var(--text-secondary)' }}>True Negatives</div>
                  </div>
                  <div style={{ padding: '1rem', background: 'rgba(255, 0, 85, 0.1)', borderRadius: '8px', border: '1px solid rgba(255, 0, 85, 0.3)', textAlign: 'center' }}>
                    <div style={{ fontSize: '2rem', color: 'var(--accent-magenta)', fontWeight: 'bold' }}>{currentMetrics.fp}</div>
                    <div style={{ color: 'var(--text-secondary)' }}>False Positives</div>
                  </div>
                  <div style={{ padding: '1rem', background: 'rgba(255, 0, 85, 0.1)', borderRadius: '8px', border: '1px solid rgba(255, 0, 85, 0.3)', textAlign: 'center' }}>
                    <div style={{ fontSize: '2rem', color: 'var(--accent-magenta)', fontWeight: 'bold' }}>{currentMetrics.fn}</div>
                    <div style={{ color: 'var(--text-secondary)' }}>False Negatives</div>
                  </div>
                  <div style={{ padding: '1rem', background: 'rgba(0, 240, 255, 0.1)', borderRadius: '8px', border: '1px solid rgba(0, 240, 255, 0.3)', textAlign: 'center' }}>
                    <div style={{ fontSize: '2rem', color: 'var(--accent-cyan)', fontWeight: 'bold' }}>{currentMetrics.tp}</div>
                    <div style={{ color: 'var(--text-secondary)' }}>True Positives</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'logs' && (
          <div className="tab-fade-in glass-panel">
            <div className="panel-title"><AlertTriangle size={20} color={activeModel.color} /> Recent Fault Logs ({activeModel.name})</div>
            <div style={{ overflowX: 'auto', marginTop: '1rem' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)', color: 'var(--text-secondary)' }}>
                    <th style={{ padding: '12px 8px' }}>Timestamp</th>
                    <th style={{ padding: '12px 8px' }}>Power (kW)</th>
                    <th style={{ padding: '12px 8px' }}>Voltage (V)</th>
                    <th style={{ padding: '12px 8px' }}>Current (A)</th>
                    <th style={{ padding: '12px 8px' }}>True Label</th>
                    <th style={{ padding: '12px 8px' }}>Detection Source</th>
                  </tr>
                </thead>
                <tbody>
                  {recentFaults.map(fault => (
                    <tr key={fault.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                      <td style={{ padding: '12px 8px' }}>{fault.timestamp}</td>
                      <td style={{ padding: '12px 8px', color: activeModel.color }}>{fault.power.toFixed(2)}</td>
                      <td style={{ padding: '12px 8px' }}>{fault.voltage.toFixed(2)}</td>
                      <td style={{ padding: '12px 8px' }}>{fault.current.toFixed(2)}</td>
                      <td style={{ padding: '12px 8px' }}>{fault.fault_indicator > 0 ? 'Fault' : 'Normal'}</td>
                      <td style={{ padding: '12px 8px' }}>
                        <span style={{ padding: '4px 8px', background: `${activeModel.color}20`, borderRadius: '4px', fontSize: '0.8rem', color: activeModel.color }}>{activeModel.name}</span>
                      </td>
                    </tr>
                  ))}
                  {recentFaults.length === 0 && (
                    <tr>
                      <td colSpan="6" style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>No recent faults detected by this model.</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

      </main>
    </div>
  );
}
