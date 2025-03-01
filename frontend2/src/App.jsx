// src/App.jsx
import { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [trainFile, setTrainFile] = useState(null);
  const [predictFile, setPredictFile] = useState(null);
  const [trainDatasetId, setTrainDatasetId] = useState('');
  const [predictDatasetId, setPredictDatasetId] = useState('');
  const [loading, setLoading] = useState(false);
  const [modelStatus, setModelStatus] = useState('');
  const [predictionStatus, setPredictionStatus] = useState('');
  const [modelResults, setModelResults] = useState(null);
  const [predictionResults, setPredicitionResults] = useState(null);
  const [error, setError] = useState('');

  const API_URL = 'http://localhost:8000';

  useEffect(() => {
    let interval;
    if (trainDatasetId && modelStatus === 'pending') {
      interval = setInterval(async () => {
        try {
          const response = await axios.get(`${API_URL}/model-results/${trainDatasetId}`);
          if (response.data.status === 'completed') {
            setModelStatus('completed');
            setModelResults(response.data);
            clearInterval(interval);
          }
        } catch (err) {
          setError(`Failed to get model results: ${err.message}`);
          clearInterval(interval);
        }
      }, 3000);
    }
    return () => clearInterval(interval);
  }, [trainDatasetId, modelStatus]);

  useEffect(() => {
    let interval;
    if (predictDatasetId && predictionStatus === 'pending') {
      interval = setInterval(async () => {
        try {
          const response = await axios.get(`${API_URL}/prediction-results/${predictDatasetId}`);
          if (response.data.status === 'completed') {
            setPredictionStatus('completed');
            setPredicitionResults(response.data);
            clearInterval(interval);
          }
        } catch (err) {
          setError(`Failed to get prediction results: ${err.message}`);
          clearInterval(interval);
        }
      }, 3000);
    }
    return () => clearInterval(interval);
  }, [predictDatasetId, predictionStatus]);

  const handleTrainFileChange = (e) => {
    if (e.target.files[0]) {
      setTrainFile(e.target.files[0]);
    }
  };

  const handlePredictFileChange = (e) => {
    if (e.target.files[0]) {
      setPredictFile(e.target.files[0]);
    }
  };

  const uploadTrainData = async () => {
    if (!trainFile) {
      setError('Please select a training file');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('file', trainFile);

      const response = await axios.post(`${API_URL}/upload/train-data/`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setTrainDatasetId(response.data.dataset_id);
      await trainModel(response.data.dataset_id);
    } catch (err) {
      setError(`Failed to upload training data: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const trainModel = async (datasetId) => {
    try {
      setModelStatus('pending');
      await axios.post(`${API_URL}/train-model/${datasetId}`);
    } catch (err) {
      setError(`Failed to start model training: ${err.message}`);
      setModelStatus('failed');
    }
  };

  const uploadPredictData = async () => {
    if (!predictFile) {
      setError('Please select a prediction file');
      return;
    }

    if (!trainDatasetId) {
      setError('Please upload and train a model first');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('file', predictFile);

      const response = await axios.post(`${API_URL}/upload/predict-data/`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setPredictDatasetId(response.data.dataset_id);
      await generatePredictions(trainDatasetId, response.data.dataset_id);
    } catch (err) {
      setError(`Failed to upload prediction data: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const generatePredictions = async (modelId, datasetId) => {
    try {
      setPredictionStatus('pending');
      await axios.post(`${API_URL}/predict/${modelId}/${datasetId}`);
    } catch (err) {
      setError(`Failed to start prediction: ${err.message}`);
      setPredictionStatus('failed');
    }
  };

  // Render metrics table
  const renderMetrics = () => {
    if (!modelResults || !modelResults.metrics) return null;

    return (
      <div className="metrics-container">
        <h3>Model Performance Metrics</h3>
        <table className="metrics-table">
          <thead>
            <tr>
              <th>Metric</th>
              <th>Value</th>
            </tr>
          </thead>
          <tbody>
            {modelResults.metrics.map((metric, index) => (
              <tr key={index}>
                <td>{metric.Metric}</td>
                <td>{metric.Value.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  // Render computation time table
  const renderComputationTime = () => {
    if (!modelResults || !modelResults.computation_time) return null;

    return (
      <div className="metrics-container">
        <h3>Computation Time</h3>
        <table className="metrics-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Training (sec)</th>
              <th>Prediction (sec)</th>
              <th>Total (sec)</th>
            </tr>
          </thead>
          <tbody>
            {modelResults.computation_time.map((item, index) => (
              <tr key={index}>
                <td>{item.Model}</td>
                <td>{item['Training Time (seconds)'].toFixed(2)}</td>
                <td>{item['Prediction Time (seconds)'].toFixed(2)}</td>
                <td>{item['Total Time (seconds)'].toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  // Render training results chart
  const renderTrainingChart = () => {
    if (!modelResults || !modelResults.chart_data) return null;

    return (
      <div className="chart-container">
        <h3>Training Results</h3>
        <div className="responsive-chart">
          <ChartComponent 
            xData={modelResults.chart_data.x} 
            actualData={modelResults.chart_data.actual}
            predictedData={modelResults.chart_data.predicted}
          />
        </div>
      </div>
    );
  };

  // Render prediction results chart
  const renderPredictionChart = () => {
    if (!predictionResults || !predictionResults.chart_data) return null;

    return (
      <div className="chart-container">
        <h3>Prediction Results</h3>
        <div className="responsive-chart">
          <ChartComponent 
            xData={predictionResults.chart_data.x} 
            predictedData={predictionResults.chart_data.predicted}
            isPrediction={true}
          />
        </div>
      </div>
    );
  };

  return (
    <div className="app-container">
      <h1>Time Series Forecasting Tool</h1>
      
      {error && <div className="error-message">{error}</div>}
      
      <div className="file-upload-container">
        <div className="upload-section">
          <h2>1. Upload Training Data</h2>
          <input 
            type="file" 
            onChange={handleTrainFileChange} 
            accept=".csv"
            className="file-input" 
          />
          <button 
            onClick={uploadTrainData} 
            disabled={loading || !trainFile}
            className="upload-button"
          >
            {loading ? 'Uploading...' : 'Upload & Train Model'}
          </button>
          {modelStatus === 'pending' && (
            <div className="status-message">Training in progress...</div>
          )}
        </div>

        <div className="upload-section">
          <h2>2. Upload Prediction Data</h2>
          <input 
            type="file" 
            onChange={handlePredictFileChange} 
            accept=".csv"
            className="file-input" 
            disabled={!trainDatasetId || modelStatus !== 'completed'}
          />
          <button 
            onClick={uploadPredictData} 
            disabled={loading || !predictFile || !trainDatasetId || modelStatus !== 'completed'}
            className="upload-button"
          >
            {loading ? 'Uploading...' : 'Upload & Generate Predictions'}
          </button>
          {predictionStatus === 'pending' && (
            <div className="status-message">Generating predictions...</div>
          )}
        </div>
      </div>

      {modelStatus === 'completed' && (
        <div className="results-container">
          <h2>Training Results</h2>
          <div className="results-grid">
            {renderMetrics()}
            {renderComputationTime()}
          </div>
          {renderTrainingChart()}
        </div>
      )}

      {predictionStatus === 'completed' && (
        <div className="results-container">
          <h2>Prediction Results</h2>
          {renderPredictionChart()}
        </div>
      )}
    </div>
  );
}

// Chart component using SVG for direct rendering
function ChartComponent({ xData, actualData, predictedData, isPrediction = false }) {
  if (!xData || !predictedData) return <div>No data available for chart</div>;

  const margin = { top: 20, right: 30, bottom: 60, left: 60 };
  const width = 800 - margin.left - margin.right;
  const height = 400 - margin.top - margin.bottom;

  // Find min and max values for y-axis
  const predictedValues = [...predictedData];
  const allValues = isPrediction ? predictedValues : [...actualData, ...predictedValues];
  const minY = Math.min(...allValues);
  const maxY = Math.max(...allValues);
  
  // X-axis scale
  const xScale = (index) => (index / (xData.length - 1)) * width;
  
  // Y-axis scale with padding
  const yScale = (value) => {
    const padding = (maxY - minY) * 0.1;
    return height - ((value - minY + padding) / (maxY - minY + padding * 2)) * height;
  };

  // Generate line path data
  const generateLinePath = (data) => {
    return data.map((value, index) => (
      index === 0 
        ? `M ${xScale(index)},${yScale(value)}` 
        : `L ${xScale(index)},${yScale(value)}`
    )).join(' ');
  };

  // Generate x-axis labels (show every nth label to avoid crowding)
  const xLabels = () => {
    const step = Math.ceil(xData.length / 10); // Show approximately 10 labels
    return xData.map((date, index) => {
      if (index % step === 0 || index === xData.length - 1) {
        return (
          <text 
            key={`x-label-${index}`}
            x={xScale(index)} 
            y={height + 20} 
            textAnchor="middle" 
            fontSize="10"
          >
            {date}
          </text>
        );
      }
      return null;
    });
  };

  // Generate y-axis labels
  const yLabels = () => {
    const step = (maxY - minY) / 5;
    return Array.from({ length: 6 }, (_, i) => {
      const value = minY + step * i;
      return (
        <g key={`y-label-${i}`}>
          <line 
            x1="0" 
            y1={yScale(value)} 
            x2={width} 
            y2={yScale(value)} 
            stroke="#eee" 
            strokeWidth="1"
          />
          <text 
            x="-10" 
            y={yScale(value)} 
            textAnchor="end" 
            dominantBaseline="middle" 
            fontSize="10"
          >
            {Math.round(value)}
          </text>
        </g>
      );
    });
  };

  return (
    <svg 
      width="100%" 
      height="100%" 
      viewBox={`0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`}
      preserveAspectRatio="xMidYMid meet"
    >
      <g transform={`translate(${margin.left},${margin.top})`}>
        {/* Y-axis line */}
        <line x1="0" y1="0" x2="0" y2={height} stroke="#000" strokeWidth="1" />
        
        {/* X-axis line */}
        <line x1="0" y1={height} x2={width} y2={height} stroke="#000" strokeWidth="1" />
        
        {/* Grid lines and axis labels */}
        {yLabels()}
        {xLabels()}
        
        {/* Axis titles */}
        <text 
          x={width / 2} 
          y={height + 45} 
          textAnchor="middle" 
          fontSize="12" 
          fontWeight="bold"
        >
          Date
        </text>
        <text 
          transform={`rotate(-90) translate(${-height/2}, -40)`}
          textAnchor="middle" 
          fontSize="12" 
          fontWeight="bold"
        >
          Value
        </text>
        
        {/* Data lines */}
        {!isPrediction && (
          <path 
            d={generateLinePath(actualData)} 
            fill="none" 
            stroke="blue" 
            strokeWidth="2"
          />
        )}
        
        <path 
          d={generateLinePath(predictedData)} 
          fill="none" 
          stroke="red" 
          strokeWidth="2" 
          strokeDasharray={isPrediction ? "none" : "5,5"}
        />
        
        {/* Legend */}
        <g transform={`translate(${width - 150}, 20)`}>
          {!isPrediction && (
            <g>
              <line x1="0" y1="0" x2="20" y2="0" stroke="blue" strokeWidth="2" />
              <text x="25" y="5" fontSize="12">Actual</text>
            </g>
          )}
          <g transform="translate(0, 20)">
            <line 
              x1="0" 
              y1="0" 
              x2="20" 
              y2="0" 
              stroke="red" 
              strokeWidth="2"
              strokeDasharray={isPrediction ? "none" : "5,5"}
            />
            <text x="25" y="5" fontSize="12">Predicted</text>
          </g>
        </g>
        
        {/* Data points */}
        {!isPrediction && actualData.map((value, index) => (
          <circle 
            key={`actual-${index}`}
            cx={xScale(index)} 
            cy={yScale(value)} 
            r="3" 
            fill="blue"
          />
        ))}
        
        {predictedData.map((value, index) => (
          <circle 
            key={`predicted-${index}`}
            cx={xScale(index)} 
            cy={yScale(value)} 
            r="3" 
            fill="red"
          />
        ))}
      </g>
    </svg>
  );
}

export default App;