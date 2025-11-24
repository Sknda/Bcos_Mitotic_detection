import { useState } from 'react';
import '@/App.css';
import { BrowserRouter, Routes, Route, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { Upload, Activity, Zap, ArrowLeft, Image as ImageIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const UploadPage = () => {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleFileSelect = (file) => {
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onload = (e) => setPreviewUrl(e.target.result);
      reader.readAsDataURL(file);
    } else {
      toast.error('Please select a valid image file');
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    handleFileSelect(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      toast.error('Please select an image first');
      return;
    }

    setIsAnalyzing(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API}/analyze`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      
      toast.success('Analysis complete!');
      navigate('/results', { state: { results: response.data } });
    } catch (error) {
      console.error('Analysis error:', error);
      toast.error('Failed to analyze image. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="upload-page">
      <div className="upload-container">
        <div className="header-section">
          <div className="icon-wrapper">
            <Activity className="header-icon" />
          </div>
          <h1 className="main-title">ML Model Comparison Lab</h1>
          <p className="subtitle">Upload an image to compare predictions from two different models with explainability insights</p>
        </div>

        <Card className="upload-card" data-testid="upload-card">
          <CardContent className="upload-content">
            <div
              className={`drop-zone ${isDragging ? 'dragging' : ''} ${previewUrl ? 'has-preview' : ''}`}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              data-testid="drop-zone"
            >
              {previewUrl ? (
                <div className="preview-container">
                  <img src={previewUrl} alt="Preview" className="preview-image" data-testid="preview-image" />
                  <div className="preview-overlay">
                    <Button
                      variant="secondary"
                      onClick={() => {
                        setSelectedFile(null);
                        setPreviewUrl(null);
                      }}
                      data-testid="change-image-btn"
                    >
                      Change Image
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="drop-zone-content">
                  <Upload className="upload-icon" />
                  <p className="drop-text">Drag and drop an image here</p>
                  <p className="drop-subtext">or</p>
                  <Button
                    variant="outline"
                    onClick={() => document.getElementById('file-input').click()}
                    data-testid="browse-files-btn"
                  >
                    Browse Files
                  </Button>
                  <input
                    id="file-input"
                    type="file"
                    accept="image/*"
                    onChange={(e) => handleFileSelect(e.target.files[0])}
                    style={{ display: 'none' }}
                    data-testid="file-input"
                  />
                </div>
              )}
            </div>

            {selectedFile && (
              <div className="action-section">
                <Button
                  className="analyze-btn"
                  onClick={handleAnalyze}
                  disabled={isAnalyzing}
                  data-testid="analyze-btn"
                >
                  {isAnalyzing ? (
                    <>
                      <Zap className="btn-icon spinning" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Zap className="btn-icon" />
                      Analyze with Both Models
                    </>
                  )}
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        <div className="features-grid">
          <div className="feature-card">
            <Activity className="feature-icon" />
            <h3>Dual Model Analysis</h3>
            <p>Compare predictions from two different ML architectures</p>
          </div>
          <div className="feature-card">
            <ImageIcon className="feature-icon" />
            <h3>Visual Explainability</h3>
            <p>Grad-CAM heatmaps and segmentation masks for transparency</p>
          </div>
          <div className="feature-card">
            <Zap className="feature-icon" />
            <h3>Instant Results</h3>
            <p>Real-time inference with confidence scores and insights</p>
          </div>
        </div>
      </div>
    </div>
  );
};

const ResultsPage = () => {
  const navigate = useNavigate();
  const [results, setResults] = useState(null);

  useState(() => {
    const state = window.history.state?.usr;
    if (state?.results) {
      setResults(state.results);
    }
  }, []);

  if (!results) {
    return (
      <div className="results-page">
        <div className="empty-state">
          <p>No results available. Please upload an image first.</p>
          <Button onClick={() => navigate('/')} data-testid="back-to-upload-btn">
            <ArrowLeft className="btn-icon" />
            Back to Upload
          </Button>
        </div>
      </div>
    );
  }

  const { original_image, model_a, model_b } = results;

  return (
    <div className="results-page">
      <div className="results-container">
        <div className="results-header">
          <Button
            variant="ghost"
            onClick={() => navigate('/')}
            className="back-btn"
            data-testid="back-btn"
          >
            <ArrowLeft className="btn-icon" />
            New Analysis
          </Button>
          <h1 className="results-title">Model Comparison Results</h1>
        </div>

        <Card className="original-image-card" data-testid="original-image-card">
          <CardHeader>
            <CardTitle>Original Image</CardTitle>
          </CardHeader>
          <CardContent>
            <img
              src={`data:image/png;base64,${original_image}`}
              alt="Original"
              className="original-image"
              data-testid="original-image"
            />
          </CardContent>
        </Card>

        <div className="comparison-grid">
          <ModelCard
            title="Model A"
            prediction={model_a.prediction}
            confidence={model_a.confidence}
            explainabilityImages={model_a.explainability_images}
            summary={model_a.explainability_summary}
            accentColor="#3b82f6"
            testId="model-a"
          />
          <ModelCard
            title="Model B"
            prediction={model_b.prediction}
            confidence={model_b.confidence}
            explainabilityImages={model_b.explainability_images}
            summary={model_b.explainability_summary}
            accentColor="#10b981"
            testId="model-b"
          />
        </div>
      </div>
    </div>
  );
};

const ModelCard = ({ title, prediction, confidence, explainabilityImages, summary, accentColor, testId }) => {
  return (
    <Card className="model-card" data-testid={`${testId}-card`}>
      <CardHeader style={{ borderLeft: `4px solid ${accentColor}` }}>
        <CardTitle className="model-title">{title}</CardTitle>
        <CardDescription>Deep Learning Classification Model</CardDescription>
      </CardHeader>
      <CardContent className="model-content">
        <div className="prediction-section">
          <div className="prediction-label">Prediction</div>
          <div className="prediction-value" data-testid={`${testId}-prediction`}>{prediction}</div>
        </div>

        <div className="confidence-section">
          <div className="confidence-header">
            <span className="confidence-label">Confidence Score</span>
            <span className="confidence-value" data-testid={`${testId}-confidence`}>{(confidence * 100).toFixed(2)}%</span>
          </div>
          <Progress value={confidence * 100} className="confidence-progress" />
        </div>

        <div className="explainability-section">
          <h4 className="section-title">Explainability Visualizations</h4>
          <div className="explainability-images">
            {explainabilityImages.map((img, idx) => (
              <div key={idx} className="explain-image-wrapper">
                <img
                  src={`data:image/png;base64,${img}`}
                  alt={`Explainability ${idx + 1}`}
                  className="explain-image"
                  data-testid={`${testId}-explain-img-${idx}`}
                />
                <div className="explain-label">{idx === 0 ? 'Grad-CAM' : 'Segmentation'}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="summary-section">
          <h4 className="section-title">Analysis Summary</h4>
          <p className="summary-text" data-testid={`${testId}-summary`}>{summary}</p>
        </div>
      </CardContent>
    </Card>
  );
};

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<UploadPage />} />
          <Route path="/results" element={<ResultsPage />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;