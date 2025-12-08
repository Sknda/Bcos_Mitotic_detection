import { useState } from 'react';
import '@/App.css';
import { BrowserRouter, Routes, Route, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { Upload, Activity, Zap, ArrowLeft, Image as ImageIcon, Target } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
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
            <Target className="header-icon" />
          </div>
          <h1 className="main-title">Mitotic Cell Detection Lab</h1>
          <p className="subtitle">Sequential AI pipeline: YOLOv11 detection → ResNet classification for precise mitotic cell analysis</p>
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
                  <p className="drop-text">Drag and drop cell microscopy image here</p>
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
                      Processing Pipeline...
                    </>
                  ) : (
                    <>
                      <Zap className="btn-icon" />
                      Run Detection Pipeline
                    </>
                  )}
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        <div className="pipeline-info">
          <h3 className="pipeline-title">Sequential Analysis Pipeline</h3>
          <div className="pipeline-steps">
            <div className="pipeline-step">
              <div className="step-number">1</div>
              <div className="step-content">
                <h4>YOLOv11 Detection</h4>
                <p>Identifies and localizes all cells in the image</p>
              </div>
            </div>
            <div className="pipeline-arrow">→</div>
            <div className="pipeline-step">
              <div className="step-number">2</div>
              <div className="step-content">
                <h4>ResNet Classification</h4>
                <p>Classifies each detected cell as mitotic or non-mitotic</p>
              </div>
            </div>
            <div className="pipeline-arrow">→</div>
            <div className="pipeline-step">
              <div className="step-number">3</div>
              <div className="step-content">
                <h4>Results & Visualization</h4>
                <p>Comprehensive analysis with explainability</p>
              </div>
            </div>
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

  const { original_image, yolo_results, resnet_results, processing_time } = results;

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
          <div>
            <h1 className="results-title">Detection Results</h1>
            <p className="results-subtitle" data-testid="processing-time">
              Processed in {processing_time?.toFixed(2)}s
            </p>
          </div>
        </div>

        <Card className="original-image-card" data-testid="original-image-card">
          <CardHeader>
            <CardTitle>Original Microscopy Image</CardTitle>
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
            title="Step 1: YOLOv11 Detection"
            modelName={yolo_results.model_name}
            predictions={yolo_results.predictions}
            totalDetections={yolo_results.total_detections}
            mitoticCount={yolo_results.mitotic_count}
            nonMitoticCount={yolo_results.non_mitotic_count}
            annotatedImage={yolo_results.annotated_image}
            explainabilityImages={yolo_results.explainability_images}
            summary={yolo_results.summary}
            accentColor="#3b82f6"
            testId="yolo"
          />
          <ModelCard
            title="Step 2: ResNet Classification"
            modelName={resnet_results.model_name}
            predictions={resnet_results.predictions}
            totalDetections={resnet_results.total_detections}
            mitoticCount={resnet_results.mitotic_count}
            nonMitoticCount={resnet_results.non_mitotic_count}
            annotatedImage={resnet_results.annotated_image}
            explainabilityImages={resnet_results.explainability_images}
            summary={resnet_results.summary}
            accentColor="#10b981"
            testId="resnet"
          />
        </div>
      </div>
    </div>
  );
};

const ModelCard = ({ 
  title, 
  modelName, 
  predictions, 
  totalDetections, 
  mitoticCount, 
  nonMitoticCount, 
  annotatedImage, 
  explainabilityImages, 
  summary, 
  accentColor, 
  testId 
}) => {
  return (
    <Card className="model-card" data-testid={`${testId}-card`}>
      <CardHeader style={{ borderLeft: `4px solid ${accentColor}` }}>
        <CardTitle className="model-title">{title}</CardTitle>
        <CardDescription>{modelName}</CardDescription>
      </CardHeader>
      <CardContent className="model-content">
        <div className="stats-grid">
          <div className="stat-box">
            <div className="stat-label">Total Cells</div>
            <div className="stat-value" data-testid={`${testId}-total`}>{totalDetections}</div>
          </div>
          <div className="stat-box mitotic">
            <div className="stat-label">Mitotic</div>
            <div className="stat-value" data-testid={`${testId}-mitotic`}>{mitoticCount}</div>
          </div>
          <div className="stat-box non-mitotic">
            <div className="stat-label">Non-Mitotic</div>
            <div className="stat-value" data-testid={`${testId}-non-mitotic`}>{nonMitoticCount}</div>
          </div>
        </div>

        <div className="annotated-section">
          <h4 className="section-title">Annotated Detection</h4>
          <div className="annotated-image-wrapper">
            <img
              src={`data:image/png;base64,${annotatedImage}`}
              alt="Annotated"
              className="annotated-image"
              data-testid={`${testId}-annotated`}
            />
          </div>
        </div>

        {explainabilityImages && explainabilityImages.length > 1 && (
          <div className="explainability-section">
            <h4 className="section-title">Explainability Heatmap</h4>
            <div className="heatmap-wrapper">
              <img
                src={`data:image/png;base64,${explainabilityImages[1]}`}
                alt="Heatmap"
                className="heatmap-image"
                data-testid={`${testId}-heatmap`}
              />
            </div>
          </div>
        )}

        <div className="summary-section">
          <h4 className="section-title">Analysis Summary</h4>
          <p className="summary-text" data-testid={`${testId}-summary`}>{summary}</p>
        </div>

        {predictions && predictions.length > 0 && (
          <div className="detections-list">
            <h4 className="section-title">Individual Detections</h4>
            <div className="detections-scroll">
              {predictions.slice(0, 5).map((pred, idx) => (
                <div key={idx} className="detection-item" data-testid={`${testId}-detection-${idx}`}>
                  <Badge variant={pred.label.includes('Mitotic') ? 'destructive' : 'default'}>
                    {pred.label}
                  </Badge>
                  <span className="detection-conf">{(pred.confidence * 100).toFixed(1)}%</span>
                  {pred.refinement && (
                    <span className="detection-refinement">{pred.refinement}</span>
                  )}
                </div>
              ))}
              {predictions.length > 5 && (
                <p className="more-detections">+ {predictions.length - 5} more detections</p>
              )}
            </div>
          </div>
        )}
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