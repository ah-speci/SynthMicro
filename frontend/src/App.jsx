import { useState } from "react";
import "./App.css";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (event) => {
    const file = event.target.files[0];

    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setError("Please select a valid image file.");
      return;
    }

    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError("");
  };

  const analyzeImage = async () => {
    if (!selectedFile) {
      setError("Please select an image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/predict",
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Analysis failed."
        );
      }

      setResult(data);
    } catch (err) {
      console.error(err);

      setError(
        "Unable to connect to the AI backend. Please make sure FastAPI is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const resetAnalysis = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError("");
  };

  const getConfidenceClass = (confidence) => {
    if (confidence >= 90) return "confidence-high";
    if (confidence >= 70) return "confidence-medium";
    return "confidence-low";
  };

  return (
    <div className="app">

      {/* ================= HEADER ================= */}

      <header className="header">

        <div className="header-inner">

          <div className="brand">

            <div className="brand-logo">
              <span>✦</span>
            </div>

            <div>
              <h1>SynthMicro</h1>
              <p>Intelligent Microscopic Cell Analysis</p>
            </div>

          </div>

          <div className="system-status">
            <span className="status-dot"></span>
            AI System Online
          </div>

        </div>

      </header>


      {/* ================= MAIN ================= */}

      <main className="main">

        {/* Hero */}

        <section className="hero">

          <div className="hero-badge">
            <span>AI</span>
            ResNet50 + Grad-CAM
          </div>

          <h2>
            Leukemia Cell
            <span> Classification</span>
          </h2>

          <p>
            Upload a microscopic blood-cell image and let the
            trained deep-learning model classify the cell while
            Grad-CAM provides a visual explanation of the prediction.
          </p>

        </section>


        {/* ================= UPLOAD ================= */}

        <section className="upload-card">

          <div className="section-heading">
            <div>
              <span className="step-number">01</span>
              <div>
                <h3>Upload Cell Image</h3>
                <p>Select a microscopic blood-cell image for analysis.</p>
              </div>
            </div>
          </div>


          <div className={`upload-zone ${preview ? "has-image" : ""}`}>

            {preview ? (

              <div className="selected-image-container">

                <img
                  src={preview}
                  alt="Selected blood cell"
                  className="selected-image"
                />

                <div className="image-overlay">
                  <span>Image selected</span>
                </div>

              </div>

            ) : (

              <div className="empty-upload">

                <div className="upload-icon">
                  ↑
                </div>

                <h3>Drop your image here</h3>

                <p>
                  or choose an image from your computer
                </p>

              </div>

            )}

            <label className="choose-button">

              <span>
                {preview ? "Choose Another Image" : "Choose Image"}
              </span>

              <input
                type="file"
                accept="image/png,image/jpeg,image/jpg"
                onChange={handleFileChange}
              />

            </label>

            <p className="supported-formats">
              Supported formats: JPG, JPEG, PNG
            </p>

          </div>


          {selectedFile && (

            <div className="file-info">

              <div className="file-icon">
                IMG
              </div>

              <div className="file-details">

                <strong>{selectedFile.name}</strong>

                <span>
                  {(selectedFile.size / 1024).toFixed(1)} KB
                </span>

              </div>

            </div>

          )}


          <div className="action-row">

            <button
              className="analyze-button"
              onClick={analyzeImage}
              disabled={!selectedFile || loading}
            >

              {loading ? (
                <>
                  <span className="spinner"></span>
                  Analyzing Cell...
                </>
              ) : (
                <>
                  Analyze Image
                  <span className="button-arrow">→</span>
                </>
              )}

            </button>

            {selectedFile && !loading && (
              <button
                className="reset-button"
                onClick={resetAnalysis}
              >
                Clear
              </button>
            )}

          </div>


          {error && (
            <div className="error-message">
              <span>!</span>
              {error}
            </div>
          )}

        </section>


        {/* ================= RESULTS ================= */}

        {result && (

          <section className="results">

            <div className="results-heading">

              <div>
                <span className="step-number">02</span>

                <div>
                  <h2>Analysis Result</h2>
                  <p>
                    AI classification and visual explanation
                  </p>
                </div>
              </div>

              <button
                className="new-analysis-button"
                onClick={resetAnalysis}
              >
                + New Analysis
              </button>

            </div>


            {/* Prediction summary */}

            <div className="prediction-card">

              <div className="prediction-content">

                <div className="prediction-icon">
                  ✓
                </div>

                <div>

                  <span className="result-label">
                    PREDICTED CELL TYPE
                  </span>

                  <h3>
                    {result.prediction}
                  </h3>

                  <p>
                    The model classified the uploaded image as{" "}
                    <strong>
                      {result.prediction}
                    </strong>.
                  </p>

                </div>

              </div>


              <div className="confidence-box">

                <span>CONFIDENCE</span>

                <strong>
                  {result.confidence.toFixed(2)}%
                </strong>

                <div className="confidence-track">

                  <div
                    className={`confidence-fill ${getConfidenceClass(
                      result.confidence
                    )}`}
                    style={{
                      width: `${result.confidence}%`,
                    }}
                  />

                </div>

              </div>

            </div>


            {/* Images */}

            <div className="visualization-grid">

              <div className="visual-card">

                <div className="visual-header">

                  <div>
                    <span className="visual-number">
                      A
                    </span>

                    <div>
                      <h3>Original Image</h3>
                      <p>Uploaded microscopy image</p>
                    </div>
                  </div>

                </div>

                <div className="visual-image-wrapper">

                  <img
                    src={preview}
                    alt="Original blood cell"
                  />

                </div>

              </div>


              <div className="visual-card">

                <div className="visual-header">

                  <div>
                    <span className="visual-number">
                      B
                    </span>

                    <div>
                      <h3>Grad-CAM Explanation</h3>
                      <p>Model attention visualization</p>
                    </div>
                  </div>

                </div>

                <div className="visual-image-wrapper">

                  <img
                    src={`data:image/png;base64,${result.gradcam}`}
                    alt="Grad-CAM visualization"
                  />

                </div>

              </div>

            </div>


            {/* XAI explanation */}

            <div className="xai-info">

              <div className="xai-icon">
                ✦
              </div>

              <div>

                <h3>
                  What does the Grad-CAM show?
                </h3>

                <p>
                  Grad-CAM highlights the regions of the image
                  that contributed most strongly to the model's
                  classification. Warmer regions indicate areas
                  with stronger influence on the prediction.
                </p>

              </div>

            </div>


            {/* Probabilities */}

            <div className="probabilities-card">

              <div className="probabilities-header">

                <div>
                  <h3>Class Probabilities</h3>
                  <p>
                    Confidence distribution across all cell classes
                  </p>
                </div>

              </div>


              <div className="probability-list">

                {Object.entries(result.probabilities).map(
                  ([name, probability]) => {

                    const isPrediction =
                      name === result.prediction;

                    return (

                      <div
                        className={`probability-row ${
                          isPrediction ? "predicted-row" : ""
                        }`}
                        key={name}
                      >

                        <div className="probability-label">

                          <div className="class-name">

                            {isPrediction && (
                              <span className="check-mark">
                                ✓
                              </span>
                            )}

                            <span>
                              {name.replace("_", " ")}
                            </span>

                          </div>

                          <strong>
                            {probability.toFixed(2)}%
                          </strong>

                        </div>


                        <div className="progress">

                          <div
                            className="progress-bar"
                            style={{
                              width: `${Math.max(
                                probability,
                                probability > 0 ? 0.5 : 0
                              )}%`,
                            }}
                          />

                        </div>

                      </div>

                    );
                  }
                )}

              </div>

            </div>


            {/* Disclaimer */}

            <div className="disclaimer">

              <strong>Research / Educational Use</strong>

              <span>
                This application is a machine-learning demonstration
                and is not intended to provide a clinical diagnosis.
              </span>

            </div>

          </section>

        )}

      </main>


      {/* ================= FOOTER ================= */}

      <footer className="footer">

        <div>
          <strong>SynthMicro</strong>
          <span>AI-assisted microscopic cell classification</span>
        </div>

        <span>
          ResNet50 • Grad-CAM • FastAPI
        </span>

      </footer>

    </div>
  );
}

export default App;