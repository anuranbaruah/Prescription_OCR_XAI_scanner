import { useEffect, useRef, useState } from "react";
import { analyzePrescription, getHealth, getModelComparison } from "./api";
import Section from "./components/Section";
import Entities from "./components/Entities";
import Recommendations from "./components/Recommendations";
import Interactions from "./components/Interactions";
import XaiGallery from "./components/XaiGallery";
import BenchmarkTables from "./components/BenchmarkTables";
import PrescriptionResult from "./components/PrescriptionResult";
import Icon from "./components/Icon";

const CAP_LABELS = {
  vision: "Vision LLM",
  trocr: "TrOCR",
  easyocr: "EasyOCR",
  tesseract: "Tesseract",
  yolo: "YOLOv8",
  ner: "BioBERT NER",
  shap: "SHAP",
  lime: "LIME",
  torch: "PyTorch",
};

function CapabilityBar({ health }) {
  if (!health) return null;
  const caps = health.capabilities || {};
  return (
    <div className="chips">
      {Object.keys(CAP_LABELS).map((k) => (
        <span className="chip" key={k}>
          <span className={`dot ${caps[k] ? "on" : "off"}`} />
          {CAP_LABELS[k]}
        </span>
      ))}
    </div>
  );
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [benchmarks, setBenchmarks] = useState(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [drag, setDrag] = useState(false);
  const [tab, setTab] = useState("analysis");
  const inputRef = useRef();

  useEffect(() => {
    let timer;
    const poll = () => {
      getHealth()
        .then((h) => {
          setHealth(h);
          // keep polling until models finish warming up
          if (h.warmup && h.warmup !== "ready" && h.warmup !== "error") {
            timer = setTimeout(poll, 2500);
          }
        })
        .catch(() => setHealth({ device: "offline", capabilities: {} }));
    };
    poll();
    getModelComparison().then(setBenchmarks).catch(() => {});
    return () => clearTimeout(timer);
  }, []);

  const warming = health && health.warmup && health.warmup !== "ready" && health.warmup !== "error";

  function pickFile(f) {
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setReport(null);
    setError(null);
  }

  async function runAnalysis() {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const r = await analyzePrescription(file);
      setReport(r);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <header className="app-header">
        <div className="header-inner">
          <div className="brand-mark">
            <Icon name="activity" size={28} strokeWidth={2.2} />
          </div>
          <div>
            <h1>Explainable Prescription Analyzer</h1>
            <p>
              Handwritten Rx → OCR → BioBERT NER → generic recommendation +
              interaction check, with transparent XAI (Grad-CAM · SHAP · LIME)
            </p>
          </div>
          <span className="header-badge">
            {warming ? (
              <Icon name="clock" size={14} className="spin-slow" />
            ) : (
              <span className={`dot ${!health || health.device === "offline" ? "offline" : ""}`} />
            )}
            {!health
              ? "connecting…"
              : warming
              ? "loading models…"
              : `device · ${health.device}`}
          </span>
        </div>
      </header>

      <div className="container">
        <Section title="System capabilities" tag="live">
          <CapabilityBar health={health} />
          <p className="muted" style={{ fontSize: "0.8rem", marginBottom: 0 }}>
            Greyed-out capabilities aren't installed yet — the pipeline still runs and
            uses fallbacks for those stages.
          </p>
        </Section>

        <div className="tabs">
          <button
            className={`tab ${tab === "analysis" ? "active" : ""}`}
            onClick={() => setTab("analysis")}
          >
            Analyze prescription
          </button>
          <button
            className={`tab ${tab === "benchmarks" ? "active" : ""}`}
            onClick={() => setTab("benchmarks")}
          >
            Model comparison
          </button>
        </div>

        {tab === "benchmarks" && (
          <Section title="Deep learning model comparison" tag="synopsis §8">
            <BenchmarkTables benchmarks={benchmarks} />
          </Section>
        )}

        {tab === "analysis" && (
          <>
            <Section title="Upload prescription image">
              <div
                className={`dropzone ${drag ? "drag" : ""}`}
                role="button"
                tabIndex={0}
                aria-label="Upload a prescription image — drag and drop or activate to browse"
                onClick={() => inputRef.current?.click()}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    inputRef.current?.click();
                  }
                }}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDrag(true);
                }}
                onDragLeave={() => setDrag(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDrag(false);
                  pickFile(e.dataTransfer.files?.[0]);
                }}
              >
                {preview ? (
                  <img className="preview" src={preview} alt="Selected prescription preview" />
                ) : (
                  <div>
                    <div className="dz-icon">
                      <Icon name="upload" size={26} />
                    </div>
                    <p className="dz-title">
                      Drag &amp; drop or click to choose a prescription image
                    </p>
                    <p className="muted" style={{ margin: "4px 0 0" }}>
                      PNG / JPG — handwritten or printed
                    </p>
                  </div>
                )}
                <input
                  ref={inputRef}
                  type="file"
                  accept="image/*"
                  hidden
                  onChange={(e) => pickFile(e.target.files?.[0])}
                />
              </div>
              <div className="row" style={{ marginTop: 14 }}>
                <button className="btn" disabled={!file || loading} onClick={runAnalysis}>
                  {loading ? (
                    <>
                      <span className="spinner" /> &nbsp;Analyzing…
                    </>
                  ) : (
                    "Analyze"
                  )}
                </button>
                {file && (
                  <button
                    className="btn secondary"
                    disabled={loading}
                    onClick={() => {
                      setFile(null);
                      setPreview(null);
                      setReport(null);
                    }}
                  >
                    Clear
                  </button>
                )}
              </div>
              {warming && (
                <p className="hint" style={{ marginTop: 12 }}>
                  <Icon name="clock" size={15} className="spin-slow" />
                  Loading models into the GPU (one-time, ~30–40s). Analysis is
                  near-instant once this finishes — you can still upload now.
                </p>
              )}
              {error && (
                <p className="error" role="alert" style={{ marginTop: 14 }}>
                  <Icon name="alertTriangle" size={18} />
                  {error}
                </p>
              )}
            </Section>

            {loading && !report && (
              <Section title="Analyzing…">
                <div className="skeleton-wrap" aria-live="polite" aria-busy="true">
                  <div className="skeleton sk-line" style={{ width: "55%" }} />
                  <div className="skeleton sk-img" />
                  <div className="skeleton sk-line" style={{ width: "80%" }} />
                  <div className="skeleton sk-line" style={{ width: "65%" }} />
                  <span className="sr-only">Running OCR, NER, recommendation and XAI…</span>
                </div>
              </Section>
            )}

            {report && (
              <>
                <PrescriptionResult data={report.prescription} />
                <Section title="Pipeline summary">
                  <p style={{ marginTop: 0 }}>{report.message}</p>
                  <div className="chips">
                    {Object.entries(report.timings_ms || {}).map(([k, v]) => (
                      <span className="chip timing" key={k}>
                        {k}: {v} ms
                      </span>
                    ))}
                  </div>
                </Section>

                <div className="grid2">
                  <Section title="Detected text regions" tag="YOLOv8">
                    {report.preprocessed_image && (
                      <img
                        src={report.preprocessed_image}
                        alt="Prescription with detected text regions outlined"
                        loading="lazy"
                        style={{ width: "100%", borderRadius: 8, aspectRatio: "auto", display: "block" }}
                      />
                    )}
                  </Section>
                  <Section title="Extracted text" tag={report.ocr_results?.[0]?.engine || "OCR"}>
                    <div className="ocr-text">{report.extracted_text || "(no text)"}</div>
                    {report.ocr_results?.length > 1 && (
                      <div className="table-wrap" style={{ marginTop: 12 }}>
                        <table>
                          <thead>
                            <tr>
                              <th>Engine</th>
                              <th>Time (ms)</th>
                              <th>Conf.</th>
                            </tr>
                          </thead>
                          <tbody>
                            {report.ocr_results.map((o, i) => (
                              <tr key={i}>
                                <td>{o.engine}</td>
                                <td>{o.inference_ms ?? "—"}</td>
                                <td>{o.confidence != null ? o.confidence.toFixed(2) : "—"}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </Section>
                </div>

                <Section title="Identified medicines" tag="BioBERT NER">
                  <Entities entities={report.entities} />
                </Section>

                <Section title="Generic drug recommendations" tag="cost savings">
                  <Recommendations recommendations={report.recommendations} />
                </Section>

                <Section title="Drug interaction warnings" tag="DrugBank rules">
                  <Interactions interactions={report.interactions} />
                </Section>

                <Section title="Explainable AI" tag="Grad-CAM · SHAP · LIME">
                  <XaiGallery xai={report.xai} />
                </Section>
              </>
            )}
          </>
        )}
      </div>
    </>
  );
}
