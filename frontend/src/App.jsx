import { useEffect, useMemo, useState } from 'react'
import { api } from './api'

const classStyles = { plastic: 'blue', paper: 'amber', cardboard: 'orange', glass: 'emerald', metal: 'slate', trash: 'rose' }
const percent = value => `${(value * 100).toFixed(1)}%`

function Stat({ label, value, note }) { return <section className="stat"><p>{label}</p><strong>{value}</strong><small>{note}</small></section> }
function BarChart({ data }) { const maximum = Math.max(...data.map(item => item.count), 1); return <div className="bars">{data.length ? data.map(item => <div className="bar-row" key={item.label}><span>{item.label}</span><i><b style={{ width: `${item.count / maximum * 100}%` }} /></i><em>{item.count}</em></div>) : <p className="muted">Predictions will appear here after classification.</p>}</div> }

export default function App() {
  const [dashboard, setDashboard] = useState(null), [history, setHistory] = useState([]), [file, setFile] = useState(null), [preview, setPreview] = useState(null), [result, setResult] = useState(null), [explain, setExplain] = useState(true), [loading, setLoading] = useState(false), [error, setError] = useState('')
  const refresh = async () => { try { const [dash, items] = await Promise.all([api.dashboard(), api.history()]); setDashboard(dash); setHistory(items) } catch (err) { setError(`Cannot reach backend: ${err.message}`) } }
  useEffect(() => { refresh() }, [])
  useEffect(() => { if (!file) return setPreview(null); const url = URL.createObjectURL(file); setPreview(url); return () => URL.revokeObjectURL(url) }, [file])
  const submit = async event => { event.preventDefault(); if (!file) return setError('Choose a waste image first.'); setLoading(true); setError(''); setResult(null); try { setResult(await api.classify(file, explain)); await refresh() } catch (err) { setError(err.message) } finally { setLoading(false) } }
  const status = dashboard?.model
  const recent = useMemo(() => history.slice(0, 6), [history])
  return <main>
    <header><div><span className="eyebrow">COMPUTER VISION · DEEP LEARNING</span><h1>EcoSort <b>AI</b></h1></div><div className={`status ${status?.ready ? 'ready' : ''}`}><i />{status?.ready ? `Model ready · ${status.model_name}` : 'Model not trained yet'}</div></header>
    <p className="intro">Classify common waste with a trained MobileNetV2 or custom CNN, retain the audit trail, and use Grad-CAM to inspect each decision.</p>
    {error && <div className="alert error">{error}</div>}
    {!status?.ready && <div className="alert">Training is required before classifications are allowed. Use the commands in README, then restart the API.</div>}
    <section className="stats"><Stat label="Classifications" value={dashboard?.total_predictions ?? '—'} note="Stored locally in SQLite" /><Stat label="Average confidence" value={dashboard ? percent(dashboard.average_confidence) : '—'} note="Across saved predictions" /><Stat label="Supported classes" value="6" note="TrashNet: plastic, paper, cardboard, glass, metal, trash" /></section>
    <section className="grid">
      <form className="panel uploader" onSubmit={submit}><div className="panel-title"><div><span className="eyebrow">NEW ANALYSIS</span><h2>Classify a waste item</h2></div></div><label className="dropzone"><input type="file" accept="image/jpeg,image/png,image/webp" onChange={event => setFile(event.target.files?.[0] || null)} />{preview ? <img src={preview} alt="Selected waste item preview" /> : <><strong>Drop an image or browse</strong><small>JPEG, PNG or WebP · max 10 MB</small></>}</label>{file && <p className="filename">{file.name}</p>}<label className="check"><input type="checkbox" checked={explain} onChange={event => setExplain(event.target.checked)} /> Include Grad-CAM explanation</label><button disabled={loading || !status?.ready}>{loading ? 'Analyzing image…' : 'Classify waste'}</button></form>
      <section className="panel result"><span className="eyebrow">MODEL OUTPUT</span>{result ? <><div className="prediction"><span className={`tag ${classStyles[result.predicted_class] || ''}`}>{result.predicted_class}</span><strong>{percent(result.confidence)}</strong></div>{result.low_confidence && <div className="alert warning">{result.warning}</div>}<h3>Top predictions</h3>{result.top_predictions.map(item => <div className="score" key={item.label}><span>{item.label}</span><i><b style={{ width: percent(item.confidence) }} /></i><em>{percent(item.confidence)}</em></div>)}<div className="recommendation"><h3>Disposal recommendation</h3><p>{result.recommendation}</p></div>{result.gradcam_image && <><h3>Grad-CAM attention map</h3><img className="gradcam" src={result.gradcam_image} alt="Grad-CAM heatmap overlay" /></>}</> : <div className="empty">Upload an image to see a ranked prediction, recommendation, and optional Grad-CAM heatmap.</div>}</section>
    </section>
    <section className="grid lower"><section className="panel"><span className="eyebrow">LOCAL ANALYTICS</span><h2>Classification distribution</h2><BarChart data={dashboard?.by_class || []} /></section><section className="panel"><span className="eyebrow">AUDIT LOG</span><h2>Recent classifications</h2><div className="table">{recent.length ? recent.map(item => <div className="record" key={item.id}><span className={`tag ${classStyles[item.predicted_class] || ''}`}>{item.predicted_class}</span><small>{item.filename}</small><b>{percent(item.confidence)}</b></div>) : <p className="muted">No stored predictions yet.</p>}</div></section></section>
    <footer>Results are model estimates, not a substitute for local waste-management rules. The “trash” class is TrashNet’s general waste label—not organic waste.</footer>
  </main>
}
