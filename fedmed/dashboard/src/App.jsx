import { useEffect, useMemo, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import MaskViewer from "./MaskViewer.jsx";

const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8765";
const PALETTE = ["#60a5fa", "#34d399", "#f472b6", "#fbbf24"];
const tip = { contentStyle: { background: "#0e1728", border: "1px solid #24314d", borderRadius: 8 } };

function useFedStream() {
  const [events, setEvents] = useState([]);
  const [live, setLive] = useState(false);
  useEffect(() => {
    let ws, timer, closed = false;
    const connect = () => {
      ws = new WebSocket(WS_URL);
      ws.onopen = () => setLive(true);
      ws.onmessage = (m) => {
        const d = JSON.parse(m.data);
        setEvents((prev) => (d.type === "history" ? d.events : [...prev, d]));
      };
      ws.onclose = () => { setLive(false); if (!closed) timer = setTimeout(connect, 1500); };
    };
    connect();
    return () => { closed = true; clearTimeout(timer); ws && ws.close(); };
  }, []);
  return { events, live };
}

export default function App() {
  const { events, live } = useFedStream();
  const cfg = events.find((e) => e.type === "config");
  const fits = events.filter((e) => e.type === "fit");
  const evals = events.filter((e) => e.type === "eval");
  const lastFit = fits[fits.length - 1];
  const lastEval = evals[evals.length - 1];
  const baseline = cfg?.baseline?.dice;

  const nodeNames = useMemo(() => [...new Set(fits.flatMap((f) => f.nodes.map((n) => n.node)))], [fits.length]);
  const rows = useMemo(() => {
    const byRound = {};
    fits.forEach((f) => {
      byRound[f.round] = { round: f.round, ...(byRound[f.round] || {}) };
      f.nodes.forEach((n) => { byRound[f.round][n.node] = +n.train_loss.toFixed(4); });
    });
    evals.forEach((e) => {
      byRound[e.round] = { round: e.round, ...(byRound[e.round] || {}), val_loss: +e.loss.toFixed(4),
        dice: +e.dice.toFixed(4), TC: +e.tc.toFixed(4), WT: +e.wt.toFixed(4), ET: +e.et.toFixed(4) };
    });
    return Object.values(byRound).sort((a, b) => a.round - b.round);
  }, [fits.length, evals.length]);

  const uplink = lastFit ? lastFit.nodes.reduce((s, n) => s + n.upload_mb, 0) : 0;
  const gap = baseline && lastEval ? (lastEval.dice / baseline) * 100 : null;

  return (
    <div className="wrap">
      <header>
        <h1>FedMed<span>cross-silo federated learning · brain tumour segmentation</span></h1>
        <span className={`pill ${live ? "live" : "off"}`}>{live ? "● live" : "○ reconnecting…"}</span>
      </header>

      <div className="grid stats">
        <div className="card stat"><small>Round</small><b>{lastEval?.round ?? 0}{cfg ? ` / ${cfg.rounds}` : ""}</b></div>
        <div className="card stat"><small>Global Dice (mean of TC/WT/ET)</small><b>{lastEval ? lastEval.dice.toFixed(3) : "–"}</b></div>
        <div className="card stat"><small>Global val loss</small><b>{lastEval ? lastEval.loss.toFixed(3) : "–"}</b></div>
        <div className="card stat"><small>vs centralized baseline</small><b>{gap ? `${gap.toFixed(1)}%` : "–"}</b></div>
        <div className="card stat"><small>Uplink last round</small><b>{uplink ? `${uplink.toFixed(0)} MB` : "–"}</b></div>
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <h2>Privacy posture</h2>
        <div className="badges">
          <span className={`badge ${cfg?.tls ? "on" : "warn"}`}>gRPC {cfg?.tls ? "+ TLS" : "PLAINTEXT"}</span>
          <span className={`badge ${cfg?.he ? "on" : "warn"}`}>{cfg?.he ? "Homomorphic encryption (CKKS)" : "HE off"}</span>
          <span className={`badge ${lastFit?.dp_noise > 0 ? "on" : "warn"}`}>
            {lastFit?.dp_noise > 0 ? `DP σ=${lastFit.dp_noise}${lastFit.epsilon ? ` · ε≲${lastFit.epsilon.toFixed(0)} (loose)` : ""}` : "Differential privacy off"}
          </span>
          <span className="badge">Server sees: {lastFit?.payload ?? "–"}</span>
          <span className={`badge ${lastFit?.dropped ? "warn" : ""}`}>Nodes dropped last round: {lastFit?.dropped ?? 0}</span>
          <span className="badge">Model params: {cfg ? cfg.params.toLocaleString() : "–"}</span>
        </div>
      </div>

      <div className="grid two">
        <div className="card">
          <h2>Loss convergence</h2>
          {rows.length === 0 ? <div className="empty">Waiting for the first round…</div> : (
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={rows}>
                <CartesianGrid stroke="#24314d" strokeDasharray="3 3" />
                <XAxis dataKey="round" stroke="#8a9ab8" /><YAxis stroke="#8a9ab8" domain={[0, 1]} />
                <Tooltip {...tip} /><Legend />
                <Line type="monotone" dataKey="val_loss" name="global val loss" stroke="#fff" strokeWidth={3} dot />
                {nodeNames.map((n, i) => <Line key={n} type="monotone" dataKey={n} name={`${n} train`} stroke={PALETTE[i % 4]} strokeDasharray="5 4" dot={false} />)}
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
        <div className="card">
          <h2>Segmentation accuracy (Dice)</h2>
          {rows.length === 0 ? <div className="empty">Waiting for the first evaluation…</div> : (
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={rows}>
                <CartesianGrid stroke="#24314d" strokeDasharray="3 3" />
                <XAxis dataKey="round" stroke="#8a9ab8" /><YAxis stroke="#8a9ab8" domain={[0, 1]} />
                <Tooltip {...tip} /><Legend />
                {baseline && <ReferenceLine y={baseline} stroke="#fbbf24" strokeDasharray="6 4" label={{ value: "centralized baseline", fill: "#fbbf24", fontSize: 11, position: "insideBottomRight" }} />}
                <Line type="monotone" dataKey="dice" name="mean Dice" stroke="#fff" strokeWidth={3} dot />
                <Line type="monotone" dataKey="WT" stroke="#facc15" dot={false} />
                <Line type="monotone" dataKey="TC" stroke="#f87171" dot={false} />
                <Line type="monotone" dataKey="ET" stroke="#60a5fa" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      <div className="grid two">
        <div className="card">
          <h2>Hospital nodes — latest round</h2>
          <table>
            <thead><tr><th>Node</th><th>Samples</th><th>Train loss</th><th>Update ‖Δ‖</th><th>Upload</th><th>Val Dice</th></tr></thead>
            <tbody>
              {(lastFit?.nodes ?? []).map((n) => (
                <tr key={n.node}><td>{n.node}</td><td>{n.n}</td><td>{n.train_loss.toFixed(4)}</td>
                  <td>{n.update_norm.toFixed(2)}</td><td>{n.upload_mb.toFixed(1)} MB</td>
                  <td>{lastEval?.per_node?.[n.node]?.toFixed(3) ?? "–"}</td></tr>
              ))}
              {!lastFit && <tr><td colSpan="6" className="empty">No nodes have reported yet</td></tr>}
            </tbody>
          </table>
        </div>
        <div className="card"><h2>MRI tumour segmentation (final global model)</h2><MaskViewer /></div>
      </div>
    </div>
  );
}
