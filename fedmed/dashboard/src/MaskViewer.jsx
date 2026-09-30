import { useEffect, useRef, useState } from "react";

// class ids exported by scripts/export_sample.py: 1=whole tumour, 2=tumour core, 3=enhancing
const COLORS = { 1: [250, 204, 21], 2: [248, 113, 113], 3: [96, 165, 250] };

function Slice({ data, mask, title }) {
  const ref = useRef(null);
  useEffect(() => {
    const [h, w] = data.size;
    const c = ref.current;
    c.width = w; c.height = h;
    const ctx = c.getContext("2d");
    const img = ctx.createImageData(w, h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const g = data.image[y][x] * 255, k = mask ? mask[y][x] : 0, i = (y * w + x) * 4;
      const col = COLORS[k];
      img.data[i] = col ? g * 0.45 + col[0] * 0.55 : g;
      img.data[i + 1] = col ? g * 0.45 + col[1] * 0.55 : g;
      img.data[i + 2] = col ? g * 0.45 + col[2] * 0.55 : g;
      img.data[i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [data, mask]);
  return <figure><canvas ref={ref} /><figcaption>{title}</figcaption></figure>;
}

export default function MaskViewer() {
  const [data, setData] = useState(null);
  useEffect(() => { fetch("/sample.json").then(r => r.ok ? r.json() : null).then(setData).catch(() => {}); }, []);
  if (!data) return <div className="empty">No sample yet. After training run <code>python scripts/export_sample.py</code></div>;
  return (
    <>
      <div className="masks">
        <Slice data={data} mask={null} title="MRI slice (FLAIR)" />
        <Slice data={data} mask={data.truth} title="Ground truth" />
        <Slice data={data} mask={data.pred} title="Federated model prediction" />
      </div>
      <div className="legend">
        <span><i style={{ background: "rgb(250,204,21)" }} />whole tumour</span>
        <span><i style={{ background: "rgb(248,113,113)" }} />tumour core</span>
        <span><i style={{ background: "rgb(96,165,250)" }} />enhancing</span>
      </div>
    </>
  );
}
