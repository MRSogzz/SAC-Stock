import { useState, useEffect, useRef } from "react";

const API = "http://localhost:8000";

const PERIOD_OPTIONS = [
  { value: "1y", label: "1 年" },
  { value: "2y", label: "2 年" },
  { value: "3y", label: "3 年" },
  { value: "5y", label: "5 年" },
  { value: "6y", label: "6 年（建議）" },
];

const REGIME_LABEL = { bull: "牛市", bear: "熊市", sideways: "盤整" };
const REGIME_COLOR = { bull: "var(--color-text-success)", bear: "var(--color-text-danger)", sideways: "var(--color-text-secondary)" };

function MetricCard({ label, value, sub, color }) {
  return (
    <div style={{ background:"var(--color-background-secondary)", borderRadius:"var(--border-radius-md)", padding:"14px 16px" }}>
      <div style={{ fontSize:12, color:"var(--color-text-secondary)", marginBottom:4 }}>{label}</div>
      <div style={{ fontSize:20, fontWeight:500, color:color||"var(--color-text-primary)" }}>{value}</div>
      {sub && <div style={{ fontSize:11, color:"var(--color-text-tertiary)", marginTop:2 }}>{sub}</div>}
    </div>
  );
}

function PortfolioChart({ portfolio, bh, dates }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current || !portfolio?.length) return;
    const canvas = ref.current;
    const ctx = canvas.getContext("2d");
    const W = canvas.offsetWidth, H = 200;
    canvas.width = W*2; canvas.height = H*2; ctx.scale(2,2);
    ctx.clearRect(0,0,W,H);
    const all = [...portfolio,...(bh||[])];
    const min = Math.min(...all), max = Math.max(...all), range = max-min||1;
    const pad = {l:48,r:16,t:16,b:32};
    const iw = W-pad.l-pad.r, ih = H-pad.t-pad.b;
    const x = i => pad.l+(i/(portfolio.length-1))*iw;
    const y = v => pad.t+ih-((v-min)/range)*ih;
    ctx.strokeStyle="rgba(128,128,128,0.12)"; ctx.lineWidth=0.5;
    [0,0.25,0.5,0.75,1].forEach(t => {
      const yy=pad.t+ih-t*ih; ctx.beginPath(); ctx.moveTo(pad.l,yy); ctx.lineTo(pad.l+iw,yy); ctx.stroke();
      const val=(min+t*range-1)*100;
      ctx.fillStyle="rgba(128,128,128,0.5)"; ctx.font="10px sans-serif";
      ctx.fillText((val>=0?"+":"")+val.toFixed(0)+"%",2,yy+3);
    });
    if (dates?.length) {
      ctx.fillStyle="rgba(128,128,128,0.5)"; ctx.font="9px sans-serif";
      [0,Math.floor(dates.length/2),dates.length-1].forEach(i => {
        ctx.fillText(dates[Math.min(i,dates.length-1)]?.slice(0,7)||"",x(i)-16,H-4);
      });
    }
    if (bh?.length) {
      ctx.beginPath(); ctx.moveTo(x(0),y(bh[0]));
      bh.forEach((v,i)=>ctx.lineTo(x(i),y(v)));
      ctx.strokeStyle="rgba(128,128,128,0.4)"; ctx.lineWidth=1;
      ctx.setLineDash([4,4]); ctx.stroke(); ctx.setLineDash([]);
    }
    ctx.beginPath(); ctx.moveTo(x(0),y(portfolio[0]));
    portfolio.forEach((v,i)=>ctx.lineTo(x(i),y(v)));
    const c = portfolio[portfolio.length-1]>=portfolio[0]?"#1D9E75":"#E24B4A";
    ctx.strokeStyle=c; ctx.lineWidth=2; ctx.stroke();
    const g=ctx.createLinearGradient(0,pad.t,0,pad.t+ih);
    g.addColorStop(0,c+"30"); g.addColorStop(1,c+"00");
    ctx.lineTo(x(portfolio.length-1),pad.t+ih); ctx.lineTo(x(0),pad.t+ih);
    ctx.closePath(); ctx.fillStyle=g; ctx.fill();
  },[portfolio,bh,dates]);
  return (
    <div>
      <canvas ref={ref} style={{width:"100%",height:200,display:"block"}}/>
      <div style={{display:"flex",gap:16,marginTop:8,fontSize:12,color:"var(--color-text-secondary)"}}>
        <span style={{display:"flex",alignItems:"center",gap:4}}><span style={{width:20,height:2,background:"#1D9E75",display:"inline-block"}}/>AI 組合</span>
        <span style={{display:"flex",alignItems:"center",gap:4}}><span style={{width:20,height:2,borderTop:"2px dashed rgba(128,128,128,0.4)",display:"inline-block"}}/>等權重持有</span>
      </div>
    </div>
  );
}

function PositionBar({ name, pct, sector }) {
  const color = pct > 25 ? "#1D9E75" : pct > 10 ? "#378ADD" : "var(--color-text-tertiary)";
  return (
    <div style={{marginBottom:10}}>
      <div style={{display:"flex",justifyContent:"space-between",fontSize:13,marginBottom:4}}>
        <span>{name} <span style={{fontSize:11,color:"var(--color-text-tertiary)"}}>{sector}</span></span>
        <span style={{fontWeight:500,color,fontFamily:"var(--font-mono)"}}>{pct}%</span>
      </div>
      <div style={{height:6,background:"var(--color-background-secondary)",borderRadius:3}}>
        <div style={{height:"100%",width:`${Math.min(pct/40*100,100)}%`,background:color,borderRadius:3,transition:"width 0.8s ease"}}/>
      </div>
    </div>
  );
}

function TradeLog({ trades }) {
  if (!trades?.length) return <div style={{fontSize:13,color:"var(--color-text-tertiary)",padding:"20px 0",textAlign:"center"}}>尚無交易紀錄</div>;
  return (
    <div style={{overflowX:"auto"}}>
      <table style={{width:"100%",borderCollapse:"collapse",fontSize:13}}>
        <thead>
          <tr style={{borderBottom:"0.5px solid var(--color-border-tertiary)"}}>
            {["日期","股票","操作","價格","股數","金額","手續費","損益","倉位"].map(h=>(
              <th key={h} style={{padding:"8px 10px",textAlign:"left",fontWeight:500,fontSize:12,color:"var(--color-text-secondary)"}}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {trades.map((t,i)=>{
            const isBuy = t.action?.includes("買入");
            return (
              <tr key={i} style={{borderBottom:"0.5px solid var(--color-border-tertiary)",background:i%2===0?"transparent":"var(--color-background-secondary)"}}>
                <td style={{padding:"8px 10px",color:"var(--color-text-secondary)",fontFamily:"var(--font-mono)",fontSize:12}}>{t.date}</td>
                <td style={{padding:"8px 10px",fontWeight:500}}>{t.stock_name}<span style={{fontSize:11,color:"var(--color-text-tertiary)",marginLeft:4}}>{t.stock}</span></td>
                <td style={{padding:"8px 10px"}}>
                  <span style={{display:"inline-block",padding:"2px 8px",borderRadius:4,fontSize:12,fontWeight:500,
                    background:isBuy?"rgba(29,158,117,0.1)":"rgba(226,75,74,0.1)",
                    color:isBuy?"#0f6e56":"#993c1d"}}>{t.action}</span>
                </td>
                <td style={{padding:"8px 10px",fontFamily:"var(--font-mono)"}}>${t.price?.toLocaleString()}</td>
                <td style={{padding:"8px 10px",fontFamily:"var(--font-mono)",color:"var(--color-text-secondary)"}}>{t.shares?.toLocaleString(undefined,{maximumFractionDigits:2})}</td>
                <td style={{padding:"8px 10px",fontFamily:"var(--font-mono)"}}>${t.amount?.toLocaleString()}</td>
                <td style={{padding:"8px 10px",fontFamily:"var(--font-mono)",color:"var(--color-text-tertiary)"}}>{t.fee!=null?"$"+t.fee?.toLocaleString():"—"}</td>
                <td style={{padding:"8px 10px",fontFamily:"var(--font-mono)",fontWeight:500,
                  color:t.profit==null?"var(--color-text-tertiary)":t.profit>=0?"var(--color-text-success)":"var(--color-text-danger)"}}>
                  {t.profit==null?"—":(t.profit>=0?"+":"")+"$"+t.profit?.toLocaleString()}
                </td>
                <td style={{padding:"8px 10px",fontFamily:"var(--font-mono)",color:"var(--color-text-secondary)"}}>{t.position!=null?(t.position*100).toFixed(0)+"%":"—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function PredictionPanel({ pred }) {
  if (!pred) return null;
  const colorMap = {"買入":"var(--color-text-success)","持有":"var(--color-text-info)","觀望":"var(--color-text-tertiary)"};
  const bgMap    = {"買入":"rgba(29,158,117,0.08)","持有":"rgba(55,138,221,0.08)","觀望":"transparent"};
  return (
    <div>
      <div style={{display:"flex",alignItems:"center",gap:12,marginBottom:16,flexWrap:"wrap"}}>
        <div style={{fontSize:12,color:"var(--color-text-secondary)"}}>截至 {pred.as_of_date}</div>
        {pred.current_regime && (
          <span style={{fontSize:12,padding:"2px 8px",borderRadius:4,
            background:"var(--color-background-secondary)",
            color:REGIME_COLOR[pred.current_regime]||"var(--color-text-secondary)"}}>
            {REGIME_LABEL[pred.current_regime]||pred.current_regime}
            {pred.selected_window && ` · 窗口 ${pred.selected_window}`}
          </span>
        )}
        <div style={{marginLeft:"auto",fontSize:12,color:"var(--color-text-secondary)"}}>
          建議現金：<span style={{fontWeight:500,color:"var(--color-text-primary)"}}>{pred.cash_pct}%</span>
        </div>
      </div>
      <div style={{display:"grid",gridTemplateColumns:"repeat(2,1fr)",gap:8}}>
        {pred.recommendations?.map(r=>(
          <div key={r.stock_id} style={{
            display:"flex",alignItems:"center",justifyContent:"space-between",
            padding:"10px 14px",borderRadius:"var(--border-radius-md)",
            background:bgMap[r.action],
            border:`0.5px solid ${r.action==="觀望"?"var(--color-border-tertiary)":"transparent"}`,
          }}>
            <div>
              <span style={{fontWeight:500,fontSize:13}}>{r.stock_name}</span>
              <span style={{fontSize:11,color:"var(--color-text-tertiary)",marginLeft:6}}>{r.stock_id}</span>
            </div>
            <div style={{textAlign:"right"}}>
              <div style={{fontSize:13,fontWeight:500,color:colorMap[r.action]}}>{r.action}</div>
              <div style={{fontSize:12,fontFamily:"var(--font-mono)",color:"var(--color-text-secondary)"}}>目標 {r.target_pct}%</div>
            </div>
          </div>
        ))}
      </div>
      <div style={{fontSize:12,color:"var(--color-text-tertiary)",marginTop:12}}>⚠ 本預測僅供參考，不構成投資建議</div>
    </div>
  );
}

function WalkForwardWindowCard({ w }) {
  const ret = w.val_return;
  return (
    <div style={{background:"var(--color-background-secondary)",borderRadius:"var(--border-radius-md)",padding:"14px 16px"}}>
      <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:8}}>
        <span style={{fontSize:13,fontWeight:500}}>窗口 {w.window}</span>
        <span style={{fontSize:12,padding:"2px 8px",borderRadius:4,
          background:"var(--color-background-primary)",
          color:REGIME_COLOR[w.regime]||"var(--color-text-secondary)"}}>
          {REGIME_LABEL[w.regime]||w.regime||"—"}
        </span>
      </div>
      <div style={{fontSize:11,color:"var(--color-text-tertiary)",marginBottom:8}}>
        訓練 {w.train_start?.slice(0,7)}~{w.train_end?.slice(0,7)}<br/>
        驗證 {w.val_start?.slice(0,7)}~{w.val_end?.slice(0,7)}
      </div>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:8}}>
        <div>
          <div style={{fontSize:11,color:"var(--color-text-tertiary)"}}>訓練報酬</div>
          <div style={{fontSize:14,fontWeight:500,color:"var(--color-text-secondary)"}}>
            {w.train_return!=null?(w.train_return>=0?"+":"")+w.train_return+"%":"—"}
          </div>
        </div>
        <div>
          <div style={{fontSize:11,color:"var(--color-text-tertiary)"}}>驗證報酬</div>
          <div style={{fontSize:14,fontWeight:500,
            color:ret==null?"var(--color-text-secondary)":ret>=0?"var(--color-text-success)":"var(--color-text-danger)"}}>
            {ret!=null?(ret>=0?"+":"")+ret+"%":"—"}
          </div>
        </div>
      </div>
      {w.win_rate!=null&&(
        <div style={{fontSize:11,color:"var(--color-text-tertiary)",marginTop:6}}>
          勝率 {w.win_rate}%
        </div>
      )}
    </div>
  );
}

const RUN_DESC = {
  A: "Composite Reward",
  B: "LogitDelta + Composite",
  C: "Linear Downside",
  D: "LogitDelta + Linear",
};

function RunComparisonTable({ runs }) {
  // runs = { A: {meta, windows:[]}, B: {...}, C: {...}, D: {...} }
  if (!runs) return null;
  const runIds = ["A","B","C","D"];
  const windows = [1,2,3];

  const retColor = (v) =>
    v == null ? "var(--color-text-tertiary)"
    : v >= 0  ? "var(--color-text-success)"
    :            "var(--color-text-danger)";
  const fmt = (v) => v != null ? `${v>=0?"+":""}${v}%` : "—";

  return (
    <div style={{overflowX:"auto"}}>
      <table style={{width:"100%",borderCollapse:"collapse",fontSize:12}}>
        <thead>
          <tr style={{borderBottom:"0.5px solid var(--color-border-tertiary)"}}>
            <th style={{padding:"8px 12px",textAlign:"left",fontWeight:500,color:"var(--color-text-secondary)",minWidth:80}}>窗口</th>
            {runIds.map(id=>(
              <th key={id} colSpan={2} style={{padding:"8px 12px",textAlign:"center",fontWeight:500,
                borderLeft:"0.5px solid var(--color-border-tertiary)"}}>
                <span style={{color:"var(--color-text-primary)"}}>Run {id}</span>
                <div style={{fontSize:10,fontWeight:400,color:"var(--color-text-tertiary)",marginTop:2}}>{RUN_DESC[id]}</div>
              </th>
            ))}
          </tr>
          <tr style={{borderBottom:"0.5px solid var(--color-border-tertiary)",background:"var(--color-background-secondary)"}}>
            <th style={{padding:"4px 12px",textAlign:"left",color:"var(--color-text-tertiary)",fontWeight:400}}></th>
            {runIds.map(id=>(
              [
                <th key={id+"t"} style={{padding:"4px 8px",textAlign:"center",color:"var(--color-text-tertiary)",fontWeight:400,
                  borderLeft:"0.5px solid var(--color-border-tertiary)"}}>訓練</th>,
                <th key={id+"v"} style={{padding:"4px 8px",textAlign:"center",color:"var(--color-text-tertiary)",fontWeight:400}}>驗證</th>
              ]
            ))}
          </tr>
        </thead>
        <tbody>
          {windows.map((w,wi)=>{
            return (
              <tr key={w} style={{borderBottom:"0.5px solid var(--color-border-tertiary)",
                background:wi%2===0?"transparent":"var(--color-background-secondary)"}}>
                <td style={{padding:"10px 12px",fontWeight:500}}>
                  窗口 {w}
                  {/* 取第一個有資料的 run 的日期 */}
                  {runIds.map(id=>runs[id]?.windows?.find(x=>x.window===w)).find(Boolean) && (
                    <div style={{fontSize:10,color:"var(--color-text-tertiary)",marginTop:2,fontWeight:400}}>
                      {(()=>{
                        const d = runIds.map(id=>runs[id]?.windows?.find(x=>x.window===w)).find(Boolean);
                        return `${d?.val_start?.slice(0,7)||""}~${d?.val_end?.slice(0,7)||""}`;
                      })()}
                    </div>
                  )}
                </td>
                {runIds.map(id=>{
                  const wd = runs[id]?.windows?.find(x=>x.window===w);
                  return [
                    <td key={id+"t"} style={{padding:"10px 8px",textAlign:"center",
                      fontFamily:"var(--font-mono)",
                      borderLeft:"0.5px solid var(--color-border-tertiary)",
                      color:"var(--color-text-secondary)"}}>
                      {fmt(wd?.train_return)}
                    </td>,
                    <td key={id+"v"} style={{padding:"10px 8px",textAlign:"center",
                      fontFamily:"var(--font-mono)",fontWeight:500,
                      color:retColor(wd?.val_return)}}>
                      {fmt(wd?.val_return)}
                    </td>
                  ];
                })}
              </tr>
            );
          })}
          {/* 平均行 */}
          <tr style={{borderTop:"1px solid var(--color-border-tertiary)",background:"var(--color-background-secondary)"}}>
            <td style={{padding:"10px 12px",fontWeight:500,fontSize:11,color:"var(--color-text-secondary)"}}>平均驗證</td>
            {runIds.map(id=>{
              const avg = runs[id]?.meta?.avg_val_return;
              return [
                <td key={id+"t"} style={{borderLeft:"0.5px solid var(--color-border-tertiary)"}}></td>,
                <td key={id+"v"} style={{padding:"10px 8px",textAlign:"center",
                  fontFamily:"var(--font-mono)",fontWeight:600,
                  color:retColor(avg)}}>
                  {fmt(avg)}
                </td>
              ];
            })}
          </tr>
        </tbody>
      </table>
    </div>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState("standard");   // "standard" | "walkforward"
  const [period, setPeriod]       = useState("6y");
  const [episodes, setEpisodes]   = useState(80);
  const [capital, setCapital]     = useState(1000000);
  const [valDays, setValDays]     = useState(250);
  const [valCapital, setValCapital] = useState(100000);

  // Standard training state
  const [status, setStatus]       = useState("idle");
  const [result, setResult]       = useState(null);
  const [error, setError]         = useState(null);
  const [elapsed, setElapsed]     = useState(0);
  const [progress, setProgress]   = useState([]);
  const [currentEp, setCurrentEp] = useState(0);
  const [prediction, setPrediction] = useState(null);
  const [predLoading, setPredLoading] = useState(false);
  const [valResult, setValResult] = useState(null);
  const [valLoading, setValLoading] = useState(false);

  // Walk-forward state
  const [wfEpisodes, setWfEpisodes]   = useState(600);
  const [wfStatus, setWfStatus]       = useState("idle");
  const [wfError, setWfError]         = useState(null);
  const [wfElapsed, setWfElapsed]     = useState(0);
  const [wfResult, setWfResult]       = useState(null);
  const [wfServerStatus, setWfServerStatus] = useState(null);
  const [wfPrediction, setWfPrediction] = useState(null);
  const [wfPredLoading, setWfPredLoading] = useState(false);

  const [savedModels, setSavedModels] = useState([]);
  const [scheduler, setScheduler]     = useState(null);
  const [history, setHistory]         = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const pollRef  = useRef(null);
  const timerRef = useRef(null);
  const wfPollRef  = useRef(null);
  const wfTimerRef = useRef(null);

  const fetchScheduler = async () => {
    try { setScheduler(await fetch(`${API}/scheduler/status`).then(r=>r.json())); } catch(e) {}
  };
  const fetchHistory = async () => {
    try {
      const d = await fetch(`${API}/scheduler/history?limit=50`).then(r=>r.json());
      setHistory(d.records||[]);
    } catch(e) {}
  };
  const runNow = async () => {
    try { await fetch(`${API}/scheduler/run-now`,{method:"POST"}); setTimeout(()=>{fetchHistory();fetchScheduler();},2000); }
    catch(e) { alert("執行失敗："+e.message); }
  };
  const fetchModels = async () => {
    try {
      const d = await fetch(`${API}/models`).then(r=>r.json());
      setSavedModels(d.models||[]);
    } catch(e) {}
  };
  const fetchWfStatus = async () => {
    try { setWfServerStatus(await fetch(`${API}/walkforward_status`).then(r=>r.json())); } catch(e) {}
  };

  useEffect(() => { fetchModels(); fetchScheduler(); fetchHistory(); fetchWfStatus(); }, []);
  useEffect(() => () => { clearInterval(pollRef.current); clearInterval(timerRef.current);
                          clearInterval(wfPollRef.current); clearInterval(wfTimerRef.current); }, []);

  const fetchPrediction = async (p) => {
    setPredLoading(true); setPrediction(null);
    try {
      const res = await fetch(`${API}/predict/${p||period}`);
      if (!res.ok) { const e=await res.json(); throw new Error(e.detail); }
      setPrediction(await res.json());
    } catch(e) { alert("預測失敗："+e.message); }
    finally { setPredLoading(false); }
  };

  const fetchWfPrediction = async () => {
    setWfPredLoading(true); setWfPrediction(null);
    try {
      const res = await fetch(`${API}/predict_walkforward/${period}`);
      if (!res.ok) { const e=await res.json(); throw new Error(e.detail); }
      setWfPrediction(await res.json());
    } catch(e) { alert("Walk-forward 預測失敗："+e.message); }
    finally { setWfPredLoading(false); }
  };

  const startTraining = async () => {
    setStatus("starting"); setError(null); setResult(null);
    setElapsed(0); setProgress([]); setCurrentEp(0);
    try {
      const res = await fetch(`${API}/train`,{method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({period,episodes,initial_capital:capital,val_days:valDays})});
      const {job_id} = await res.json();
      setStatus("running");
      timerRef.current = setInterval(()=>setElapsed(e=>e+1),1000);
      pollRef.current  = setInterval(async()=>{
        const s = await fetch(`${API}/status/${job_id}`).then(r=>r.json());
        if (s.progress) setProgress(s.progress);
        if (s.current_ep) setCurrentEp(s.current_ep);
        if (s.status==="done") {
          clearInterval(pollRef.current); clearInterval(timerRef.current);
          const r = await fetch(`${API}/result/${job_id}`).then(r=>r.json());
          setResult(r); setStatus("done"); fetchModels(); fetchPrediction(period);
        } else if (s.status==="error") {
          clearInterval(pollRef.current); clearInterval(timerRef.current);
          setError(s.error); setStatus("error");
        }
      },2000);
    } catch(e) { setError("無法連接後端"); setStatus("error"); }
  };

  const startWalkForward = async () => {
    setWfStatus("starting"); setWfError(null); setWfResult(null);
    setWfElapsed(0);
    try {
      const res = await fetch(`${API}/train_walkforward`,{method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({period,episodes:wfEpisodes,initial_capital:capital})});
      const {job_id} = await res.json();
      setWfStatus("running");
      wfTimerRef.current = setInterval(()=>setWfElapsed(e=>e+1),1000);
      wfPollRef.current  = setInterval(async()=>{
        const s = await fetch(`${API}/status/${job_id}`).then(r=>r.json());
        if (s.status==="done") {
          clearInterval(wfPollRef.current); clearInterval(wfTimerRef.current);
          const r = await fetch(`${API}/result/${job_id}`).then(r=>r.json());
          setWfResult(r); setWfStatus("done");
          fetchWfStatus(); fetchWfPrediction();
        } else if (s.status==="error") {
          clearInterval(wfPollRef.current); clearInterval(wfTimerRef.current);
          setWfError(s.error); setWfStatus("error");
        }
      },3000);
    } catch(e) { setWfError("無法連接後端"); setWfStatus("error"); }
  };

  const startValidation = async () => {
    setValLoading(true); setValResult(null);
    try {
      const res = await fetch(`${API}/validate`,{method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({period,val_days:valDays,initial_capital:valCapital})});
      const {job_id} = await res.json();
      const poll = setInterval(async()=>{
        const s = await fetch(`${API}/status/${job_id}`).then(r=>r.json());
        if (s.status==="done") { clearInterval(poll); setValResult(await fetch(`${API}/result/${job_id}`).then(r=>r.json())); setValLoading(false); }
        else if (s.status==="error") { clearInterval(poll); alert("驗證失敗："+s.error); setValLoading(false); }
      },2000);
    } catch(e) { alert("驗證失敗："+e.message); setValLoading(false); }
  };

  const STOCK_POOL = [
    {id:"2330",name:"台積電",sector:"半導體"},{id:"2317",name:"鴻海",sector:"電子製造"},
    {id:"2454",name:"聯發科",sector:"IC設計"},{id:"2412",name:"中華電",sector:"電信"},
    {id:"2308",name:"台達電",sector:"電源"},{id:"2882",name:"國泰金",sector:"金融"},
    {id:"1301",name:"台塑",sector:"石化"},{id:"2002",name:"中鋼",sector:"鋼鐵"},
    {id:"2886",name:"兆豐金",sector:"金融"},{id:"0050",name:"元大台灣50",sector:"ETF"},
  ];

  const aiReturn   = result?.total_return;
  const rfReturn   = result?.risk_free_return;
  const outperform = aiReturn!=null&&rfReturn!=null?(aiReturn-rfReturn).toFixed(2):null;

  const tabStyle = (t) => ({
    padding:"8px 20px", fontSize:14, fontWeight:activeTab===t?500:400,
    borderBottom:activeTab===t?"2px solid #1D9E75":"2px solid transparent",
    color:activeTab===t?"var(--color-text-primary)":"var(--color-text-secondary)",
    cursor:"pointer", background:"transparent", border:"none",
    borderBottom:activeTab===t?"2px solid #1D9E75":"2px solid transparent",
  });

  return (
    <div style={{minHeight:"100vh",background:"var(--color-background-tertiary)",fontFamily:"var(--font-sans)",color:"var(--color-text-primary)"}}>
      {/* 頂部導覽 */}
      <div style={{background:"var(--color-background-primary)",borderBottom:"0.5px solid var(--color-border-tertiary)",padding:"16px 24px",position:"sticky",top:0,zIndex:10}}>
        <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:12}}>
          <div>
            <div style={{fontSize:18,fontWeight:500}}>AI 投資組合管理系統</div>
            <div style={{fontSize:12,color:"var(--color-text-secondary)",marginTop:2}}>Portfolio SAC · 10 支台股 · 31 特徵 · Walk-forward 驗證</div>
          </div>
          <div style={{fontSize:11,color:"var(--color-text-tertiary)",background:"var(--color-background-secondary)",padding:"4px 10px",borderRadius:"var(--border-radius-md)"}}>台股 10 支</div>
        </div>
        {/* Tab 切換 */}
        <div style={{display:"flex",gap:4}}>
          <button style={tabStyle("standard")} onClick={()=>setActiveTab("standard")}>單模型訓練</button>
          <button style={tabStyle("walkforward")} onClick={()=>setActiveTab("walkforward")}>Walk-forward 訓練</button>
        </div>
      </div>

      <div style={{maxWidth:960,margin:"0 auto",padding:"24px 16px"}}>

        {/* ── 股票池 ── */}
        <div style={{background:"var(--color-background-primary)",border:"0.5px solid var(--color-border-tertiary)",borderRadius:"var(--border-radius-lg)",padding:"20px 24px",marginBottom:16}}>
          <div style={{fontSize:14,fontWeight:500,marginBottom:12}}>股票池（10 支）</div>
          <div style={{display:"grid",gridTemplateColumns:"repeat(5,1fr)",gap:8}}>
            {STOCK_POOL.map(s=>(
              <div key={s.id} style={{background:"var(--color-background-secondary)",borderRadius:"var(--border-radius-md)",padding:"8px 12px"}}>
                <div style={{fontSize:13,fontWeight:500}}>{s.name}</div>
                <div style={{fontSize:11,color:"var(--color-text-tertiary)"}}>{s.id} · {s.sector}</div>
              </div>
            ))}
          </div>
        </div>

        {/* ══════════════════════════════════════════════════════
            TAB 1：單模型訓練（原有功能）
        ══════════════════════════════════════════════════════ */}
        {activeTab==="standard"&&(<>

          {/* 訓練設定 */}
          <div style={{background:"var(--color-background-primary)",border:"0.5px solid var(--color-border-tertiary)",borderRadius:"var(--border-radius-lg)",padding:"20px 24px",marginBottom:16}}>
            <div style={{fontSize:14,fontWeight:500,marginBottom:16}}>訓練設定</div>
            {period!=="6y"&&(
              <div style={{background:"rgba(186,117,23,0.08)",border:"0.5px solid rgba(186,117,23,0.3)",borderRadius:"var(--border-radius-md)",padding:"10px 14px",marginBottom:16,fontSize:12,color:"#854f0b"}}>
                建議選擇「6 年」以配合 pos_252 年度位置特徵
              </div>
            )}
            <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:16,marginBottom:20}}>
              <div>
                <label style={{fontSize:12,color:"var(--color-text-secondary)",display:"block",marginBottom:6}}>歷史資料期間</label>
                <select value={period} onChange={e=>setPeriod(e.target.value)} style={{width:"100%"}}>
                  {PERIOD_OPTIONS.map(o=><option key={o.value} value={o.value}>{o.label}</option>)}
                </select>
              </div>
              <div>
                <label style={{fontSize:12,color:"var(--color-text-secondary)",display:"block",marginBottom:6}}>訓練回合：{episodes}</label>
                <input type="range" min={20} max={200} step={10} value={episodes} onChange={e=>setEpisodes(+e.target.value)} style={{width:"100%"}}/>
              </div>
              <div>
                <label style={{fontSize:12,color:"var(--color-text-secondary)",display:"block",marginBottom:6}}>初始資金</label>
                <select value={capital} onChange={e=>setCapital(+e.target.value)} style={{width:"100%"}}>
                  {[10000,30000,50000,100000,500000,1000000,3000000].map(v=>(
                    <option key={v} value={v}>{v>=10000?(v/10000)+"萬":v}</option>
                  ))}
                </select>
              </div>
              <div>
                <label style={{fontSize:12,color:"var(--color-text-secondary)",display:"block",marginBottom:6}}>驗證段：{valDays} 天</label>
                <input type="range" min={60} max={500} step={10} value={valDays} onChange={e=>setValDays(+e.target.value)} style={{width:"100%"}}/>
              </div>
            </div>
            <button onClick={startTraining} disabled={status==="running"||status==="starting"} style={{fontSize:14,padding:"8px 24px"}}>
              {status==="running"?`訓練中… ${elapsed}s`:status==="starting"?"啟動中…":"開始訓練 AI →"}
            </button>
            {status==="running"&&(
              <div style={{marginTop:14}}>
                <div style={{display:"flex",justifyContent:"space-between",fontSize:12,color:"var(--color-text-secondary)",marginBottom:8}}>
                  <span>Episode {currentEp} / {episodes}</span>
                  <span>{elapsed}s</span>
                </div>
                <div style={{height:6,background:"var(--color-background-secondary)",borderRadius:3,overflow:"hidden",marginBottom:12}}>
                  <div style={{height:"100%",borderRadius:3,background:"#1D9E75",width:`${episodes>0?Math.round(currentEp/episodes*100):0}%`,transition:"width 0.4s ease"}}/>
                </div>
                {progress.length>0&&(
                  <div style={{background:"var(--color-background-secondary)",borderRadius:"var(--border-radius-md)",padding:"10px 14px",maxHeight:140,overflowY:"auto",fontFamily:"var(--font-mono)",fontSize:12}}>
                    {[...progress].reverse().slice(0,6).map((p,i)=>(
                      <div key={i} style={{display:"flex",gap:16,padding:"3px 0",borderBottom:"0.5px solid var(--color-border-tertiary)",color:i===0?"var(--color-text-primary)":"var(--color-text-tertiary)"}}>
                        <span style={{minWidth:70}}>ep {String(p.ep).padStart(3,"0")}/{p.total}</span>
                        <span style={{minWidth:80,color:p.ret>=1?"var(--color-text-success)":"var(--color-text-danger)"}}>
                          {p.ret>=1?"+":""}{((p.ret-1)*100).toFixed(2)}%
                        </span>
                        <span>α={p.alpha}</span>
                        <span style={{color:"var(--color-text-tertiary)"}}>{p.elapsed}s</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
            {error&&<div style={{marginTop:12,padding:"10px 14px",borderRadius:"var(--border-radius-md)",background:"var(--color-background-danger)",color:"var(--color-text-danger)",fontSize:13}}>{error}</div>}
          </div>

          {/* 訓練結果 */}
          {result&&(<>
            <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:12,marginBottom:16}}>
              <MetricCard label="初始資金" value={`$${result.initial_capital?.toLocaleString()}`} sub="投入本金"/>
              <MetricCard label="最終資金" value={`$${result.final_capital?.toLocaleString()}`}
                sub={`報酬率 ${aiReturn>=0?"+":""}${aiReturn}%`}
                color={result.final_capital>=result.initial_capital?"var(--color-text-success)":"var(--color-text-danger)"}/>
              <MetricCard label="超過無風險利率" value={`${outperform>=0?"+":""}${outperform}%`}
                sub={`定存基準 +${rfReturn}%`}
                color={outperform>=0?"var(--color-text-success)":"var(--color-text-danger)"}/>
              <MetricCard label="交易勝率" value={`${result.win_rate}%`}
                sub={`共 ${result.n_trades} 筆交易`}
                color={result.win_rate>=50?"var(--color-text-success)":"var(--color-text-danger)"}/>
            </div>

            <div style={{background:"var(--color-background-primary)",border:"0.5px solid var(--color-border-tertiary)",borderRadius:"var(--border-radius-lg)",padding:"20px 24px",marginBottom:16}}>
              <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:4}}>
                <div style={{fontSize:14,fontWeight:500}}>明日持倉建議</div>
                <button onClick={()=>fetchPrediction(period)} disabled={predLoading} style={{fontSize:12,padding:"5px 14px"}}>
                  {predLoading?"預測中…":"重新預測"}
                </button>
              </div>
              <PredictionPanel pred={prediction}/>
            </div>

            {result.avg_positions&&(
              <div style={{background:"var(--color-background-primary)",border:"0.5px solid var(--color-border-tertiary)",borderRadius:"var(--border-radius-lg)",padding:"20px 24px",marginBottom:16}}>
                <div style={{fontSize:14,fontWeight:500,marginBottom:12}}>AI 平均持倉比例</div>
                <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"0 24px"}}>
                  {STOCK_POOL.filter(s=>s.id!=="0050").map(s=>(
                    <PositionBar key={s.id} name={s.name} pct={result.avg_positions[s.id]||0} sector={s.sector}/>
                  ))}
                </div>
              </div>
            )}

            <div style={{background:"var(--color-background-primary)",border:"0.5px solid var(--color-border-tertiary)",borderRadius:"var(--border-radius-lg)",padding:"20px 24px",marginBottom:16}}>
              <div style={{fontSize:14,fontWeight:500,marginBottom:16}}>回測淨值曲線</div>
              <PortfolioChart portfolio={result.portfolio_curve} bh={result.bh_curve} dates={result.dates}/>
            </div>

            <div style={{background:"var(--color-background-primary)",border:"0.5px solid var(--color-border-tertiary)",borderRadius:"var(--border-radius-lg)",padding:"20px 24px",marginBottom:16}}>
              <div style={{fontSize:14,fontWeight:500,marginBottom:4}}>回測交易明細</div>
              <div style={{fontSize:12,color:"var(--color-text-secondary)",marginBottom:16}}>含手續費 0.1425% + 證交稅 0.3%，零股最低手續費 1 元</div>
              <TradeLog trades={result.trade_log}/>
            </div>
          </>)}

          {/* 驗證 */}
          <div style={{background:"var(--color-background-primary)",border:"0.5px solid var(--color-border-tertiary)",borderRadius:"var(--border-radius-lg)",padding:"20px 24px",marginBottom:16}}>
            <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:12}}>
              <div style={{fontSize:14,fontWeight:500}}>驗證模式</div>
              <span style={{fontSize:11,color:"var(--color-text-tertiary)"}}>載入已訓練模型，不重新訓練</span>
            </div>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr auto",gap:16,alignItems:"end",marginBottom:16}}>
              <div>
                <label style={{fontSize:12,color:"var(--color-text-secondary)",display:"block",marginBottom:6}}>驗證資金</label>
                <select value={valCapital} onChange={e=>setValCapital(+e.target.value)} style={{width:"100%"}}>
                  {[10000,30000,50000,100000,500000,1000000,3000000].map(v=>(
                    <option key={v} value={v}>{v>=10000?(v/10000)+"萬":v}</option>
                  ))}
                </select>
              </div>
              <div>
                <label style={{fontSize:12,color:"var(--color-text-secondary)",display:"block",marginBottom:6}}>驗證段：{valDays} 天</label>
                <input type="range" min={60} max={500} step={10} value={valDays} onChange={e=>setValDays(+e.target.value)} style={{width:"100%"}}/>
              </div>
              <button onClick={startValidation} disabled={valLoading} style={{fontSize:13,padding:"8px 20px",whiteSpace:"nowrap"}}>
                {valLoading?"驗證中…":"開始驗證 →"}
              </button>
            </div>
            {valResult&&(<>
              <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:12,marginBottom:16}}>
                {[
                  {label:"驗證期間",value:`${valResult.val_start?.slice(0,7)} ~ ${valResult.val_end?.slice(0,7)}`,sub:`共 ${valResult.val_days} 個交易日`},
                  {label:"AI 報酬率",value:`${valResult.total_return>=0?"+":""}${valResult.total_return}%`,
                   sub:`$${valResult.final_capital?.toLocaleString()}`,
                   color:valResult.total_return>=0?"var(--color-text-success)":"var(--color-text-danger)"},
                  {label:"超過無風險利率",value:`${(valResult.total_return-(valResult.risk_free_return||0)).toFixed(2)}%`,
                   sub:`定存基準 +${valResult.risk_free_return||0}%`,
                   color:(valResult.total_return-(valResult.risk_free_return||0))>=0?"var(--color-text-success)":"var(--color-text-danger)"},
                  {label:"交易勝率",value:`${valResult.win_rate}%`,sub:`共 ${valResult.n_trades} 筆交易`,
                   color:valResult.win_rate>=50?"var(--color-text-success)":"var(--color-text-danger)"},
                ].map((m,i)=><MetricCard key={i} {...m}/>)}
              </div>
              <PortfolioChart portfolio={valResult.portfolio_curve} bh={valResult.bh_curve} dates={valResult.dates}/>
              <div style={{marginTop:16,fontSize:13,fontWeight:500,marginBottom:8}}>驗證期間交易明細</div>
              <TradeLog trades={valResult.trade_log}/>
            </>)}
          </div>
        </>)}

        {/* ══════════════════════════════════════════════════════
            TAB 2：Walk-forward 訓練
        ══════════════════════════════════════════════════════ */}
        {activeTab==="walkforward"&&(<>

          {/* Walk-forward 說明 */}
          <div style={{background:"rgba(29,158,117,0.06)",border:"0.5px solid rgba(29,158,117,0.2)",borderRadius:"var(--border-radius-lg)",padding:"16px 20px",marginBottom:16,fontSize:13}}>
            <div style={{fontWeight:500,marginBottom:6}}>Walk-forward 訓練策略</div>
            <div style={{color:"var(--color-text-secondary)",lineHeight:1.7}}>
              將資料切成 3 個滾動窗口（各 3 年訓練 + 1 年驗證），每個窗口獨立訓練一個模型。
              預測時根據當前市場環境（牛市/熊市/盤整）自動選擇最適合的模型，提升策略泛化能力。
            </div>
          </div>

          {/* Walk-forward 設定 */}
          <div style={{background:"var(--color-background-primary)",border:"0.5px solid var(--color-border-tertiary)",borderRadius:"var(--border-radius-lg)",padding:"20px 24px",marginBottom:16}}>
            <div style={{fontSize:14,fontWeight:500,marginBottom:16}}>Walk-forward 設定</div>
            <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:16,marginBottom:20}}>
              <div>
                <label style={{fontSize:12,color:"var(--color-text-secondary)",display:"block",marginBottom:6}}>資料期間（建議 6y）</label>
                <select value={period} onChange={e=>setPeriod(e.target.value)} style={{width:"100%"}}>
                  {PERIOD_OPTIONS.map(o=><option key={o.value} value={o.value}>{o.label}</option>)}
                </select>
              </div>
              <div>
                <label style={{fontSize:12,color:"var(--color-text-secondary)",display:"block",marginBottom:6}}>每窗口回合：{wfEpisodes}</label>
                <input type="range" min={10} max={700} step={10} value={wfEpisodes} onChange={e=>setWfEpisodes(+e.target.value)} style={{width:"100%"}}/>
                <div style={{fontSize:11,color:"var(--color-text-tertiary)",marginTop:4}}>
                  總計約 {wfEpisodes * 3} 回合（3 個窗口）
                </div>
              </div>
              <div>
                <label style={{fontSize:12,color:"var(--color-text-secondary)",display:"block",marginBottom:6}}>初始資金</label>
                <select value={capital} onChange={e=>setCapital(+e.target.value)} style={{width:"100%"}}>
                  {[10000,30000,50000,100000,500000,1000000,3000000].map(v=>(
                    <option key={v} value={v}>{v>=10000?(v/10000)+"萬":v}</option>
                  ))}
                </select>
              </div>
            </div>

            {/* 窗口預覽 */}
            <div style={{background:"var(--color-background-secondary)",borderRadius:"var(--border-radius-md)",padding:"12px 16px",marginBottom:16,fontSize:12}}>
              <div style={{fontWeight:500,marginBottom:8,color:"var(--color-text-secondary)"}}>窗口規劃（3年訓練 + 1年驗證）</div>
              <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:8}}>
                {[
                  {w:1,train:"2020-04 ~ 2023-04",val:"2023-04 ~ 2024-04",regime:"bull"},
                  {w:2,train:"2021-04 ~ 2024-04",val:"2024-04 ~ 2025-04",regime:"sideways"},
                  {w:3,train:"2022-04 ~ 2025-04",val:"2025-04 ~ 2026-04",regime:"bear"},
                ].map(({w,train,val,regime})=>(
                  <div key={w} style={{background:"var(--color-background-primary)",borderRadius:"var(--border-radius-md)",padding:"10px 12px"}}>
                    <div style={{display:"flex",justifyContent:"space-between",marginBottom:4}}>
                      <span style={{fontWeight:500}}>窗口 {w}</span>
                      <span style={{color:REGIME_COLOR[regime]}}>{REGIME_LABEL[regime]}</span>
                    </div>
                    <div style={{color:"var(--color-text-tertiary)",fontSize:11}}>
                      訓練 {train}<br/>驗證 {val}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <button onClick={startWalkForward} disabled={wfStatus==="running"||wfStatus==="starting"}
              style={{fontSize:14,padding:"8px 24px"}}>
              {wfStatus==="running"?`Walk-forward 訓練中… ${wfElapsed}s`:
               wfStatus==="starting"?"啟動中…":"開始 Walk-forward 訓練 →"}
            </button>
            {wfStatus==="running"&&(
              <div style={{marginTop:12,padding:"10px 14px",borderRadius:"var(--border-radius-md)",
                background:"var(--color-background-secondary)",fontSize:12,color:"var(--color-text-secondary)"}}>
                正在訓練 3 個窗口，每個窗口 {wfEpisodes} 回合。訓練時間約 {Math.round(wfEpisodes*3*2/60)} 分鐘…
              </div>
            )}
            {wfError&&<div style={{marginTop:12,padding:"10px 14px",borderRadius:"var(--border-radius-md)",
              background:"var(--color-background-danger)",color:"var(--color-text-danger)",fontSize:13}}>{wfError}</div>}
          </div>

          {/* Run A~D 對比績效表 */}
          {(wfServerStatus?.runs||wfResult)&&(
            <div style={{background:"var(--color-background-primary)",border:"0.5px solid var(--color-border-tertiary)",borderRadius:"var(--border-radius-lg)",padding:"20px 24px",marginBottom:16}}>
              <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:16}}>
                <div>
                  <div style={{fontSize:14,fontWeight:500}}>Run A ~ D 策略對比</div>
                  <div style={{fontSize:12,color:"var(--color-text-secondary)",marginTop:2}}>
                    3 個窗口 × 4 種策略的訓練集 / 驗證集報酬
                  </div>
                </div>
                <button onClick={fetchWfStatus} style={{fontSize:12,padding:"4px 12px"}}>重新整理</button>
              </div>
              <RunComparisonTable runs={wfServerStatus?.runs}/>
            </div>
          )}

          {/* Walk-forward 預測 */}
          <div style={{background:"var(--color-background-primary)",border:"0.5px solid var(--color-border-tertiary)",borderRadius:"var(--border-radius-lg)",padding:"20px 24px",marginBottom:16}}>
            <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:12}}>
              <div>
                <div style={{fontSize:14,fontWeight:500}}>明日持倉建議（Regime 選模型）</div>
                <div style={{fontSize:12,color:"var(--color-text-secondary)",marginTop:2}}>
                  根據當前市場環境自動選擇最適合的窗口模型
                </div>
              </div>
              <button onClick={fetchWfPrediction} disabled={wfPredLoading} style={{fontSize:12,padding:"5px 14px"}}>
                {wfPredLoading?"預測中…":"取得預測"}
              </button>
            </div>
            {/* 模型來源標籤 */}
            {wfPrediction&&(
              <div style={{display:"flex",gap:8,flexWrap:"wrap",marginBottom:12}}>
                {wfPrediction.selected_run&&(
                  <div style={{display:"inline-flex",alignItems:"center",gap:6,
                    padding:"4px 12px",borderRadius:6,fontSize:12,fontWeight:500,
                    background:"rgba(29,158,117,0.1)",border:"0.5px solid rgba(29,158,117,0.3)",
                    color:"#0f6e56"}}>
                    <span style={{width:6,height:6,borderRadius:"50%",background:"#1D9E75",display:"inline-block"}}/>
                    Run {wfPrediction.selected_run}
                    <span style={{fontWeight:400,color:"var(--color-text-secondary)"}}>
                      — {RUN_DESC[wfPrediction.selected_run]||""}
                    </span>
                  </div>
                )}
                {wfPrediction.selected_window&&(
                  <div style={{display:"inline-flex",alignItems:"center",gap:6,
                    padding:"4px 12px",borderRadius:6,fontSize:12,
                    background:"var(--color-background-secondary)",
                    color:"var(--color-text-secondary)"}}>
                    窗口 {wfPrediction.selected_window}
                  </div>
                )}
                {wfPrediction.current_regime&&(
                  <div style={{display:"inline-flex",alignItems:"center",gap:6,
                    padding:"4px 12px",borderRadius:6,fontSize:12,
                    background:"var(--color-background-secondary)",
                    color:REGIME_COLOR[wfPrediction.current_regime]||"var(--color-text-secondary)"}}>
                    {REGIME_LABEL[wfPrediction.current_regime]||wfPrediction.current_regime}
                  </div>
                )}
              </div>
            )}
            <PredictionPanel pred={wfPrediction}/>
          </div>
        </>)}

        {/* ── 排程狀態（兩個 Tab 共用）── */}
        <div style={{background:"var(--color-background-primary)",border:"0.5px solid var(--color-border-tertiary)",borderRadius:"var(--border-radius-lg)",padding:"20px 24px",marginBottom:16}}>
          <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:16}}>
            <div>
              <div style={{fontSize:14,fontWeight:500,marginBottom:4}}>每日自動預測排程</div>
              <div style={{fontSize:12,color:"var(--color-text-secondary)"}}>每週一至週五 15:30 自動執行</div>
            </div>
            <div style={{display:"flex",gap:8}}>
              <button onClick={runNow} style={{fontSize:12,padding:"6px 14px"}}>立即執行</button>
              <button onClick={()=>{setShowHistory(h=>!h);fetchHistory();}} style={{fontSize:12,padding:"6px 14px"}}>
                {showHistory?"隱藏歷史":"查看歷史"}
              </button>
            </div>
          </div>
          <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:12}}>
            <div style={{background:"var(--color-background-secondary)",borderRadius:"var(--border-radius-md)",padding:"12px 16px"}}>
              <div style={{fontSize:11,color:"var(--color-text-tertiary)",marginBottom:4}}>排程狀態</div>
              <div style={{fontSize:14,fontWeight:500,color:scheduler?.running?"var(--color-text-success)":"var(--color-text-danger)"}}>
                {scheduler?.running?"執行中":"已停止"}
              </div>
            </div>
            <div style={{background:"var(--color-background-secondary)",borderRadius:"var(--border-radius-md)",padding:"12px 16px"}}>
              <div style={{fontSize:11,color:"var(--color-text-tertiary)",marginBottom:4}}>下次執行</div>
              <div style={{fontSize:13,fontWeight:500,fontFamily:"var(--font-mono)"}}>{scheduler?.next_run?.slice(0,16)||"—"}</div>
            </div>
            <div style={{background:"var(--color-background-secondary)",borderRadius:"var(--border-radius-md)",padding:"12px 16px"}}>
              <div style={{fontSize:11,color:"var(--color-text-tertiary)",marginBottom:4}}>歷史紀錄</div>
              <div style={{fontSize:14,fontWeight:500}}>{scheduler?.history_count||0} 筆</div>
            </div>
          </div>
          {showHistory&&history.length>0&&(
            <div style={{marginTop:16,overflowX:"auto"}}>
              <table style={{width:"100%",borderCollapse:"collapse",fontSize:12}}>
                <thead><tr style={{borderBottom:"0.5px solid var(--color-border-tertiary)"}}>
                  {["預測日期","股票","建議","目標倉位","當時價格","實際報酬"].map(h=>(
                    <th key={h} style={{padding:"6px 10px",textAlign:"left",fontWeight:500,color:"var(--color-text-secondary)"}}>{h}</th>
                  ))}
                </tr></thead>
                <tbody>
                  {history.map((r,i)=>{
                    const isPos=r.actual_return?.startsWith("+");
                    const isNeg=r.actual_return?.startsWith("-");
                    return (
                      <tr key={i} style={{borderBottom:"0.5px solid var(--color-border-tertiary)",background:i%2===0?"transparent":"var(--color-background-secondary)"}}>
                        <td style={{padding:"6px 10px",fontFamily:"var(--font-mono)",color:"var(--color-text-secondary)"}}>{r.date}</td>
                        <td style={{padding:"6px 10px",fontWeight:500}}>{r.stock_name}<span style={{fontSize:11,color:"var(--color-text-tertiary)",marginLeft:4}}>{r.stock_id}</span></td>
                        <td style={{padding:"6px 10px"}}>
                          <span style={{padding:"2px 8px",borderRadius:4,fontSize:11,
                            background:r.action==="買入"?"rgba(29,158,117,0.1)":r.action==="持有"?"rgba(55,138,221,0.1)":"var(--color-background-secondary)",
                            color:r.action==="買入"?"#0f6e56":r.action==="持有"?"#185FA5":"var(--color-text-tertiary)"}}>
                            {r.action}
                          </span>
                        </td>
                        <td style={{padding:"6px 10px",fontFamily:"var(--font-mono)"}}>{r.target_pct}%</td>
                        <td style={{padding:"6px 10px",fontFamily:"var(--font-mono)"}}>${parseFloat(r.latest_price).toLocaleString()}</td>
                        <td style={{padding:"6px 10px",fontFamily:"var(--font-mono)",fontWeight:500,
                          color:isPos?"var(--color-text-success)":isNeg?"var(--color-text-danger)":"var(--color-text-tertiary)"}}>
                          {r.actual_return||"待回填"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* 已儲存模型 */}
        {savedModels.length>0&&(
          <div style={{background:"var(--color-background-primary)",border:"0.5px solid var(--color-border-tertiary)",borderRadius:"var(--border-radius-lg)",padding:"20px 24px",marginBottom:16}}>
            <div style={{fontSize:14,fontWeight:500,marginBottom:16}}>已儲存的模型</div>
            <table style={{width:"100%",borderCollapse:"collapse",fontSize:13}}>
              <thead><tr style={{borderBottom:"0.5px solid var(--color-border-tertiary)"}}>
                {["期間","目標回合","累積訓練","AI 報酬","儲存時間"].map(h=>(
                  <th key={h} style={{padding:"8px 12px",textAlign:"left",fontWeight:500,fontSize:12,color:"var(--color-text-secondary)"}}>{h}</th>
                ))}
              </tr></thead>
              <tbody>
                {savedModels.map((m,i)=>(
                  <tr key={i} style={{borderBottom:"0.5px solid var(--color-border-tertiary)",background:i%2===0?"transparent":"var(--color-background-secondary)"}}>
                    <td style={{padding:"9px 12px",fontWeight:500}}>
                      {m.period||"—"}
                    </td>
                    <td style={{padding:"9px 12px",fontFamily:"var(--font-mono)",color:"var(--color-text-secondary)"}}>
                      {m.episodes!=null?m.episodes:"—"}
                    </td>
                    <td style={{padding:"9px 12px",fontFamily:"var(--font-mono)",
                      color:m.episodes_done>m.episodes?"var(--color-text-success)":m.episodes_done===m.episodes?"var(--color-text-primary)":"var(--color-text-warning)"}}>
                      {m.episodes_done!=null?m.episodes_done:"—"}
                      {m.episodes_done!=null&&m.episodes_done>=m.episodes&&
                        <span style={{fontSize:11,marginLeft:4,color:"var(--color-text-success)"}}>✓</span>}
                    </td>
                    <td style={{padding:"9px 12px",fontFamily:"var(--font-mono)",fontWeight:500,
                      color:m.total_return==null?"var(--color-text-tertiary)":m.total_return>=0?"var(--color-text-success)":"var(--color-text-danger)"}}>
                      {m.total_return!=null
                        ? `${m.total_return>=0?"+":""}${m.total_return}%`
                        : "—"}
                    </td>
                    <td style={{padding:"9px 12px",fontSize:12,color:"var(--color-text-tertiary)",fontFamily:"var(--font-mono)"}}>
                      {m.saved_at||"—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}