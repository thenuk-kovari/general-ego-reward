"""Gradio web app backed by the paper-faithful TOPReward scorer."""

from __future__ import annotations

import argparse
import html
import json
import tempfile
from pathlib import Path

import gradio as gr
from PIL import Image

from .serve import sample_frames
from .topreward import TOPRewardScorer, minmax

SYNC_HEAD = r"""
<script>
(() => {
  const NS = "http://www.w3.org/2000/svg";
  function setup() {
    const root = document.querySelector("#reward-live-chart .reward-sync");
    const video = document.querySelector("#reward-video video");
    if (!root || !video || root.dataset.bound === video.currentSrc) return;
    let points;
    try { points = JSON.parse(root.dataset.points); } catch (_) { return; }
    if (!points.length) return;
    root.dataset.bound = video.currentSrc || "pending";
    const svg = root.querySelector("svg");
    const W=760, H=390, L=60, R=18, T=24, B=48;
    const maxT = Math.max(video.duration || 0, ...points.map(p => p.time_s), 1);
    const sx = t => L + Math.min(Math.max(t/maxT, 0), 1)*(W-L-R);
    const sy = r => T + (1-r)*(H-T-B);
    const add = (name, attrs, text) => {
      const el = document.createElementNS(NS, name);
      Object.entries(attrs).forEach(([k,v]) => el.setAttribute(k,v));
      if (text !== undefined) el.textContent = text;
      svg.appendChild(el); return el;
    };
    svg.replaceChildren();
    [0,.25,.5,.75,1].forEach(v => {
      add("line", {x1:L,x2:W-R,y1:sy(v),y2:sy(v),stroke:"#273449","stroke-width":1});
      add("text", {x:L-10,y:sy(v)+5,fill:"#94a3b8","text-anchor":"end","font-size":13}, v.toFixed(2));
    });
    add("line", {x1:L,x2:W-R,y1:H-B,y2:H-B,stroke:"#64748b","stroke-width":1.5});
    add("polyline", {points:points.map(p => `${sx(p.time_s)},${sy(p.reward)}`).join(" "),fill:"none",stroke:"#60a5fa","stroke-width":4,"stroke-linejoin":"round"});
    points.forEach(p => add("circle", {cx:sx(p.time_s),cy:sy(p.reward),r:5,fill:"#93c5fd",stroke:"#172033","stroke-width":2}));
    [0,maxT/4,maxT/2,3*maxT/4,maxT].forEach(t => add("text", {x:sx(t),y:H-18,fill:"#94a3b8","text-anchor":"middle","font-size":13}, `${t.toFixed(1)}s`));
    const cursor = add("line", {x1:L,x2:L,y1:T,y2:H-B,stroke:"#fbbf24","stroke-width":3});
    const dot = add("circle", {cx:L,cy:sy(points[0].reward),r:7,fill:"#fbbf24",stroke:"#fff","stroke-width":2});
    const readout = root.querySelector(".reward-readout");
    const update = () => {
      const t = video.currentTime || 0;
      let a=points[0], b=points[points.length-1];
      for (let i=1; i<points.length; i++) if (t <= points[i].time_s) { a=points[i-1]; b=points[i]; break; }
      const mix = Math.min(Math.max((t-a.time_s)/Math.max(b.time_s-a.time_s,1e-6),0),1);
      const reward = a.reward + mix*(b.reward-a.reward);
      const x=sx(t), y=sy(reward);
      cursor.setAttribute("x1",x); cursor.setAttribute("x2",x);
      dot.setAttribute("cx",x); dot.setAttribute("cy",y);
      readout.textContent = `t = ${t.toFixed(2)}s · reward = ${reward.toFixed(3)}`;
    };
    video.addEventListener("timeupdate",update);
    video.addEventListener("loadedmetadata",() => { root.dataset.bound=""; setup(); });
    update();
  }
  new MutationObserver(setup).observe(document.documentElement,{childList:true,subtree:true});
  window.addEventListener("load",setup);
  setInterval(setup,1000);
})();
</script>
"""


def reward_chart_html(points: list[dict]) -> str:
    encoded = html.escape(json.dumps(points), quote=True)
    return (
        f'<div class="reward-sync" data-points="{encoded}" style="background:#111827;border:1px solid #334155;border-radius:12px;padding:12px;">'
        '<div style="display:flex;justify-content:space-between;align-items:center;margin:0 8px 4px;color:#e2e8f0">'
        '<b>Reward vs. video time</b><span class="reward-readout" style="font-variant-numeric:tabular-nums;color:#fbbf24"></span></div>'
        '<svg viewBox="0 0 760 390" style="width:100%;height:auto;display:block"></svg></div>'
    )


def build_app(model_name: str, fps: float, load_in_4bit: bool = False, initial_results: Path | None = None, initial_video: Path | None = None) -> gr.Blocks:
    scorer = TOPRewardScorer(model_name, load_in_4bit=load_in_4bit)
    initial_details = ""
    initial_task = ""
    initial_chart = reward_chart_html([])
    if initial_results:
        payload = json.loads(initial_results.read_text())
        initial_task = payload["task"]
        points = payload["points"]
        for point in points:
            point.setdefault("time_s", point["source_frame"] / 30.0)
        initial_details = json.dumps(points, indent=2)
        initial_chart = reward_chart_html(points)

    def analyze(video: str | None, task: str, samples: int, progress: gr.Progress = gr.Progress()):
        if not video or not task.strip():
            raise gr.Error("Upload a video and provide a task description.")
        with tempfile.TemporaryDirectory(prefix="topreward-") as temp:
            sampled = sample_frames(video, int(samples), Path(temp))
            images = [Image.open(path).convert("RGB") for _, path in sampled]
            raw, metadata = [], []
            for end in range(2, len(images)+1):
                progress((end-2)/max(len(images)-1,1), desc=f"Scoring prefix {end-1}/{len(images)-1}")
                value, token_id, token_text = scorer.score(images[:end], task.strip(), fps)
                raw.append(value)
                metadata.append({"time_s": sampled[end-1][0], "log_p_true": value, "token_id": token_id, "token_text": token_text})
        for item, reward in zip(metadata, minmax(raw), strict=True):
            item["reward"] = reward
        return reward_chart_html(metadata), json.dumps(metadata, indent=2)

    with gr.Blocks(title="Ego TOPReward") as app:
        gr.Markdown("# Ego TOPReward\nPlay the head video to move the yellow cursor through the zero-shot reward curve in real time.")
        with gr.Row(equal_height=True):
            video = gr.Video(value=initial_video, label="GenRobot head camera", sources=["upload"], elem_id="reward-video", autoplay=False)
            chart = gr.HTML(value=initial_chart, elem_id="reward-live-chart")
        with gr.Row():
            task = gr.Textbox(label="Task", value=initial_task, placeholder="fold and store clothes", scale=3)
            samples = gr.Slider(3,16,value=8,step=1,label="Uniform temporal samples",scale=2)
            submit = gr.Button("Compute TOPReward",variant="primary",scale=1)
        details = gr.Code(value=initial_details,label="Raw log P(True) and normalized reward",language="json")
        submit.click(analyze,inputs=[video,task,samples],outputs=[chart,details])
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--fps",type=float,default=2.0)
    parser.add_argument("--load-in-4bit",action="store_true",help="Use NF4 weights for 16 GB GPUs")
    parser.add_argument("--host",required=True,help="Tailscale IP to bind")
    parser.add_argument("--port",type=int,default=7860)
    parser.add_argument("--initial-results",type=Path,help="rewards.json to display when the page opens")
    parser.add_argument("--initial-video",type=Path,help="video to display when the page opens")
    args = parser.parse_args()
    build_app(args.model,args.fps,args.load_in_4bit,args.initial_results,args.initial_video).launch(
        server_name=args.host,server_port=args.port,share=False,inbrowser=False,
        allowed_paths=[str(args.initial_video.resolve())] if args.initial_video else None,head=SYNC_HEAD,
    )


if __name__ == "__main__":
    main()
