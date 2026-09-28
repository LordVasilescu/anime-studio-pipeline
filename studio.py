# © 2026 Mathew. All rights reserved.
"""Anime Studio — dead-simple UI on top of ComfyUI's API.

Pick an image, type the motion, hit ANIMATE. Clips land in a gallery;
check the ones you want and hit MAKE MOVIE to stitch them.

Run:  python studio.py   (needs ComfyUI running on 127.0.0.1:8188)
UI :  http://127.0.0.1:8500
"""
import base64
import io
import json
import mimetypes
import os
import random
import re
import shutil
import struct
import subprocess
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

COMFY = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
PORT = 8500
HERE = os.path.dirname(os.path.abspath(__file__))
COMFY_VIDEO_DIR = os.environ.get("COMFY_VIDEO_DIR", r"D:\ComfyUI\output\video")
MOVIE_DIR = os.path.join(HERE, "movies")
STITCH_PS1 = os.path.join(HERE, "stitch.ps1")

NEGATIVE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
            "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
            "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")

MODELS = {
    "unet_high": "Wan2_2-I2V_AniSoraV3_2_HIGH_14B_fp8_e4m3fn_scaled_KJ.safetensors",
    "unet_low": "Wan2_2-I2V_AniSoraV3_2_LOW_14B_fp8_e4m3fn_scaled_KJ.safetensors",
    "lora_high": "Wan_2_2_I2V_A14B_HIGH_lightx2v_4step_lora_rank_64_fp16.safetensors",
    "lora_low": "Wan_2_2_I2V_A14B_LOW_lightx2v_4step_lora_rank_64_fp16.safetensors",
    "clip": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "vae": "wan_2.1_vae.safetensors",
    "hy_unet": "hunyuanvideo1.5_480p_t2v_cfg_distilled_fp8_scaled.safetensors",
    "hy_clip1": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "hy_clip2": "byt5_small_glyphxl_fp16.safetensors",
    "hy_vae": "hunyuanvideo15_vae_fp16.safetensors",
}


def image_size(data: bytes):
    """Width/height from PNG or JPEG bytes (no PIL needed)."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        w, h = struct.unpack(">II", data[16:24])
        return w, h
    if data[:2] == b"\xff\xd8":  # JPEG: walk segments to a SOF marker
        i = 2
        while i < len(data) - 9:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                return w, h
            i += 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
    return 832, 480


def fit_resolution(w, h, budget=832 * 480):
    """Scale to ~480p area, dims multiples of 16."""
    scale = (budget / (w * h)) ** 0.5
    fw = max(256, int(w * scale) // 16 * 16)
    fh = max(256, int(h * scale) // 16 * 16)
    return fw, fh


def comfy_json(path, payload=None):
    req = urllib.request.Request(COMFY + path)
    if payload is not None:
        req.data = json.dumps(payload).encode()
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def comfy_upload(name, data):
    boundary = uuid.uuid4().hex
    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode())
    body.write(f'Content-Disposition: form-data; name="image"; filename="{name}"\r\n'.encode())
    body.write(b"Content-Type: application/octet-stream\r\n\r\n")
    body.write(data)
    body.write(f"\r\n--{boundary}--\r\n".encode())
    req = urllib.request.Request(COMFY + "/upload/image", data=body.getvalue())
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["name"]


def build_graph(image_name, prompt, width, height):
    """AniSora image-to-video (anime-specialized Wan 2.2 finetune)."""
    seed = random.randint(0, 2**48)
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": MODELS["unet_high"], "weight_dtype": "default"}},
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": MODELS["unet_low"], "weight_dtype": "default"}},
        "3": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": MODELS["lora_high"], "strength_model": 1.0}},
        "4": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["2", 0], "lora_name": MODELS["lora_low"], "strength_model": 1.0}},
        "5": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["3", 0], "shift": 8.0}},
        "6": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["4", 0], "shift": 8.0}},
        "7": {"class_type": "CLIPLoader", "inputs": {"clip_name": MODELS["clip"], "type": "wan"}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["7", 0]}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["7", 0]}},
        "10": {"class_type": "VAELoader", "inputs": {"vae_name": MODELS["vae"]}},
        "11": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "12": {"class_type": "WanImageToVideo", "inputs": {
            "positive": ["8", 0], "negative": ["9", 0], "vae": ["10", 0],
            "start_image": ["11", 0], "width": width, "height": height,
            "length": 81, "batch_size": 1}},
        "13": {"class_type": "KSamplerAdvanced", "inputs": {
            "model": ["5", 0], "add_noise": "enable", "noise_seed": seed,
            "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
            "positive": ["12", 0], "negative": ["12", 1], "latent_image": ["12", 2],
            "start_at_step": 0, "end_at_step": 2, "return_with_leftover_noise": "enable"}},
        "14": {"class_type": "KSamplerAdvanced", "inputs": {
            "model": ["6", 0], "add_noise": "disable", "noise_seed": 0,
            "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
            "positive": ["12", 0], "negative": ["12", 1], "latent_image": ["13", 0],
            "start_at_step": 2, "end_at_step": 10000, "return_with_leftover_noise": "disable"}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["10", 0]}},
        "16": {"class_type": "CreateVideo", "inputs": {"images": ["15", 0], "fps": 16}},
        "17": {"class_type": "SaveVideo", "inputs": {
            "video": ["16", 0], "filename_prefix": "video/AnimeStudio",
            "format": "mp4", "codec": "h264"}},
    }


def build_t2v_graph(prompt, width, height):
    """HunyuanVideo 1.5 direct text-to-video (cfg-distilled 480p), per official template."""
    seed = random.randint(0, 2**48)
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": MODELS["hy_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": 7.0}},
        "3": {"class_type": "DualCLIPLoader", "inputs": {
            "clip_name1": MODELS["hy_clip1"], "clip_name2": MODELS["hy_clip2"], "type": "hunyuan_video_15"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["3", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["3", 0]}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": MODELS["hy_vae"]}},
        "7": {"class_type": "EmptyHunyuanVideo15Latent", "inputs": {
            "width": width, "height": height, "length": 121, "batch_size": 1}},
        "8": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "9": {"class_type": "CFGGuider", "inputs": {
            "model": ["2", 0], "positive": ["4", 0], "negative": ["5", 0], "cfg": 1.0}},
        "10": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "11": {"class_type": "BasicScheduler", "inputs": {
            "model": ["2", 0], "scheduler": "simple", "steps": 20, "denoise": 1.0}},
        "12": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["8", 0], "guider": ["9", 0], "sampler": ["10", 0],
            "sigmas": ["11", 0], "latent_image": ["7", 0]}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 1], "vae": ["6", 0]}},
        "16": {"class_type": "CreateVideo", "inputs": {"images": ["15", 0], "fps": 24}},
        "17": {"class_type": "SaveVideo", "inputs": {
            "video": ["16", 0], "filename_prefix": "video/AnimeStudio",
            "format": "mp4", "codec": "h264"}},
    }


def find_video_in_history(hist_entry):
    for node_out in hist_entry.get("outputs", {}).values():
        for items in node_out.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict) and str(item.get("filename", "")).lower().endswith((".mp4", ".webm")):
                    return item["filename"]
    return None


def safe_name(name):
    return os.path.basename(name.replace("..", ""))


SDXL_PY = os.environ.get("SDXL_PY", r"D:\ZaryaArt\venv\Scripts\python.exe")
SDXL_GEN = os.environ.get("SDXL_GEN", r"D:\ZaryaArt\generate.py")


def sdxl_keyframe(prompt):
    """Generate an anime keyframe from words via the Zarya Art Studio SDXL CLI."""
    out = os.path.join(HERE, "keyframes", f"kf_{uuid.uuid4().hex[:8]}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    styled = f"anime style, high quality anime key visual, detailed, vibrant colors, {prompt}"
    r = subprocess.run([SDXL_PY, SDXL_GEN, "--prompt", styled, "--out", out, "--wide"],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or not os.path.isfile(out):
        raise RuntimeError(f"keyframe generation failed: {(r.stderr or r.stdout)[-300:]}")
    with open(out, "rb") as f:
        return f.read()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path):
        if not os.path.isfile(path):
            return self._send(404, {"error": "not found"})
        size = os.path.getsize(path)
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        rng = self.headers.get("Range")
        with open(path, "rb") as f:
            if rng:
                m = re.match(r"bytes=(\d+)-(\d*)", rng)
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
                f.seek(start)
                data = f.read(end - start + 1)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                data = f.read()
                self.send_response(200)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    def _body(self):
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))

    def do_GET(self):
        path, _, query = self.path.partition("?")
        params = dict(p.split("=", 1) for p in query.split("&") if "=" in p)
        if path == "/":
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if path == "/api/health":
            try:
                comfy_json("/queue")
                return self._send(200, {"ok": True})
            except Exception as e:
                return self._send(200, {"ok": False, "error": str(e)})
        if path.startswith("/api/status/"):
            pid = path.rsplit("/", 1)[1]
            try:
                hist = comfy_json(f"/history/{pid}")
                if pid in hist:
                    entry = hist[pid]
                    status = entry.get("status", {})
                    if status.get("status_str") == "error":
                        msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                        detail = msgs[0][1].get("exception_message", "unknown") if msgs else "unknown"
                        return self._send(200, {"state": "error", "error": detail})
                    video = find_video_in_history(entry)
                    return self._send(200, {"state": "done", "video": video})
                q = comfy_json("/queue")
                running = any(item[1] == pid for item in q.get("queue_running", []))
                pending = any(item[1] == pid for item in q.get("queue_pending", []))
                state = "running" if running else ("queued" if pending else "running")
                return self._send(200, {"state": state})
            except Exception as e:
                return self._send(200, {"state": "error", "error": str(e)})
        if path == "/api/clips":
            clips = []
            if os.path.isdir(COMFY_VIDEO_DIR):
                for f in os.listdir(COMFY_VIDEO_DIR):
                    if f.lower().endswith((".mp4", ".webm")):
                        full = os.path.join(COMFY_VIDEO_DIR, f)
                        clips.append({"name": f, "mtime": os.path.getmtime(full)})
            clips.sort(key=lambda c: c["mtime"], reverse=True)
            movies = []
            if os.path.isdir(MOVIE_DIR):
                movies = sorted((f for f in os.listdir(MOVIE_DIR) if f.endswith(".mp4")), reverse=True)
            return self._send(200, {"clips": clips, "movies": movies})
        if path == "/clip":
            return self._send_file(os.path.join(COMFY_VIDEO_DIR, safe_name(params.get("f", ""))))
        if path == "/movie":
            return self._send_file(os.path.join(MOVIE_DIR, safe_name(params.get("f", ""))))
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/api/animate":
            try:
                body = self._body()
                prompt = body.get("prompt", "").strip()
                if body.get("image_b64"):
                    data = base64.b64decode(body["image_b64"])
                elif body.get("paint"):
                    # optional: paint an anime keyframe with SDXL, then animate it
                    if not prompt:
                        return self._send(200, {"error": "type what you want to see first"})
                    data = sdxl_keyframe(prompt)
                else:
                    # direct text-to-video via HunyuanVideo 1.5
                    if not prompt:
                        return self._send(200, {"error": "type what you want to see first"})
                    graph = build_t2v_graph("anime style. " + prompt, 832, 480)
                    res = comfy_json("/prompt", {"prompt": graph, "client_id": "animestudio"})
                    return self._send(200, {"prompt_id": res["prompt_id"], "width": 832, "height": 480})
                w, h = fit_resolution(*image_size(data))
                ext = ".png" if data[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg"
                name = comfy_upload(f"animestudio_{uuid.uuid4().hex[:8]}{ext}", data)
                graph = build_graph(name, prompt or "subtle natural motion, gentle camera push in", w, h)
                res = comfy_json("/prompt", {"prompt": graph, "client_id": "animestudio"})
                return self._send(200, {"prompt_id": res["prompt_id"], "width": w, "height": h})
            except Exception as e:
                return self._send(200, {"error": str(e)})
        if self.path == "/api/stitch":
            try:
                body = self._body()
                files = [safe_name(f) for f in body.get("files", [])]
                if not files:
                    return self._send(200, {"error": "no clips selected"})
                tmp = os.path.join(HERE, "clips_tmp")
                shutil.rmtree(tmp, ignore_errors=True)
                os.makedirs(tmp)
                for i, f in enumerate(files):
                    shutil.copy2(os.path.join(COMFY_VIDEO_DIR, f), os.path.join(tmp, f"{i:03d}_{f}"))
                os.makedirs(MOVIE_DIR, exist_ok=True)
                out = os.path.join(MOVIE_DIR, f"movie_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
                r = subprocess.run(
                    ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                     "-File", STITCH_PS1, "-ClipDir", tmp, "-Out", out],
                    capture_output=True, text=True, timeout=900)
                shutil.rmtree(tmp, ignore_errors=True)
                if r.returncode != 0 or not os.path.isfile(out):
                    return self._send(200, {"error": (r.stderr or r.stdout or "stitch failed")[-500:]})
                return self._send(200, {"movie": os.path.basename(out)})
            except Exception as e:
                return self._send(200, {"error": str(e)})
        return self._send(404, {"error": "not found"})


PAGE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Anime Studio</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#0a0a12;--panel:#12121f;--neon:#00ffc8;--pink:#ff2d78;--txt:#d8d8e8;--dim:#7a7a90}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font-family:'Segoe UI',sans-serif;max-width:980px;margin:0 auto;padding:24px}
h1{font-size:26px;letter-spacing:2px;color:var(--neon);text-shadow:0 0 12px #00ffc855;margin-bottom:4px}
h1 span{color:var(--pink);text-shadow:0 0 12px #ff2d7855}
.sub{color:var(--dim);margin-bottom:20px;font-size:13px}
.panel{background:var(--panel);border:1px solid #23233a;border-radius:12px;padding:18px;margin-bottom:18px}
.step{color:var(--pink);font-size:12px;letter-spacing:2px;margin-bottom:10px}
#drop{border:2px dashed #2e2e4a;border-radius:10px;min-height:160px;display:flex;align-items:center;justify-content:center;cursor:pointer;overflow:hidden;transition:.2s}
#drop:hover{border-color:var(--neon)}
#drop img{max-width:100%;max-height:300px;border-radius:8px}
#drop .hint{color:var(--dim);padding:30px;text-align:center}
textarea{width:100%;background:#0d0d18;border:1px solid #2e2e4a;border-radius:10px;color:var(--txt);padding:12px;font-size:15px;min-height:74px;resize:vertical;outline:none}
textarea:focus{border-color:var(--neon)}
.examples{margin-top:8px;display:flex;gap:8px;flex-wrap:wrap}
.examples button{background:#1a1a2e;border:1px solid #2e2e4a;color:var(--dim);border-radius:20px;padding:5px 12px;font-size:12px;cursor:pointer}
.examples button:hover{color:var(--neon);border-color:var(--neon)}
#go{width:100%;padding:18px;font-size:20px;letter-spacing:4px;font-weight:bold;background:linear-gradient(90deg,#00ffc8,#00b8ff);border:none;border-radius:12px;color:#001a14;cursor:pointer;transition:.2s}
#go:hover{filter:brightness(1.15)}
#go:disabled{filter:grayscale(.8);cursor:wait}
#status{margin-top:12px;text-align:center;color:var(--dim);min-height:22px;font-size:14px}
#status.err{color:var(--pink)}
#result video{width:100%;border-radius:10px;margin-top:10px}
.gal{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:12px}
.card{background:#0d0d18;border:1px solid #2e2e4a;border-radius:10px;padding:8px}
.card video{width:100%;border-radius:6px}
.card label{display:flex;gap:6px;align-items:center;font-size:12px;color:var(--dim);margin-top:6px;cursor:pointer;word-break:break-all}
.rowbtns{display:flex;gap:10px;margin-top:14px}
.rowbtns button{flex:1;padding:12px;border-radius:10px;border:1px solid var(--pink);background:transparent;color:var(--pink);font-size:15px;letter-spacing:2px;cursor:pointer}
.rowbtns button:hover{background:var(--pink);color:#fff}
#offline{display:none;background:#2a0d18;border:1px solid var(--pink);color:#ffb3cd;padding:12px;border-radius:10px;margin-bottom:18px;font-size:14px}
</style></head><body>
<h1>ANIME <span>STUDIO</span></h1>
<div class="sub">image in, motion prompt, clip out — all local on the 5090</div>
<div id="offline">ComfyUI engine is not running. Double-click <b>Anime Studio.bat</b> on the Desktop, or start D:\ComfyUI, then refresh.</div>

<div class="panel"><div class="step">STEP 1 — PICK A KEYFRAME IMAGE (OPTIONAL)</div>
<div id="drop"><div class="hint">Click or drop an image here<br><small>or leave this EMPTY for direct text-to-video from your words</small></div></div>
<input type="file" id="file" accept="image/png,image/jpeg" hidden>
<label style="display:flex;gap:8px;align-items:center;margin-top:10px;color:var(--dim);font-size:13px;cursor:pointer">
<input type="checkbox" id="paint"> no image? paint an anime keyframe first (SDXL) and animate it with AniSora instead of direct text-to-video</label>
</div>

<div class="panel"><div class="step">STEP 2 — WHAT SHOULD MOVE?</div>
<textarea id="prompt" placeholder="with an image: describe the MOTION — e.g. her hair blows in the wind as she turns and smiles, camera pushes in. Without an image: describe the whole shot — e.g. a samurai girl walking through a neon market in the rain"></textarea>
<div class="examples">
<button>hair and clothes blow in the wind, camera slowly pushes in</button>
<button>she turns toward the camera and smiles softly</button>
<button>walking forward through neon rain, puddles ripple</button>
<button>epic anime action, dynamic camera orbit, sparks fly</button>
</div></div>

<div class="panel"><div class="step">STEP 3 — MAKE IT MOVE (5 second clip)</div>
<button id="go">ANIMATE</button>
<div id="status"></div>
<div id="result"></div>
</div>

<div class="panel"><div class="step">YOUR CLIPS — check some, then make a movie</div>
<div class="gal" id="gal"></div>
<div class="rowbtns"><button id="stitch">MAKE MOVIE FROM CHECKED CLIPS</button></div>
<div id="movies"></div>
</div>

<script>
const $=id=>document.getElementById(id);
let imgB64=null;
$('drop').onclick=()=>$('file').click();
$('drop').ondragover=e=>e.preventDefault();
$('drop').ondrop=e=>{e.preventDefault();loadFile(e.dataTransfer.files[0])};
$('file').onchange=e=>loadFile(e.target.files[0]);
function loadFile(f){if(!f)return;const r=new FileReader();r.onload=()=>{imgB64=r.result.split(',')[1];$('drop').innerHTML='<img src="'+r.result+'">'};r.readAsDataURL(f)}
document.querySelectorAll('.examples button').forEach(b=>b.onclick=()=>$('prompt').value=b.textContent);

let t0=0,timer=null;
$('go').onclick=async()=>{
  if(!imgB64&&!$('prompt').value.trim()){setStatus('give me an image or some words, choom',true);return}
  $('go').disabled=true;$('result').innerHTML='';
  const paint=$('paint').checked;
  setStatus(imgB64?'sending to the engine...':(paint?'painting your keyframe from words first (about a minute), then animating...':'dreaming your shot directly from words...'));
  const r=await fetch('/api/animate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image_b64:imgB64,prompt:$('prompt').value,paint:paint})}).then(r=>r.json()).catch(e=>({error:e}));
  if(r.error){setStatus('error: '+r.error,true);$('go').disabled=false;return}
  t0=Date.now();poll(r.prompt_id);
};
function setStatus(msg,err){$('status').textContent=msg;$('status').className=err?'err':''}
async function poll(pid){
  timer=setInterval(async()=>{
    const s=await fetch('/api/status/'+pid).then(r=>r.json()).catch(()=>({state:'running'}));
    const el=Math.round((Date.now()-t0)/1000);
    if(s.state==='done'){clearInterval(timer);$('go').disabled=false;setStatus('done in '+el+'s');
      if(s.video)$('result').innerHTML='<video controls autoplay loop src="/clip?f='+encodeURIComponent(s.video)+'"></video>';
      loadGallery();
    }else if(s.state==='error'){clearInterval(timer);$('go').disabled=false;setStatus('error: '+s.error,true);
    }else{setStatus((s.state==='queued'?'waiting in queue':'cooking')+'... '+el+'s elapsed (first run loads 14 GB of model, give it a few minutes)')}
  },3000);
}
async function loadGallery(){
  const d=await fetch('/api/clips').then(r=>r.json());
  $('gal').innerHTML=d.clips.map(c=>'<div class="card"><video preload="metadata" controls src="/clip?f='+encodeURIComponent(c.name)+'"></video><label><input type="checkbox" value="'+c.name+'">'+c.name+'</label></div>').join('')||'<div style="color:var(--dim)">no clips yet — make your first one above</div>';
  $('movies').innerHTML=d.movies.length?'<div class="step" style="margin-top:16px">FINISHED MOVIES</div>'+d.movies.map(m=>'<video controls style="width:100%;border-radius:10px;margin-top:8px" src="/movie?f='+encodeURIComponent(m)+'"></video>').join(''):'';
}
$('stitch').onclick=async()=>{
  const files=[...document.querySelectorAll('#gal input:checked')].map(i=>i.value);
  if(!files.length){setStatus('check at least one clip first',true);return}
  $('stitch').disabled=true;$('stitch').textContent='STITCHING...';
  const r=await fetch('/api/stitch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({files})}).then(r=>r.json());
  $('stitch').disabled=false;$('stitch').textContent='MAKE MOVIE FROM CHECKED CLIPS';
  if(r.error){setStatus('stitch error: '+r.error,true)}else{setStatus('movie ready!');loadGallery()}
};
fetch('/api/health').then(r=>r.json()).then(h=>{if(!h.ok)$('offline').style.display='block'});
loadGallery();
</script></body></html>"""


if __name__ == "__main__":
    os.makedirs(MOVIE_DIR, exist_ok=True)
    print(f"Anime Studio UI -> http://127.0.0.1:{PORT}  (engine: {COMFY})")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
