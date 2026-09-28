# Anime Studio — Local AI Anime Video Pipeline

A single-file web app that turns images or text into short anime clips on a local GPU, then stitches the chosen clips into a movie. It uses **ComfyUI's HTTP API** as the render engine and hides the node graphs behind a simple UI: pick an image, describe the motion, press **Animate**.

## Features

- **Three ways to start a shot**
  - **Image → video**: upload a keyframe. It is uploaded to ComfyUI and animated with the *AniSora V3.2* anime fine-tune of Wan 2.2 14B image-to-video, using 4-step lightx2v speed LoRAs
  - **Text → keyframe → video**: paints an anime keyframe with SDXL (via the companion `zarya-image-studio` CLI), then animates it
  - **Text → video**: direct text-to-video with HunyuanVideo 1.5 (480p)
- **Automatic resolution fitting**: reads PNG/JPEG dimensions from the raw bytes (no PIL) and scales them to an ~832×480 pixel budget, keeping the aspect ratio
- **Job tracking**: polls the ComfyUI history and queue for queued, running, done or error states, and reports ComfyUI execution errors
- **Clip gallery and movie maker**: lists rendered clips, lets you pick clips in order and stitch them into a timestamped MP4
- **`stitch.ps1`**: first normalizes every clip to 1280×720 at 24 fps H.264 (so mixed sources don't break the concat), then joins them with FFmpeg; it can add a music track
- **Built-in HTTP server** (Python standard library `ThreadingHTTPServer`) with HTTP range requests for video scrubbing and filename sanitising on file routes

## How it works

```
Browser UI (:8500) ──▶ studio.py ──/prompt, /upload/image──▶ ComfyUI (:8188)
                           │  ◀──/history, /queue────────────┘
                           └──▶ stitch.ps1 (FFmpeg) ──▶ movies/movie_<timestamp>.mp4
```

Workflow graphs are built in Python as ComfyUI API-format JSON (`build_graph`, `build_t2v_graph`).

## Tech stack

Python 3 (standard library only) · ComfyUI API · Wan 2.2 I2V (AniSora V3.2) · HunyuanVideo 1.5 · SDXL · FFmpeg · PowerShell · HTML/CSS/JS

## Setup

1. Install [ComfyUI](https://github.com/comfyanonymous/ComfyUI) with the models listed in `NOTES_original.md` and start it on `127.0.0.1:8188`.
2. Install FFmpeg and add it to `PATH`.
3. Optional: configure paths with environment variables (see `.env.example`): `COMFY_URL`, `COMFY_VIDEO_DIR`, `SDXL_PY`, `SDXL_GEN`.
4. Run:

```powershell
python studio.py     # then open http://127.0.0.1:8500
```

Standalone stitching: `.\stitch.ps1 -ClipDir .\clips -Out movie.mp4 [-Music track.mp3]`

Generated keyframes, clips, movies and model weights are not included.

---

© 2026 Mathew Vasilescu. All rights reserved. See [LICENSE](LICENSE).
