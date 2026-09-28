# AnimeStudio — local anime movie pipeline

Make full anime-style movies and stories on the 5090, fully local.

## The pipeline

1. **Write the story** — break it into shots (a few seconds each).
2. **Keyframes** — generate one anime-style image per shot:
   - Flux 2 dev in ComfyUI (`D:\ComfyUI`), or
   - SDXL studio (`D:\ZaryaArt`, Desktop "Zarya Art Studio.bat").
   Reuse the same character description/seed across shots for consistency.
3. **Animate** — ComfyUI workflow **AniSora_Anime_I2V** (in the workflow sidebar):
   - Load your keyframe image, write a motion prompt, queue.
   - AniSora V3.2 = Bilibili's anime-specialized Wan 2.2 I2V finetune.
   - 4-step lightx2v LoRAs are wired in = fast generations.
   - Output lands in `D:\ComfyUI\output\video\`.
4. **Assemble** — drop clips into `clips\` here (name them `010_x.mp4`, `020_y.mp4` for order), then:
   ```powershell
   .\stitch.ps1                          # -> movie.mp4 at 720p24
   .\stitch.ps1 -Music soundtrack.mp3    # with a music track
   ```
5. **Voices (later)** — ZaryaVoice Chatterbox server (:8765) can clone narrator/character voices; pipe lines to WAV and mux with ffmpeg.

## Models installed

| File | Role |
|---|---|
| `Wan2_2-I2V_AniSoraV3_2_HIGH/LOW_14B_fp8_e4m3fn_scaled_KJ.safetensors` | anime image-to-video (diffusion_models) |
| `Wan_2_2_I2V_A14B_HIGH/LOW_lightx2v_4step_lora_rank_64_fp16.safetensors` | 4-step speed LoRAs (loras) |
| `hunyuanvideo1.5_480p_t2v_cfg_distilled_fp8_scaled.safetensors` (+ qwen/byt5 encoders + VAE) | text-to-video words mode |

Shared: `umt5_xxl_fp8_e4m3fn_scaled` text encoder + `wan_2.1_vae`.
