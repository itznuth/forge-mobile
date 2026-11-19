# F.O.R.G.E. Mobile

**Flexible Operations & Robotics Guided Environment**, rebuilt to run entirely on an iPhone.

The original F.O.R.G.E. was a CSULB senior design smart workbench (Raspberry Pi 5, YOLOv11, laser pan-tilt,
Vosk voice, FastAPI + React dashboard). This version moves the whole pipeline onto the phone:

| Original (Pi 5)              | Mobile                                              |
|------------------------------|-----------------------------------------------------|
| Pi Camera + YOLOv11 (Python) | iPhone camera + YOLOv11 ONNX in onnxruntime-web (WASM) |
| Laser on pan-tilt servos     | On-screen laser reticle + direction arrow           |
| Vosk wake-word voice         | Web Speech API ("find wrench", "what do you see")   |
| Piper / eSpeak TTS           | speechSynthesis                                     |
| SQLite event tables          | Local `tool_events` / `command_events` store        |
| `FORGE_SIM=1`                | Automatic simulation mode when no model is present  |

Everything runs on-device: no server, no cloud inference.

## Run it on your iPhone

1. Create a GitHub repo (e.g. `forge-mobile`) and upload these files.
2. Repo **Settings → Pages → Deploy from branch → main / root**.
3. Open `https://<your-username>.github.io/forge-mobile/` in Safari → **Start camera** → allow camera.
4. Optional: **Share → Add to Home Screen** to get an app icon.

With no model in `model/`, it runs in **simulation mode** so you can try the UI right away.
To get real detection, follow **TRAINING.md** and drop `best.onnx` + `classes.json` into `model/`.

## Commands
Buttons or voice (🎙): `find wrench` · `find pliers` · `find screwdriver` · `what do you see` · `status` · `auto` · `unlock`

## Files
- `index.html`: UI
- `forge.js`: camera, preprocessing (letterbox 640), YOLO postprocess + NMS, tracker, guidance, voice, event store
- `model/`: `best.onnx` + `classes.json`
- `TRAINING.md`: retrain from the Roboflow dataset in Colab and export to ONNX
