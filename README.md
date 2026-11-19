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

## Two detection modes (tap the mode pill to switch)
- **Tools:** the FORGE workbench model, YOLO11n trained on 7 tools: allen keys, hammer, handsaw, pliers,
  screwdriver, tape measure, wrench. Validation: **mAP@50 98.1%, mAP@50-95 94.5%, precision 96.6%, recall 96.2%**.
  Runs in `onnxruntime-web` (WASM).
- **80 objects:** pretrained COCO-SSD in TensorFlow.js (WebGL), for everyday things like cups, phones and laptops.

## Run it on your iPhone

1. Create a GitHub repo (e.g. `forge-mobile`) and upload these files.
2. Repo **Settings → Pages → Deploy from branch → main / root**.
3. Open `https://<your-username>.github.io/forge-mobile/` in Safari → **Start camera** → allow camera.
4. Optional: **Share → Add to Home Screen** to get an app icon.

If no model can load, it falls back to **simulation mode** so the UI still works.
To retrain the tools model, follow **TRAINING.md** and replace `model/best.onnx` + `model/classes.json`.

## Commands
Type, tap or say (🎙): `find wrench` · `find allen key` · `find tape measure` · `what do you see` · `status` · `auto` · `unlock`

## Files
- `index.html`: UI
- `forge.js`: camera, preprocessing (letterbox 640), YOLO postprocess + NMS, tracker, guidance, voice, event store
- `model/`: `best.onnx` + `classes.json`
- `TRAINING.md`: retrain from the Roboflow dataset in Colab and export to ONNX
