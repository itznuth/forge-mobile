# F.O.R.G.E. Mobile

**Flexible Operations & Robotics Guided Environment**, rebuilt to run entirely in a phone's browser.

**▶ Live demo: https://itznuth.github.io/forge-mobile/** (open on a phone, tap **Start camera**)

F.O.R.G.E. started as my CSULB senior design project: a smart workbench on a Raspberry Pi 5 that detected tools with
YOLOv11, pointed at them with a laser on a pan-tilt mount, and took voice commands through Vosk. This version moves the
whole pipeline onto the phone. There is no server and no cloud inference: every camera frame is processed on the device.

## Features
- **Live tool detection:** a custom YOLO11n model trained on 7 workbench tools, running in onnxruntime-web (WASM).
- **Everyday mode:** tap the mode pill to switch to COCO-SSD (80 common objects) in TensorFlow.js (WebGL).
- **Find and guide:** say, type or tap "find wrench". FORGE locks on with a laser reticle, shows "pan left / up" hints,
  points an arrow to where it last saw the tool, and plays **sonar beeps** that speed up as the tool gets centred.
- **Tap to lock:** tap any box on the camera view to track that object; tap it again to release.
- **Bench watch:** prop the phone over your bench and tap **Save bench**. FORGE logs every tool that leaves and comes
  back with times and durations, and reminds you about tools that have been out too long, like the original workbench.
- **Voice in and out:** Web Speech API commands and spoken replies.
- **Snapshots:** save or share a photo with the detections drawn on it.
- **Works offline:** a service worker caches the app, libraries and model after the first visit. Installable to the
  home screen.
- **Settings:** confidence threshold, voice, sonar, box smoothing, reminder time, flashlight (where supported), front/back camera.

## Tools model
| | |
|---|---|
| Architecture | YOLO11n, 2.6M parameters, 640×640 input, exported to ONNX (opset 12) |
| Classes | allen keys, hammer, handsaw, pliers, screwdriver, tape measure, wrench |
| Validation | **mAP@50 98.1%** · mAP@50-95 94.5% · precision 96.6% · recall 96.2% |

## Pi → phone
| Original (Pi 5)              | Mobile                                                   |
|------------------------------|----------------------------------------------------------|
| Pi Camera + YOLOv11 (Python) | Phone camera + YOLO11n in onnxruntime-web (WASM)         |
| Laser on pan-tilt servos     | Laser reticle, direction hints, arrow and sonar beeps    |
| Vosk voice commands          | Web Speech API                                           |
| Piper / eSpeak TTS           | speechSynthesis                                          |
| SQLite event tables          | Local `tool_events` / `command_events` + bench check-out log |
| `FORGE_SIM=1`                | Automatic simulation mode when no model can load         |

## How it works
```
camera ─► letterbox 640² ─► YOLO11n (ONNX/WASM) ─► decode + NMS ─► box smoothing ─► tracker
                                                                                      │
                     guidance (reticle, hints, sonar) ◄─┬─ voice (commands / replies) ◄┤
                                                        └─ bench watch (check-out log) ◄┘
```
- `forge.js` does preprocessing (letterbox to 640, CHW float tensor), YOLO output decoding and per-class NMS,
  IoU-matched box smoothing, presence tracking, guidance, voice, bench watch and the event store.
- The object-fit: cover crop is mapped exactly, so boxes, taps and "left/right" directions match what is on screen.
- iOS Safari details handled: `playsinline` camera, speech and audio unlocked on the first tap, the microphone and
  speech output never used at the same time, WebGL → CPU fallback, single-threaded WASM (GitHub Pages can't send the
  headers multithreading needs).

## Commands
Type, tap or say (🎙):
`find wrench` · `find allen key` · `how many screwdrivers` · `what do you see` · `save bench` · `what's missing` ·
`stop watching` · `snapshot` · `everyday mode` · `tools mode` · `flashlight on` · `unlock` · `help`

## Run it yourself
It's a static site: no build step, no backend.
1. Fork or clone this repo.
2. **Settings → Pages → Deploy from a branch → `main` / root**.
3. Open `https://<your-username>.github.io/forge-mobile/` on your phone.

For local testing, serve the folder over `localhost` (for example `python3 -m http.server`). Cameras only work on
https or localhost.

## Files
- `index.html`: UI
- `forge.js`: camera, models, decoding, tracking, guidance, voice, bench watch, snapshots, settings
- `sw.js`: offline cache
- `model/best.onnx` + `model/classes.json`: the tools model and its class order
- `TRAINING.md`: retrain the model from a Roboflow dataset in free Google Colab and export it to ONNX

## License note
The tools model was trained with Ultralytics YOLO, which is AGPL-3.0.
