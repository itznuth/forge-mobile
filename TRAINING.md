# Retrain the FORGE detector and export it for the phone (about 30 min, free)

The original `best_n.pt` was lost with the Pi, so you retrain it from your Roboflow dataset.
Use Google Colab's free GPU.

1. In Roboflow, open your FORGE dataset → **Versions → Export → YOLOv11 format → "Show download code"**.
   Copy the snippet (it has your API key, workspace, project and version).
2. Go to colab.research.google.com → New notebook → **Runtime → Change runtime type → T4 GPU**.
3. Run these cells, pasting your Roboflow snippet into cell 2:

```python
# cell 1
!pip install -q ultralytics roboflow onnx onnxslim
```

```python
# cell 2  — paste YOUR Roboflow snippet here, e.g.:
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")
dataset = project.version(1).download("yolov11")
```

```python
# cell 3  — train the nano model (fast enough for a phone)
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.train(data=f"{dataset.location}/data.yaml", epochs=100, imgsz=640, batch=16, patience=25)
```

```python
# cell 4  — export for the browser + save class names in the right order
import json, shutil
best = YOLO("runs/detect/train/weights/best.pt")
print(best.val().box.map50)          # your new mAP@50 — put THIS number on your resume
best.export(format="onnx", imgsz=640, opset=12, simplify=True)
json.dump([best.names[i] for i in range(len(best.names))], open("classes.json", "w"))
from google.colab import files
files.download("runs/detect/train/weights/best.onnx")
files.download("classes.json")
files.download("runs/detect/train/weights/best.pt")   # keep a backup this time
```

4. Put `best.onnx` and `classes.json` into `model/` in this repo and push.
   `*.onnx` is in `.gitignore` so weights aren't committed by accident; when you deliberately want the
   model live on GitHub Pages, add it explicitly: `git add -f model/best.onnx` (a YOLO11n export is ~10 MB,
   well under GitHub's 100 MB file limit).

**Tip:** If you only want the 3 demo tools and the dataset has more classes, that's fine.
The app makes a button for every class in `classes.json`.
