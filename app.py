import torch
import cv2
import numpy as np
import uvicorn
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# thong tin xac thuc co ban
USERNAME = "admin"
PASSWORD = "123"

# Khoi tao YOLO models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# Load models
model_lp = YOLO("model/best_KBS_8n.pt")
model_char = YOLO("model/best_char8n.pt")

# chuyen cac mo hinh sang cpu hoac cuda
model_lp = model_lp.to(device)
model_char = model_char.to(device)

model_lp.fuse()
model_char.fuse()

# cac mo  hinh Pydantic cho các payload yeu cau & phan hoi
class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class DetectedObject(BaseModel):
    label: str
    bounding_box: BoundingBox
    chars: List[str]

def format_LP(chars, char_centers):
    if not chars:
        return []

    char_centers = np.array(char_centers)
    y_mean = char_centers[:, 1].mean()

    sorted_indices = np.argsort(char_centers[:, 0])
    sorted_chars = np.array(chars)[sorted_indices]

    first_line = sorted_chars[char_centers[sorted_indices, 1] < y_mean]
    second_line = sorted_chars[char_centers[sorted_indices, 1] >= y_mean]

    return list(first_line) + ['-'] + list(second_line) if len(second_line) > 0 else list(first_line)

def warm_up_model(model, size=(640, 640)):
    dummy_input = torch.rand(1, 3, *size).to(device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

# khoi dong truoc cac mo hinh truoc khi xu ly cac khung hinh thuc te
print("Warming up LP model...")
warm_up_model(model_lp)
print("Warming up char model...")
warm_up_model(model_char)

# Route to serve index.html
@app.get("/")
async def read_index():
    return FileResponse("templates/index_draw.html")

# ham process_frame
def process_frame(frame_img):
    results = model_lp(frame_img, conf=0.4, iou=0.2)
    detected_objects = []

    if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.to(device)

        # Chuẩn bị batch của các biển số
        plates = []
        plate_info = []

        for box in boxes:
            x, y, w, h = box.xywh[0].tolist()
            cls = box.cls.item()
            label = model_lp.names[int(cls)]

            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            plate = frame_img[y1:y2, x1:x2]
            plates.append(plate)
            plate_info.append((label, BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)))

        # Xử lý các biển số
        if plates:
            # Chuẩn bị batch input
            batch = torch.stack([torch.from_numpy(cv2.resize(p, (640, 640))).permute(2, 0, 1).float().div(255.0) for p in plates]).to(device)

            # Sử dụng batch processing
            results_plates = model_char(batch, conf=0.4, iou=0.2, agnostic_nms=True)

            for idx, ((label, bounding_box), result_plate) in enumerate(zip(plate_info, results_plates)):
                detected_chars = []
                char_centers = []

                if result_plate.boxes is not None and len(result_plate.boxes) > 0:
                    boxes_char = result_plate.boxes.to(device)
                    for box_char in boxes_char:
                        x_char, y_char, w_char, h_char = box_char.xywh[0].tolist()
                        cls_char = box_char.cls.item()
                        label_char = model_char.names[int(cls_char)]
                        detected_chars.append(label_char)

                        center_x = int(x_char)
                        center_y = int(y_char)
                        char_centers.append((center_x, center_y))

                detected_texts = [''.join(format_LP(detected_chars, char_centers))] if detected_chars else []

                obj = DetectedObject(
                    label=label,
                    bounding_box=bounding_box,
                    chars=detected_texts
                )
                detected_objects.append(obj)

    return detected_objects

# Route de xu ly khung hinh
@app.post("/process_frame", response_model=List[DetectedObject])
async def process_frame_api(frame: UploadFile = File(...), username: str = USERNAME, password: str = PASSWORD):
    if username != USERNAME or password != PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized")

    frame_data = await frame.read()
    frame_np = np.frombuffer(frame_data, np.uint8)
    frame_img = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)

    loop = asyncio.get_event_loop()
    detected_objects = await loop.run_in_executor(None, process_frame, frame_img)

    return detected_objects

if __name__ == "__main__":
    uvicorn.run(app, host="192.168.10.9", port=6066)
    # uvicorn.run(app, host="192.168.10.8", port=6066, ssl_certfile="server.crt", ssl_keyfile="server.key")
