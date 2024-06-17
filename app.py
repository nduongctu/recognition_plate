import os
import torch
from flask import Flask, request, render_template, Response
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading YOLO models...")
model_lp = YOLO('model/best_KBS_8n.pt', task='detect')
model_char = YOLO("model/best_char_8n.pt", task='detect')

model_lp.to(device)
model_char.to(device)
print("YOLO models loaded successfully")

CHAR_THRES = 0.7

def format_LP(chars, char_centers):
    if not chars:
        return []

    x = [c[0] for c in char_centers]
    y = [c[1] for c in char_centers]
    y_mean = np.mean(y)

    # Kiểm tra xem ký tự có trên cùng một dòng không
    if max(y) - min(y) < 10:
        return [i for _, i in sorted(zip(x, chars))]

    # Sắp xếp các ký tự theo tọa độ X
    sorted_chars = [i for _, i in sorted(zip(x, chars))]
    y = [i for _, i in sorted(zip(x, y))]

    # Phân chia ký tự thành hai dòng dựa trên tọa độ Y trung bình
    first_line = [i for i in range(len(chars)) if y[i] < y_mean]
    second_line = [i for i in range(len(chars)) if y[i] >= y_mean]

    # Ghép các ký tự đã sắp xếp theo thứ tự dòng và thêm dấu gạch ngang giữa hai dòng
    return [sorted_chars[i] for i in first_line] + ['-'] + [sorted_chars[i] for i in second_line]

def detect_license_plate(video_path):
    cap = cv2.VideoCapture(video_path)
    track_history = defaultdict(list)
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model_lp.track(frame, persist=True)
        boxes = results[0].boxes.xywh.to(device)
        track_ids = results[0].boxes.id.int().tolist()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box

            x_min = int(max(x - w / 2, 0))
            y_min = int(max(y - h / 2, 0))
            x_max = int(min(x + w / 2, frame.shape[1]))
            y_max = int(min(y + h / 2, frame.shape[0]))
            roi = frame[y_min:y_max, x_min:x_max]

            annotated_frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            detected_texts = []
            results_char = model_char(roi)
            chars = []
            char_centers = []
            for result in results_char:
                boxes = result.boxes.data.tolist()
                for box in boxes:
                    x_min_char, y_min_char, x_max_char, y_max_char, confidence, cls = box
                    if confidence > CHAR_THRES:
                        x_min_char, y_min_char, x_max_char, y_max_char = map(int, [x_min_char, y_min_char, x_max_char, y_max_char])
                        cls = int(cls)
                        chars.append(model_char.names[cls])

                        center_x = (x_min_char + x_max_char) // 2
                        center_y = (y_min_char + y_max_char) // 2
                        char_centers.append((center_x, center_y))

            if chars:
                detected_texts.append(''.join(format_LP(chars, char_centers)))
                text = detected_texts[0]
                cv2.putText(annotated_frame, text, (x_min, y_min - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 4)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', annotated_frame)[1].tobytes() + b'\r\n')

        frame_count += 1

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file:
        filename = 'test.mp4'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return Response(detect_license_plate(filepath), mimetype='multipart/x-mixed-replace; boundary=frame')

    return "Upload failed", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5505, debug=True)