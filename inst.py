from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # small model; downloads automatically
results = model.predict("https://ultralytics.com/images/bus.jpg")
print(results[0].boxes)
