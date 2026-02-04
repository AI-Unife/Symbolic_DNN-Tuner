import cv2
from pytorchyolo import detect, models

model = models.load_model(
    "config/yolov3/yolov3-aerial.cfg", "checkpoints/yolov3_ckpt_42.pth"
)
img_path = "/home/miele/vision-models-pytorch/downloaded_datasets/aerial_maritime/train/DJI_0320_JPG.rf.1714d3480e0cdfd770e26b30ca926364.jpg"

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

boxes = detect.detect_image(model, img)

print(boxes)
