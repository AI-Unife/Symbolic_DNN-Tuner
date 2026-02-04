from pytorchyolo import models

def get_yolov3(weights_path: str):
    model = models.load_model(
        'models/other_nets/detection/coco/models/yolov3/yolov3.cfg',
        weights_path
        )
    
    return model