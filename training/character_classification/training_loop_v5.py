import os

from ultralytics import YOLO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

data_dir = os.path.join(PROJECT_ROOT, 'datasets', 'character_classification', 'output')
project_dir = os.path.join(PROJECT_ROOT, 'results', 'character_classification', 'training_loop_v5')

model = YOLO('yolov8n-cls.pt')

if __name__ == '__main__':
    model.train(
        data=data_dir,
        epochs=300,
        imgsz=16,
        batch=256,
        augment=True,
        project=project_dir,
        name='yolov8_cls_v1',
        device=0,
    )