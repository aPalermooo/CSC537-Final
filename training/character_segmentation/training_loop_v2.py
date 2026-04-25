import os

from ultralytics import YOLO


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

data_dir = os.path.join(PROJECT_ROOT, 'models', 'character_segmentation', 'ver_2', 'data.yaml')
project_dir = os.path.join(PROJECT_ROOT, 'results', 'character_segmentation', 'training_loop_v1')

model = YOLO('yolov8n.pt')

if __name__ == '__main__':
    model.train(
        data=data_dir,
        epochs=300,
        imgsz=320,
        batch=16,
        augment=True,
        project=project_dir,
        name='yolov8_v1',
        resume=True,
    )
