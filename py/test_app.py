import os 
import pytest 
from ultralytics import YOLO

MODEL_PATH = "runs/detect/face-eye-detector/weights/best.pt"
OUTPUT_FOLDER = "results"

@pytest.fixture(scope="module")
def model():
    """Load YOLO model once for all tests."""
    return YOLO(MODEL_PATH)

def test_model_loads(model):
    """Test that YOLO model loads correctly."""
    assert hasattr(model, "names"), "Model should have 'names' attribute"
    assert len(model.names) > 0, "Model should contain class names"

def test_results_folder_exists():
    """Test that results folder exists or can be created."""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    assert os.path.exists(OUTPUT_FOLDER), "Results folder should exist"

def test_process_invalid_image(model):
    """Test that processing invalid image path doesn't crash."""
    with pytest.raises(FileNotFoundError):
        model.predict(source="invalid_path.jpg")

def test_prediction_output(model, tmp_path):
    """Test that model can run inference on a sample image."""

    import numpy as np, cv2
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    temp_file = tmp_path / "test.jpg"
    cv2.imwrite(str(temp_file), dummy_image)

    results = model.predict(source=str(temp_file), conf=0.25)
    assert len(results) > 0, "Model should return at least one result"