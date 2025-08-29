
import argparse
import os
import sys
import yaml
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F

# Add the project's root directory to the Python path to allow for module imports
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # This should be /home/user/fdm/halibut-mtl
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Now we can import our custom modules and the correct LetterBox class
from mtl.multi_task_model import MultiTaskModel
from ultralytics.data.augment import LetterBox
from ultralytics.engine.results import Results
from ultralytics.utils import ops

def load_model(weights_path, model_yaml_path, device):
    """Loads the MultiTaskModel and weights."""
    print(f"Loading model structure from: {model_yaml_path}")
    model = MultiTaskModel(cfg=model_yaml_path, ch=3)
    
    print(f"Loading weights from: {weights_path}")
    ckpt = torch.load(weights_path, map_location=device)
    state_dict = ckpt['model'].float().state_dict()
    state_dict.pop('cls_criterion.weight', None)
    
    model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model.eval()
    model.fuse() 
    
    print("Model loaded successfully.")
    return model

def preprocess_image(image_path, device, new_shape=(640, 640)):
    """
    Loads an image, applies letterbox padding using the ultralytics LetterBox class,
    and converts it to a tensor that matches the training-time preprocessing.
    """
    # 1. Load image with OpenCV
    original_image = cv2.imread(str(image_path))
    
    # 2. Apply LetterBox transformation
    # This is the standard ultralytics preprocessing function.
    letterbox = LetterBox(new_shape, auto=True, stride=32)
    image = letterbox(image=original_image)
    
    # 3. Convert from HWC (Height, Width, Channels) to CHW (Channels, Height, Width) format
    image = image.transpose((2, 0, 1))
    # Convert from BGR (OpenCV default) to RGB
    image = image[::-1]
    
    # 4. Convert to a contiguous torch tensor
    image = torch.from_numpy(image.copy()).to(device)
    # Normalize from [0, 255] to [0.0, 1.0]
    image = image.float() / 255.0
    # Add batch dimension if it's missing
    if image.ndimension() == 3:
        image = image.unsqueeze(0)
        
    return image, original_image

def postprocess_and_display_results(
    det_output, 
    cls_output, 
    source_path, 
    img_tensor,
    original_image,
    cls_class_names, 
    det_class_names,
    conf_threshold=0.25,
    iou_threshold=0.45
):
    """
    Processes model output, prints classification, and saves detection visualization.
    """
    # --- Classification Results ---
    print("\n" + "="*30)
    print("CLASSIFICATION RESULTS")
    print("="*30)
    
    cls_logits = cls_output[1] if isinstance(cls_output, tuple) else cls_output
    probabilities = F.softmax(cls_logits, dim=1)
    top3_probs, top3_indices = torch.topk(probabilities, 3, dim=1)
    
    for i in range(3):
        prob = top3_probs[0, i].item()
        class_idx = top3_indices[0, i].item()
        class_name = cls_class_names.get(class_idx, f"Unknown Class {class_idx}")
        print(f"  {i+1}. {class_name}: {prob:.4f}")

    # --- Detection Results ---
    print("\n" + "="*30)
    print("DETECTION RESULTS")
    print("="*30)

    preds = ops.non_max_suppression(
        prediction=det_output,
        conf_thres=conf_threshold,
        iou_thres=iou_threshold,
        agnostic=False,
        max_det=300,
        classes=None
    )
    
    pred_for_image = preds[0]

    if pred_for_image is None or len(pred_for_image) == 0:
        print("No objects detected.")
        output_path = "detection_result.jpg"
        cv2.imwrite(output_path, original_image)
        print(f"Result image saved to: {output_path}")
        return

    # Scale boxes from the model's input size back to the original image size
    pred_for_image[:, :4] = ops.scale_boxes(img_tensor.shape[2:], pred_for_image[:, :4], original_image.shape)

    # Create a Results object to handle plotting
    results = Results(
        orig_img=original_image,
        path=str(source_path),
        names=det_class_names,
        boxes=pred_for_image
    )

    output_path = "detection_result.jpg"
    annotated_image = results.plot()
    
    cv2.imwrite(output_path, annotated_image)
    
    print(f"Detection results saved to: {output_path}")
    print(f"Found {len(pred_for_image)} objects.")

def main(args):
    """Main function to run the prediction."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: Source file not found at {source_path}")
        return

    with open(args.data_yaml, 'r') as f:
        data_cfg = yaml.safe_load(f)
    cls_class_names = data_cfg.get('names_cls', {})
    det_class_names = data_cfg.get('names', {})

    model = load_model(args.weights, args.model_yaml, device)

    # Preprocess the image using the corrected LetterBox method
    image_tensor, original_image = preprocess_image(source_path, device)
    
    with torch.no_grad():
        det_output, cls_output = model(image_tensor)

    # Post-process and display results
    postprocess_and_display_results(
        det_output,
        cls_output,
        source_path,
        image_tensor,
        original_image,
        cls_class_names,
        det_class_names,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-task inference.")
    parser.add_argument('--weights', type=str, default='/home/user/fdm/halibut-mtl/result/y8n_merged_weighted_class_300/weights/best.pt', help='Path to the trained model weights (.pt file).')
    parser.add_argument('--source', type=str, default='/home/user/fdm/halibut-mtl/result/F02_U01_O2177_D2022-09-15_L365_W0585_S3_R10_B01_I00061319.jpg', help='Path to the source image.')
    parser.add_argument('--model-yaml', type=str, default='/home/user/fdm/halibut-mtl/yaml/model/y8n_merged.yaml', help='Path to the model definition yaml file.')
    parser.add_argument('--data-yaml', type=str, default='/home/user/fdm/halibut-mtl/yaml/data/merged.yaml', help='Path to the data configuration yaml file.')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for detection.')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU threshold for NMS.')
    
    args = parser.parse_args()
    main(args)
