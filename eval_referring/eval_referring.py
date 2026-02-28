import json
import argparse
import numpy as np
import re
from collections import defaultdict


def parse_bbox(bbox_str):
    """
    Parse bbox string like '{<16><21><79><76>}' to [x1, y1, x2, y2]
    Values are normalized coordinates (0-100)
    """
    if not bbox_str or bbox_str.strip() == '':
        return None
    
    # Extract numbers from the format {<x1><y1><x2><y2>}
    numbers = re.findall(r'<(\d+)>', bbox_str)
    
    if len(numbers) != 4:
        return None
    
    try:
        bbox = [float(num) for num in numbers]
        return bbox
    except:
        return None


def compute_iou(bbox1, bbox2):
    """
    Compute IoU between two bboxes [x1, y1, x2, y2]
    Returns: iou, intersection_area, union_area
    """
    if bbox1 is None or bbox2 is None:
        return 0.0, 0.0, 0.0
    
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Compute intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    intersection_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    
    # Compute union
    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = bbox1_area + bbox2_area - intersection_area
    
    # Compute IoU
    if union_area > 0:
        iou = intersection_area / union_area
    else:
        iou = 0.0
    
    return iou, intersection_area, union_area


def evaluate_referring(gt_path, pred_path, output_path=None):
    """
    Evaluate referring expression comprehension task
    
    Args:
        gt_path: Path to ground truth JSON file
        pred_path: Path to prediction JSON file
        output_path: Optional path to save detailed results
    """
    print(f"Loading ground truth from: {gt_path}")
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    print(f"Loading predictions from: {pred_path}")
    with open(pred_path, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)
    
    # Build index by image_id for both GT and predictions
    gt_dict = {}
    for item in gt_data:
        image_id = item['image_id']
        if image_id not in gt_dict:
            gt_dict[image_id] = []
        gt_dict[image_id].append(item)
    
    pred_dict = {}
    for item in pred_data:
        image_id = item['image_id']
        if image_id not in pred_dict:
            pred_dict[image_id] = []
        pred_dict[image_id].append(item)
    
    print(f"\nGround truth samples: {len(gt_data)}")
    print(f"Ground truth unique images: {len(gt_dict)}")
    print(f"Prediction samples: {len(pred_data)}")
    print(f"Prediction unique images: {len(pred_dict)}")
    
    # Evaluation metrics
    iou_thresholds = [0.5, 0.7]
    count_at_threshold = {t: 0 for t in iou_thresholds}
    
    total_iou = 0.0
    total_intersection = 0.0
    total_union = 0.0
    
    total_samples = 0
    valid_samples = 0
    missing_samples = 0
    invalid_bbox_samples = 0
    
    detailed_results = []
    
    # Iterate through all GT samples
    for gt_item in gt_data:
        image_id = gt_item['image_id']
        question_id = gt_item['question_id']
        gt_bbox_str = gt_item['ground_truth']
        
        # Parse GT bbox
        gt_bbox = parse_bbox(gt_bbox_str)
        if gt_bbox is None:
            print(f"Warning: Invalid GT bbox for {image_id}, question_id {question_id}: {gt_bbox_str}")
            # Skip samples with invalid GT bbox (don't count in total)
            continue
        
        total_samples += 1
        
        # Find corresponding prediction by matching image_id first, then question_id/question
        pred_item = None
        if image_id in pred_dict:
            # Try to match by question_id first
            for p in pred_dict[image_id]:
                if p['question_id'] == question_id:
                    pred_item = p
                    break
            
            # If not found, try to match by question text
            if pred_item is None:
                for p in pred_dict[image_id]:
                    if p['question'] == gt_item['question']:
                        pred_item = p
                        break
        
        if pred_item is None:
            # Missing prediction - treat as IoU = 0
            # This handles cases where:
            # 1. The image_id is not in prediction file
            # 2. The image_id exists but this specific question is missing
            missing_samples += 1
            iou = 0.0
            intersection = 0.0
            union = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
            pred_bbox_str = "MISSING"
            pred_bbox = None
        else:
            # Parse predicted bbox
            pred_bbox_str = pred_item['ground_truth']  # In pred file, this field contains prediction
            pred_bbox = parse_bbox(pred_bbox_str)
            
            if pred_bbox is None:
                # Invalid prediction bbox - treat as IoU = 0
                invalid_bbox_samples += 1
                iou = 0.0
                intersection = 0.0
                union = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
            else:
                # Compute IoU
                iou, intersection, union = compute_iou(gt_bbox, pred_bbox)
                valid_samples += 1
        
        # Accumulate metrics
        total_iou += iou
        total_intersection += intersection
        total_union += union
        
        # Count correct predictions at different thresholds
        for threshold in iou_thresholds:
            if iou >= threshold:
                count_at_threshold[threshold] += 1
        
        # Store detailed result
        detailed_results.append({
            'image_id': image_id,
            'question_id': question_id,
            'question': gt_item['question'],
            'gt_bbox': gt_bbox_str,
            'pred_bbox': pred_bbox_str,
            'iou': float(iou),
            'intersection': float(intersection),
            'union': float(union)
        })
    
    # Compute final metrics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total GT samples (valid GT bbox): {total_samples}")
    print(f"Valid predictions (valid bbox): {valid_samples}")
    print(f"Missing predictions (IoU=0): {missing_samples}")
    print(f"Invalid prediction bbox (IoU=0): {invalid_bbox_samples}")
    print()
    print("NOTE: Missing and invalid predictions are counted as IoU=0")
    print("      in all metrics calculation.")
    print("      GT samples with invalid bbox are excluded from evaluation.")
    print()
    
    # Accuracy at different IoU thresholds
    for threshold in iou_thresholds:
        acc = count_at_threshold[threshold] / total_samples * 100
        print(f"Acc@IoU_{threshold}: {acc:.2f}% ({count_at_threshold[threshold]}/{total_samples})")
    
    # Mean IoU (average of all samples, including missing/invalid as 0)
    mean_iou = total_iou / total_samples * 100
    print(f"meanIoU: {mean_iou:.2f}% (sum of all IoU / total GT samples)")
    
    # Cumulative IoU
    if total_union > 0:
        cum_iou = total_intersection / total_union * 100
        print(f"cumIoU: {cum_iou:.2f}% (total intersection / total union)")
    else:
        print(f"cumIoU: N/A (total union is 0)")
    
    print("="*60)
    
    # Save detailed results if output path is provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'total_samples': total_samples,
                    'valid_samples': valid_samples,
                    'missing_samples': missing_samples,
                    'invalid_bbox_samples': invalid_bbox_samples,
                    'acc_at_iou_0.5': count_at_threshold[0.5] / total_samples * 100,
                    'acc_at_iou_0.7': count_at_threshold[0.7] / total_samples * 100,
                    'mean_iou': mean_iou,
                    'cumulative_iou': total_intersection / total_union * 100 if total_union > 0 else 0
                },
                'detailed_results': detailed_results
            }, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {output_path}")
    
    return {
        'total_samples': total_samples,
        'acc@0.5': count_at_threshold[0.5] / total_samples * 100,
        'acc@0.7': count_at_threshold[0.7] / total_samples * 100,
        'mean_iou': mean_iou,
        'cum_iou': total_intersection / total_union * 100 if total_union > 0 else 0
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Referring Expression Comprehension')
    parser.add_argument('--gt', type=str, required=True, help='Path to ground truth JSON file')
    parser.add_argument('--pred', type=str, required=True, help='Path to prediction JSON file')
    parser.add_argument('--output', type=str, default=None, help='Path to save detailed results')
    
    args = parser.parse_args()
    
    evaluate_referring(args.gt, args.pred, args.output)

