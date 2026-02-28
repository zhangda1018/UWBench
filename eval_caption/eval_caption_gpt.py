"""
GPT-based evaluation for Image Captioning task using CLAIR Score
CLAIR: Uses GPT to assess semantic similarity between predicted and reference captions
"""

import json
import argparse
import numpy as np
import re
from tqdm import tqdm
import os
import sys

# 导入现有的模块
try:
    from key.chat_bots import ChatBots
    from key.models_config import get_model_config, get_available_model_names
    CHATBOTS_AVAILABLE = True
except ImportError as e:
    CHATBOTS_AVAILABLE = False
    print(f"WARNING: ChatBots not found: {e}")
    print("Please ensure key/chat_bots.py and key/models_config.py exist")


_CLAIR_PROMPT = """\
You are trying to tell if a candidate caption is describing the same image as a reference caption.
Candidate caption:
{candidate_caption}

Reference caption:
{reference_caption}

On a precise scale from 0 to 100, how likely is it that the candidate caption is \
describing the same image as the reference caption? (JSON format, with a key "score", \
value between 0 and 100, and a key "reason" with a string value.)
"""


def clair_score(chat_bots, candidate, reference, max_retries=3):
    """
    Compute CLAIR score using GPT via ChatBots
    
    Args:
        chat_bots: ChatBots instance
        candidate: Predicted caption
        reference: Ground truth caption
        max_retries: Maximum number of retries
        
    Returns:
        score (0-1), reason
    """
    if not CHATBOTS_AVAILABLE:
        return 0.0, "ChatBots not available"
    
    formatted_prompt = _CLAIR_PROMPT.format(
        candidate_caption=candidate,
        reference_caption=reference
    )
    
    for attempt in range(max_retries):
        try:
            # 使用 ChatBots.call 调用模型
            # call(txt, img=None, isMsg=False, test=False, system_prompt=None)
            result = chat_bots.call(txt=formatted_prompt, img=None, isMsg=False, test=False)
            
            if result is None:
                print(f"Warning: ChatBots returned None on attempt {attempt + 1}")
                continue
            
            # result = [content, prompt_tokens, completion_tokens]
            response_text = result[0]
            
            # Parse JSON response
            try:
                # Extract JSON object
                parsed = response_text.split("{")[1]
                parsed = "{" + parsed.split("}")[0] + "}"
                data = json.loads(parsed)
                score = float(data["score"])
                reason = data.get("reason", 'Unknown')
                return score / 100.0, reason
            except (json.JSONDecodeError, KeyError, IndexError):
                # Try to extract first number using regex
                parsed = re.findall(r"\d*\.?\d+", response_text)
                if len(parsed) > 0:
                    score = float(parsed[0])
                    if score < 1:
                        score *= 100
                    
                    # Look for reason
                    reason_match = re.findall(r"(?i)reason.*", response_text)
                    if len(reason_match) > 0:
                        reason = reason_match[0].strip()[len('reason'):].replace(':', '').strip()
                    else:
                        reason = 'Unknown'
                    
                    return score / 100.0, reason
                else:
                    print(f"Warning: Could not parse response: {response_text[:100]}...")
                    continue
        
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                continue
            else:
                return 0.0, f"Error: {str(e)}"
    
    return 0.0, "Failed after all retries"


def evaluate_caption_gpt(gt_path, pred_path, output_path=None, model_name="gpt-4o-mini", sample_size=None):
    """
    Evaluate image captioning using GPT-based CLAIR score
    
    Args:
        gt_path: Path to ground truth JSON file
        pred_path: Path to prediction JSON file
        output_path: Optional path to save detailed results
        model_name: Model name from models_config (default: gpt-4o-mini)
        sample_size: If set, only evaluate a random sample of this size
    """
    if not CHATBOTS_AVAILABLE:
        print("ERROR: ChatBots not available. Cannot run GPT-based evaluation.")
        print("Please ensure key/chat_bots.py and key/models_config.py exist")
        return None
    
    # 初始化 ChatBots
    try:
        model_config = get_model_config(model_name)
        chat_bots = ChatBots(model_config, max_try=6, do_log=False)
        print(f"Initialized ChatBots with model: {model_name}")
    except Exception as e:
        print(f"ERROR: Failed to initialize ChatBots: {e}")
        print(f"Available models: {get_available_model_names()}")
        return None
    
    print(f"Loading ground truth from: {gt_path}")
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    print(f"Loading predictions from: {pred_path}")
    with open(pred_path, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)
    
    # Build dictionaries indexed by image_id
    gt_dict = {}
    for item in gt_data:
        image_id = item['image_id']
        caption = item['ground_truth'].strip().replace('\n', ' ')
        gt_dict[image_id] = caption
    
    pred_dict = {}
    for item in pred_data:
        image_id = item['image_id']
        caption = item['ground_truth'].strip().replace('\n', ' ')  # prediction is in this field
        pred_dict[image_id] = caption
    
    print(f"\nGround truth images: {len(gt_dict)}")
    print(f"Prediction images: {len(pred_dict)}")
    
    # Prepare evaluation pairs
    eval_pairs = []
    missing_count = 0
    
    for image_id in gt_dict:
        if image_id in pred_dict:
            eval_pairs.append({
                'image_id': image_id,
                'reference': gt_dict[image_id],
                'candidate': pred_dict[image_id]
            })
        else:
            # Missing prediction
            missing_count += 1
            eval_pairs.append({
                'image_id': image_id,
                'reference': gt_dict[image_id],
                'candidate': ""
            })
    
    print(f"Total samples: {len(gt_dict)}")
    print(f"Valid predictions: {len(gt_dict) - missing_count}")
    print(f"Missing predictions: {missing_count}")
    
    # Sample if requested
    if sample_size and sample_size < len(eval_pairs):
        print(f"\nSampling {sample_size} samples for evaluation...")
        import random
        random.seed(42)
        eval_pairs = random.sample(eval_pairs, sample_size)
    
    # 检查是否有已完成的评测结果（断点续传）
    evaluated_images = set()
    existing_results = []
    
    if output_path and os.path.exists(output_path):
        print(f"\n发现已有结果文件: {output_path}")
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                if 'detailed_results' in existing_data:
                    existing_results = existing_data['detailed_results']
                    evaluated_images = {r['image_id'] for r in existing_results}
                    print(f"已加载 {len(existing_results)} 条已评测结果")
                    print(f"将跳过这些样本，继续评测剩余样本")
        except Exception as e:
            print(f"警告: 无法读取已有结果文件: {e}")
            print("将从头开始评测")
    
    # 过滤掉已评测的样本
    remaining_pairs = [p for p in eval_pairs if p['image_id'] not in evaluated_images]
    
    print(f"\n总样本数: {len(eval_pairs)}")
    print(f"已评测: {len(evaluated_images)}")
    print(f"待评测: {len(remaining_pairs)}")
    print(f"使用模型: {model_name}")
    print("="*60)
    
    # Evaluate remaining samples
    results = existing_results.copy()
    clair_scores = [r['clair_score'] for r in existing_results]
    
    # 准备临时结果文件（实时保存）
    temp_output_path = output_path + '.tmp' if output_path else None
    
    for pair in tqdm(remaining_pairs, desc="Computing CLAIR scores"):
        score, reason = clair_score(
            chat_bots,
            pair['candidate'],
            pair['reference']
        )
        
        result_item = {
            'image_id': pair['image_id'],
            'reference': pair['reference'],
            'candidate': pair['candidate'],
            'clair_score': score,
            'clair_reason': reason
        }
        
        clair_scores.append(score)
        results.append(result_item)
        
        # 实时保存结果到临时文件
        if temp_output_path:
            try:
                # 保存所有结果（包括之前的和新的）
                temp_data = {
                    'summary': {
                        'total_samples': len(gt_dict),
                        'evaluated_samples': len(results),
                        'valid_predictions': len(gt_dict) - missing_count,
                        'missing_predictions': missing_count,
                        'model': model_name,
                        'status': 'in_progress',
                    },
                    'detailed_results': results
                }
                with open(temp_output_path, 'w', encoding='utf-8') as f:
                    json.dump(temp_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"\n警告: 保存临时结果失败: {e}")
    
    # 评测完成后，删除临时文件
    if temp_output_path and os.path.exists(temp_output_path):
        try:
            os.remove(temp_output_path)
        except:
            pass
    
    # Compute statistics
    mean_clair = np.mean(clair_scores) * 100
    std_clair = np.std(clair_scores) * 100
    median_clair = np.median(clair_scores) * 100
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS (GPT-based CLAIR Score)")
    print("="*60)
    print(f"Total samples evaluated: {len(eval_pairs)}")
    print(f"Model: {model_name}")
    print()
    print(f"Mean CLAIR Score:   {mean_clair:.2f} ± {std_clair:.2f}")
    print(f"Median CLAIR Score: {median_clair:.2f}")
    print()
    
    # Distribution
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_labels = ['0-20', '20-40', '40-60', '60-80', '80-100']
    hist, _ = np.histogram(clair_scores, bins=bins)
    
    print("Score Distribution:")
    for label, count in zip(bin_labels, hist):
        percentage = count / len(clair_scores) * 100
        print(f"  {label}%: {count:4d} ({percentage:5.1f}%)")
    
    print("="*60)
    
    # Save final results
    if output_path:
        output_data = {
            'summary': {
                'total_samples': len(gt_dict),
                'evaluated_samples': len(results),
                'valid_predictions': len(gt_dict) - missing_count,
                'missing_predictions': missing_count,
                'model': model_name,
                'mean_clair': mean_clair,
                'std_clair': std_clair,
                'median_clair': median_clair,
                'status': 'completed',  # 标记为完成
            },
            'detailed_results': results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 评测完成！详细结果已保存到: {output_path}")
        print(f"  - 总样本数: {len(gt_dict)}")
        print(f"  - 已评测: {len(results)}")
        print(f"  - 平均 CLAIR 分数: {mean_clair:.2f}")
    
    return {
        'mean_clair': mean_clair,
        'std_clair': std_clair,
        'median_clair': median_clair
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate Image Captioning with GPT-based CLAIR Score',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 查看可用模型
  python eval_caption_gpt.py --list-models
  
  # 使用默认模型评测
  python eval_caption_gpt.py --gt gt.json --pred pred.json --output results.json
  
  # 使用 GPT-4o 评测 100 个样本
  python eval_caption_gpt.py --gt gt.json --pred pred.json --output results.json --model gpt4o --sample-size 100
  
  # 强制重新评测（忽略已有结果）
  python eval_caption_gpt.py --gt gt.json --pred pred.json --output results.json --force
  
  # 恢复中断的评测（自动跳过已评测样本）
  python eval_caption_gpt.py --gt gt.json --pred pred.json --output results.json
        """
    )
    
    parser.add_argument('--gt', type=str, default=r'UWBench_EVAL_Caption.json', 
                        help='Path to ground truth JSON file')
    parser.add_argument('--pred', type=str, default=r'UWBench_EVAL_Caption_gpt5.json', 
                        help='Path to prediction JSON file')
    parser.add_argument('--output', type=str, default=r'UWBench_EVAL_Caption_gpt5_results.json', 
                        help='Path to save detailed results (supports resume)')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', 
                        help=f'Model name from config (default: gpt-4o-mini). Use --list-models to see all')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Sample size for evaluation (default: all samples)')
    parser.add_argument('--list-models', action='store_true',
                        help='List all available models and exit')
    parser.add_argument('--force', action='store_true',
                        help='Force re-evaluation, ignore existing results')
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        if CHATBOTS_AVAILABLE:
            print("="*60)
            print("可用的评测模型")
            print("="*60)
            for i, model_name in enumerate(get_available_model_names(), 1):
                print(f"  {i:2d}. {model_name}")
            print("="*60)
        else:
            print("ChatBots not available. Cannot list models.")
        sys.exit(0)
    
    # Check if output exists and warn about force flag
    if args.force and args.output and os.path.exists(args.output):
        print(f"警告: 使用 --force 参数，将忽略已有结果文件: {args.output}")
        print("正在备份已有结果...")
        backup_path = args.output + '.backup'
        try:
            import shutil
            shutil.copy2(args.output, backup_path)
            print(f"✓ 已备份到: {backup_path}")
            os.remove(args.output)
        except Exception as e:
            print(f"✗ 备份失败: {e}")
    
    evaluate_caption_gpt(args.gt, args.pred, args.output, args.model, args.sample_size)

