"""
GPT-based evaluation for Visual Question Answering (VQA) task
Uses GPT to assess semantic similarity between predicted and reference answers
"""

import json
import argparse
import numpy as np
import re
from tqdm import tqdm
import os
import sys
from collections import defaultdict

# 导入现有的模块
try:
    from key.chat_bots import ChatBots
    from key.models_config import get_model_config, get_available_model_names
    CHATBOTS_AVAILABLE = True
except ImportError as e:
    CHATBOTS_AVAILABLE = False
    print(f"WARNING: ChatBots not found: {e}")
    print("Please ensure key/chat_bots.py and key/models_config.py exist")


_VQA_MATCH_PROMPT = """Question: {question}
Ground Truth Answer: {ground_truth}
Predicted Answer: {predicted}

Does the predicted answer match the ground truth? Answer 1 for match and 0 for not match. 
Use semantic meaning not exact match. Synonyms are also treated as a match, e.g., football and soccer, playground and ground track field, building and rooftop, pond and swimming pool. 
Do not explain the reason. Only output 1 or 0.
"""


def check_answer_match(chat_bots, question, ground_truth, predicted, max_retries=3):
    """
    Use GPT to check if predicted answer matches ground truth
    
    Args:
        chat_bots: ChatBots instance
        question: The question text
        ground_truth: Ground truth answer
        predicted: Predicted answer
        max_retries: Maximum number of retries
        
    Returns:
        '1' for match, '0' for not match
    """
    if not CHATBOTS_AVAILABLE:
        return '0'
    
    formatted_prompt = _VQA_MATCH_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        predicted=predicted
    )
    
    for attempt in range(max_retries):
        try:
            result = chat_bots.call(txt=formatted_prompt, img=None, isMsg=False, test=False)
            
            if result is None:
                print(f"Warning: ChatBots returned None on attempt {attempt + 1}")
                continue
            
            # result = [content, prompt_tokens, completion_tokens]
            response_text = result[0].strip()
            
            # 尝试提取 1 或 0
            if '1' in response_text and '0' not in response_text:
                return '1'
            elif '0' in response_text and '1' not in response_text:
                return '0'
            elif response_text.startswith('1'):
                return '1'
            elif response_text.startswith('0'):
                return '0'
            else:
                # 如果无法判断，尝试从第一个数字提取
                numbers = re.findall(r'[01]', response_text)
                if numbers:
                    return numbers[0]
                print(f"Warning: Unclear response: {response_text}")
                continue
                
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                continue
            else:
                return '0'  # 默认返回不匹配
    
    return '0'  # 失败后默认返回不匹配


def simple_match_check(ground_truth, predicted):
    """
    Simple rule-based matching for common cases
    
    Returns:
        '1' for match, '0' for not match, None for uncertain (need GPT)
    """
    gt = ground_truth.lower().strip()
    pred = predicted.lower().strip()
    
    # 精确匹配
    if gt == pred:
        return '1'
    
    # GT 包含在预测中
    if gt in pred:
        return '1'
    
    # 预测包含在 GT 中（可能过于宽松，谨慎使用）
    # if pred in gt and len(pred) > 3:
    #     return '1'
    
    # Yes/No 问题和数字 - 需要精确匹配
    if gt in ['yes', 'no'] + [str(i) for i in range(100)]:
        return '1' if gt == pred else '0'
    
    # 需要 GPT 判断
    return None


def evaluate_vqa_gpt(gt_path, pred_path, output_path=None, model_name="gpt-4o-mini", sample_size=None):
    """
    Evaluate VQA using GPT-based semantic matching
    
    Args:
        gt_path: Path to ground truth JSON file
        pred_path: Path to prediction JSON file
        output_path: Optional path to save detailed results
        model_name: Model name from models_config
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
    
    # Build dictionaries indexed by image_id and question_id
    gt_dict = {}
    for item in gt_data:
        key = (item['image_id'], item['question_id'])
        gt_dict[key] = item
    
    pred_dict = {}
    for item in pred_data:
        key = (item['image_id'], item['question_id'])
        pred_dict[key] = item
    
    print(f"\nGround truth samples: {len(gt_dict)}")
    print(f"Prediction samples: {len(pred_dict)}")
    
    # Prepare evaluation pairs
    eval_pairs = []
    missing_count = 0
    
    for key, gt_item in gt_dict.items():
        if key in pred_dict:
            pred_item = pred_dict[key]
            eval_pairs.append({
                'image_id': gt_item['image_id'],
                'question_id': gt_item['question_id'],
                'question': gt_item['question'],
                'ground_truth': gt_item['ground_truth'],
                'predicted': pred_item['ground_truth'],  # prediction is in this field
                'type': gt_item.get('type', 'unknown')
            })
        else:
            # Missing prediction
            missing_count += 1
            eval_pairs.append({
                'image_id': gt_item['image_id'],
                'question_id': gt_item['question_id'],
                'question': gt_item['question'],
                'ground_truth': gt_item['ground_truth'],
                'predicted': "",
                'type': gt_item.get('type', 'unknown')
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
    evaluated_keys = set()
    existing_results = []
    
    if output_path and os.path.exists(output_path):
        print(f"\n发现已有结果文件: {output_path}")
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                if 'detailed_results' in existing_data:
                    existing_results = existing_data['detailed_results']
                    evaluated_keys = {(r['image_id'], r['question_id']) for r in existing_results}
                    print(f"已加载 {len(existing_results)} 条已评测结果")
                    print(f"将跳过这些样本，继续评测剩余样本")
        except Exception as e:
            print(f"警告: 无法读取已有结果文件: {e}")
            print("将从头开始评测")
    
    # 过滤掉已评测的样本
    remaining_pairs = [p for p in eval_pairs if (p['image_id'], p['question_id']) not in evaluated_keys]
    
    print(f"\n总样本数: {len(eval_pairs)}")
    print(f"已评测: {len(evaluated_keys)}")
    print(f"待评测: {len(remaining_pairs)}")
    print(f"使用模型: {model_name}")
    print("="*60)
    
    # Evaluate remaining samples
    results = existing_results.copy()
    
    # 统计信息
    rule_match_count = 0
    gpt_match_count = 0
    
    # 准备临时结果文件（实时保存）
    temp_output_path = output_path + '.tmp' if output_path else None
    
    for pair in tqdm(remaining_pairs, desc="Evaluating VQA"):
        # 首先尝试规则匹配
        match_result = simple_match_check(pair['ground_truth'], pair['predicted'])
        
        if match_result is None:
            # 需要 GPT 判断
            match_result = check_answer_match(
                chat_bots,
                pair['question'],
                pair['ground_truth'],
                pair['predicted']
            )
            gpt_match_count += 1
        else:
            rule_match_count += 1
        
        result_item = {
            'image_id': pair['image_id'],
            'question_id': pair['question_id'],
            'type': pair['type'],
            'question': pair['question'],
            'ground_truth': pair['ground_truth'],
            'predicted': pair['predicted'],
            'correct': match_result
        }
        
        results.append(result_item)
        
        # 实时保存结果到临时文件
        if temp_output_path:
            try:
                temp_data = {
                    'summary': {
                        'total_samples': len(gt_dict),
                        'evaluated_samples': len(results),
                        'model': model_name,
                        'rule_match_count': rule_match_count,
                        'gpt_match_count': gpt_match_count,
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
    
    # Calculate statistics
    correct = sum([1 for r in results if r['correct'] == '1'])
    total = len(results)
    overall_acc = correct / total * 100 if total > 0 else 0
    
    # Statistics by question type
    correct_per_type = defaultdict(int)
    total_per_type = defaultdict(int)
    
    for result in results:
        q_type = result['type'].lower()
        total_per_type[q_type] += 1
        if result['correct'] == '1':
            correct_per_type[q_type] += 1
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS (VQA)")
    print("="*60)
    print(f"Total samples: {total}")
    print(f"Model: {model_name}")
    print()
    print(f"Rule-based matches: {rule_match_count}")
    print(f"GPT-based matches: {gpt_match_count}")
    print()
    print(f"Overall Accuracy: {overall_acc:.2f}% ({correct}/{total})")
    print()
    
    # Print accuracy by type
    if correct_per_type:
        print("Accuracy by Question Type:")
        print("-" * 60)
        type_accs = []
        for q_type in sorted(total_per_type.keys()):
            if total_per_type[q_type] > 0:
                acc = correct_per_type[q_type] / total_per_type[q_type] * 100
                type_accs.append(acc)
                print(f"  {q_type:25s}: {acc:6.2f}% ({correct_per_type[q_type]}/{total_per_type[q_type]})")
        
        if type_accs:
            print("-" * 60)
            print(f"  {'Average':25s}: {np.mean(type_accs):6.2f}%")
    
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
                'overall_accuracy': overall_acc,
                'correct_count': correct,
                'rule_match_count': rule_match_count,
                'gpt_match_count': gpt_match_count,
                'accuracy_by_type': {q_type: correct_per_type[q_type] / total_per_type[q_type] * 100 
                                     for q_type in total_per_type.keys() if total_per_type[q_type] > 0},
                'status': 'completed',
            },
            'detailed_results': results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 评测完成！详细结果已保存到: {output_path}")
        print(f"  - 总样本数: {len(gt_dict)}")
        print(f"  - 已评测: {len(results)}")
        print(f"  - 总准确率: {overall_acc:.2f}%")
    
    return {
        'overall_accuracy': overall_acc,
        'correct_count': correct,
        'total_count': total,
        'accuracy_by_type': {q_type: correct_per_type[q_type] / total_per_type[q_type] * 100 
                            for q_type in total_per_type.keys() if total_per_type[q_type] > 0}
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate Visual Question Answering with GPT-based Semantic Matching',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 查看可用模型
  python eval_vqa_gpt.py --list-models
  
  # 使用默认模型评测
  python eval_vqa_gpt.py --gt gt.json --pred pred.json --output results.json
  
  # 使用 GPT-4o 评测
  python eval_vqa_gpt.py --gt gt.json --pred pred.json --output results.json --model gpt4o
  
  # 评测 1000 个样本（测试）
  python eval_vqa_gpt.py --gt gt.json --pred pred.json --output results.json --sample-size 1000
  
  # 强制重新评测
  python eval_vqa_gpt.py --gt gt.json --pred pred.json --output results.json --force
        """
    )
    
    parser.add_argument('--gt', type=str, default=r'UWBench_EVAL_VQA.json', 
                        help='Path to ground truth JSON file')
    parser.add_argument('--pred', type=str, default=r'InternVL3.5-241B\UWBench_EVAL_VQA_InternVL3.5-241B.json', 
                        help='Path to prediction JSON file')
    parser.add_argument('--output', type=str, default=r'UWBench_EVAL_VQA_InternVL3.5-241B_results.json', 
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
    
    evaluate_vqa_gpt(args.gt, args.pred, args.output, args.model, args.sample_size)

