
import re
import subprocess
import tempfile
import sys
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path


class AnswerValidator:
    
    def __init__(self):
        pass
    
    def validate(self, 
                prediction: str, 
                ground_truth: Any, 
                domain: str,
                **kwargs) -> Tuple[bool, Dict[str, Any]]:
        if domain == 'gsm8k':
            return self._validate_gsm8k(prediction, ground_truth)
        elif domain == 'mmlu':
            return self._validate_mmlu(prediction, ground_truth)
        elif domain == 'gpqa':
            return self._validate_mmlu(prediction, ground_truth)
        elif domain == 'gaia':
            return self._validate_gaia(prediction, ground_truth)
        elif domain == 'humaneval':
            return self._validate_humaneval(prediction, ground_truth, **kwargs)
        elif domain == 'mbpp':
            test_list = kwargs.get('test_list', [])
            return self._validate_mbpp(prediction, test_list)
        elif domain == 'hotpotqa':
            return self._validate_hotpotqa(prediction, ground_truth)
        elif domain == 'alfworld':
            return self._validate_alfworld(prediction, ground_truth, **kwargs)
        elif domain == 'math':
            return self._validate_math(prediction, ground_truth)
        else:
            return self._validate_default(prediction, ground_truth)

    def _validate_gaia(self, prediction: str, ground_truth: Any) -> Tuple[bool, Dict[str, Any]]:
        pred = "" if prediction is None else str(prediction)
        true = "" if ground_truth is None else str(ground_truth)

        pred_last = pred.strip().splitlines()[-1].strip() if pred.strip() else ""

        def norm(s: str) -> str:
            s = s.strip().lower()
            for prefix in ("final answer:", "answer:", "the answer is", "final:"):
                if s.startswith(prefix):
                    s = s[len(prefix):].strip()
            s = re.sub(r"[\\s\\t\\n\\r]+", " ", s)
            s = re.sub(r"[\\.,;:!\\?\\(\\)\\[\\]\\{\\}\"'`]", "", s)
            return s.strip()

        pred_n = norm(pred_last)
        true_n = norm(true)

        try:
            pred_num = float(re.sub(r"[^0-9\\-\\.]", "", pred_n))
            true_num = float(re.sub(r"[^0-9\\-\\.]", "", true_n))
            is_correct = abs(pred_num - true_num) < 1e-6
            return is_correct, {"predicted": pred_last, "ground_truth": true, "method": "numeric_or_normalized"}
        except Exception:
            pass

        is_correct = pred_n == true_n and pred_n != ""
        return is_correct, {"predicted": pred_last, "ground_truth": true, "method": "normalized_string_match"}
    
    def _validate_gsm8k(self, prediction: str, ground_truth: str) -> Tuple[bool, Dict[str, Any]]:
        if '####' in str(ground_truth):
            true_answer = str(ground_truth).split('####')[-1].strip()
        else:
            true_answer = str(ground_truth).strip()
        
        if '####' in prediction:
            pred_answer = prediction.split('####')[-1].strip()
        else:
            numbers = re.findall(r'-?\d+\.?\d*', prediction)
            pred_answer = numbers[-1] if numbers else prediction.strip()
        
        pred_clean = re.sub(r'[,\s$]', '', pred_answer)
        true_clean = re.sub(r'[,\s$]', '', true_answer)
        
        try:
            pred_num = float(pred_clean)
            true_num = float(true_clean)
            is_correct = abs(pred_num - true_num) < 1e-6
        except ValueError:
            is_correct = pred_clean.lower() == true_clean.lower()
        
        return is_correct, {
            'predicted': pred_answer,
            'ground_truth': true_answer,
            'method': 'numerical_comparison' if isinstance(is_correct, bool) else 'string_match'
        }
    
    def _validate_mmlu(self, prediction: str, ground_truth: str) -> Tuple[bool, Dict[str, Any]]:
        true_answer = str(ground_truth).strip().upper()
        
        pred_match = re.search(r'\b([ABCD])\b', prediction.upper())
        if pred_match:
            pred_answer = pred_match.group(1)
        else:
            answer_pattern = re.search(r'(?:answer|choice|option)[:\s]+([ABCD])', prediction.upper())
            if answer_pattern:
                pred_answer = answer_pattern.group(1)
            else:
                pred_answer = prediction.strip().upper()[0] if prediction.strip() else ''
        
        is_correct = pred_answer == true_answer
        
        return is_correct, {
            'predicted': pred_answer,
            'ground_truth': true_answer,
            'method': 'multiple_choice'
        }
    
    def _validate_humaneval(self, 
                           prediction: str, 
                           ground_truth: str,
                           test_code: Optional[str] = None,
                           entry_point: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
        if test_code is None or test_code.strip() == "":
            return False, {"method": "test_execution", "error": "empty_test_code"}
        
        if test_code and entry_point:
            try:
                code_block = prediction
                code_block = re.sub(r'<\|[^>]+?\|>', '', code_block)
                code_block = code_block.strip()
                if "```python" in code_block:
                    try:
                        code_block = code_block.split("```python")[1].split("```")[0]
                    except IndexError:
                        pass
                elif "```" in code_block:
                    try:
                        code_block = code_block.split("```")[1].split("```")[0]
                    except IndexError:
                        pass
                pattern = rf'def\s+{re.escape(entry_point)}\s*\('
                m = re.search(pattern, code_block)
                if not m:
                    m = re.search(r'def\s+\w+\s*\(', code_block)
                if m:
                    code_block = code_block[m.start():]
                
                code_block = self._fix_common_syntax_errors(code_block)
                
                header = "from __future__ import annotations\nfrom typing import *\n"
                full_code = f"{header}\n{code_block}\n\n{test_code}"
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(full_code)
                    temp_file = f.name
                
                try:
                    result = subprocess.run(
                        [sys.executable, temp_file],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    is_correct = result.returncode == 0
                    error_msg = result.stderr if not is_correct else None
                    
                finally:
                    Path(temp_file).unlink()
                
                return is_correct, {
                    'predicted': prediction[:100] + '...' if len(prediction) > 100 else prediction,
                    'ground_truth': ground_truth[:100] + '...' if len(ground_truth) > 100 else ground_truth,
                    'method': 'test_execution',
                    'error': error_msg
                }
                
            except subprocess.TimeoutExpired:
                return False, {
                    'method': 'test_execution',
                    'error': 'Test execution timeout'
                }
            except Exception as e:
                return False, {
                    'method': 'test_execution',
                    'error': str(e)
                }
        else:
            pred_clean = re.sub(r'#.*?\n', '\n', prediction)
            pred_clean = re.sub(r'\s+', ' ', pred_clean).strip()
            
            truth_clean = re.sub(r'#.*?\n', '\n', str(ground_truth))
            truth_clean = re.sub(r'\s+', ' ', truth_clean).strip()
            
            is_correct = pred_clean == truth_clean
            
            return is_correct, {
                'method': 'code_similarity',
                'note': 'No test code provided, using similarity comparison'
            }
    
    def _validate_mbpp(self, 
                      prediction: str, 
                      test_list: List[str]) -> Tuple[bool, Dict[str, Any]]:
        code_block = prediction
        if "```python" in prediction:
            try:
                code_block = prediction.split("```python")[1].split("```")[0]
            except IndexError:
                pass
        elif "```" in prediction:
            try:
                code_block = prediction.split("```")[1].split("```")[0]
            except IndexError:
                pass
        
        import re
        code_block = re.sub(r'<\|[^>]+?\|>', '', code_block)
        code_block = code_block.strip()
        m = re.search(r'def\s+\w+\s*\(', code_block)
        if m:
            code_block = code_block[m.start():]
        
        code_block = self._fix_common_syntax_errors(code_block)
        
        
        header = "from __future__ import annotations\nfrom typing import *\n"
        full_test_script = header + "\n" + code_block + "\n\n" + "\n".join(test_list)
        
        is_correct = False
        error_msg = ""
        
        import tempfile
        import subprocess
        import os
        from pathlib import Path
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write(full_test_script)
                temp_path = f.name
            
            import sys
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                is_correct = True
            else:
                error_msg = result.stderr
                
            Path(temp_path).unlink()
            
        except subprocess.TimeoutExpired:
            error_msg = "Execution timed out"
            if os.path.exists(temp_path):
                Path(temp_path).unlink()
        except Exception as e:
            error_msg = str(e)
            if os.path.exists(temp_path):
                Path(temp_path).unlink()
        
        return is_correct, {
            'predicted': prediction[:200] + '...',
            'test_list_sample': test_list[:1],
            'error': error_msg[:200] if error_msg else None,
            'method': 'code_execution'
        }

    def _validate_hotpotqa(self, prediction: str, ground_truth: str) -> Tuple[bool, Dict[str, Any]]:
        pred_clean = prediction.strip().lower()
        truth_clean = str(ground_truth).strip().lower()
        
        if pred_clean == truth_clean:
            return True, {
                'predicted': prediction,
                'ground_truth': ground_truth,
                'method': 'exact_match'
            }
        
        if truth_clean in pred_clean:
            return True, {
                'predicted': prediction,
                'ground_truth': ground_truth,
                'method': 'substring_match'
            }
        
        if pred_clean in truth_clean:
            return True, {
                'predicted': prediction,
                'ground_truth': ground_truth,
                'method': 'reverse_substring_match'
            }
        
        pred_words = set(re.findall(r'\b\w+\b', pred_clean))
        truth_words = set(re.findall(r'\b\w+\b', truth_clean))
        
        if truth_words:
            overlap = len(pred_words & truth_words) / len(truth_words)
            if overlap >= 0.7:
                return True, {
                    'predicted': prediction,
                    'ground_truth': ground_truth,
                    'method': 'keyword_overlap',
                    'overlap_ratio': overlap
                }
        
        return False, {
            'predicted': prediction,
            'ground_truth': ground_truth,
            'method': 'no_match'
        }
    
    def _validate_alfworld(self, 
                          prediction: str, 
                          ground_truth: str,
                          subgoals: Optional[List[str]] = None) -> Tuple[bool, Dict[str, Any]]:
        if subgoals:
            achieved_subgoals = []
            for idx, subgoal_pattern in enumerate(subgoals):
                try:
                    if re.search(subgoal_pattern, prediction, re.IGNORECASE):
                        achieved_subgoals.append(idx)
                except re.error:
                    continue
            
            achievement_rate = len(achieved_subgoals) / len(subgoals) if len(subgoals) > 0 else 0.0
            
            if achievement_rate >= 0.65:
                return True, {
                    'predicted': prediction[:200] + '...' if len(prediction) > 200 else prediction,
                    'ground_truth': ground_truth,
                    'subgoals': subgoals,
                    'achieved_subgoals': achieved_subgoals,
                    'achievement_rate': achievement_rate,
                    'method': 'subgoal_regex_relaxed'
                }
            
            action_keywords = [
                'go to', 'goto', 'examine', 'open', 'close', 'take', 'put', 
                'pick up', 'place', 'toggle', 'heat', 'cool', 'clean', 'slice',
                'you see', 'you pick', 'you put', 'you open', 'you close'
            ]
            
            goal_lower = str(ground_truth).lower()
            pred_lower = prediction.lower()
            
            action_hits = [kw for kw in action_keywords if kw in pred_lower]
            action_found = len(action_hits) >= 1
            
            stop_words = {'a', 'an', 'the', 'in', 'on', 'at', 'under', 'to', 'from', 'with'}
            goal_words = set(word for word in re.findall(r'\b\w+\b', goal_lower) if word not in stop_words)
            pred_words = set(re.findall(r'\b\w+\b', pred_lower))
            
            if goal_words:
                object_overlap = len(goal_words & pred_words) / len(goal_words)
            else:
                object_overlap = 0.0
            
            is_correct_keyword = action_found and object_overlap >= 0.5
            
            if is_correct_keyword:
                return True, {
                    'predicted': prediction[:200] + '...' if len(prediction) > 200 else prediction,
                    'ground_truth': ground_truth,
                    'subgoals': subgoals,
                    'achieved_subgoals': achieved_subgoals,
                    'achievement_rate': achievement_rate,
                    'action_hits': action_hits,
                    'object_overlap': object_overlap,
                    'method': 'keyword_matching'
                }
            
            return False, {
                'predicted': prediction[:200] + '...' if len(prediction) > 200 else prediction,
                'ground_truth': ground_truth,
                'subgoals': subgoals,
                'achieved_subgoals': achieved_subgoals,
                'achievement_rate': achievement_rate,
                'action_found': action_found,
                'object_overlap': object_overlap,
                'method': 'both_failed'
            }
        else:
            return False, {
                'predicted': prediction,
                'ground_truth': ground_truth,
                'subgoals': subgoals,
                'method': 'no_subgoals_provided'
            }
    
    def _validate_math(self, prediction: str, ground_truth: str) -> Tuple[bool, Dict[str, Any]]:
        def normalize_latex(text: str) -> str:
            text = re.sub(r'\\left\(', '(', text)
            text = re.sub(r'\\right\)', ')', text)
            text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', text)
            text = re.sub(r'\\pi', 'π', text)
            text = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', text)
            text = re.sub(r'\\boxed\{([^}]+)\}', r'\1', text)
            text = re.sub(r'\\[a-zA-Z]+\{([^}]+)\}', r'\1', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text.lower()
        
        pred_normalized = normalize_latex(prediction)
        truth_normalized = normalize_latex(str(ground_truth))
        
        if pred_normalized == truth_normalized:
            return True, {
                'predicted': prediction,
                'ground_truth': ground_truth,
                'method': 'normalized_exact_match'
            }
        
        pred_tokens = set(re.findall(r'[\dπ\+\-\*/\(\)]+', pred_normalized))
        truth_tokens = set(re.findall(r'[\dπ\+\-\*/\(\)]+', truth_normalized))
        
        if truth_tokens and pred_tokens == truth_tokens:
            return True, {
                'predicted': prediction,
                'ground_truth': ground_truth,
                'method': 'token_match'
            }
        
        try:
            pred_numbers = re.findall(r'-?\d+\.?\d*', prediction)
            truth_numbers = re.findall(r'-?\d+\.?\d*', str(ground_truth))
            
            if pred_numbers and truth_numbers:
                pred_num = float(pred_numbers[-1])
                truth_num = float(truth_numbers[-1])
                if abs(pred_num - truth_num) < 1e-6:
                    return True, {
                        'predicted': prediction,
                        'ground_truth': ground_truth,
                        'method': 'numerical_match'
                    }
        except (ValueError, IndexError):
            pass
        
        return False, {
            'predicted': prediction,
            'ground_truth': ground_truth,
            'method': 'no_match'
        }
    
    def _validate_default(self, prediction: str, ground_truth: str) -> Tuple[bool, Dict[str, Any]]:
        pred_clean = prediction.strip().upper()
        truth_clean = str(ground_truth).strip().upper()
        
        is_correct = pred_clean == truth_clean or truth_clean in pred_clean
        
        return is_correct, {
            'predicted': prediction,
            'ground_truth': ground_truth,
            'method': 'default_string_match'
        }
    
    def _fix_common_syntax_errors(self, code: str) -> str:
        code = re.sub(
            r'^(\s*def\s+\w+\s*\([^)]*\)(?:\s*->\s*[^:]+)?)\s*$',
            r'\1:',
            code,
            flags=re.MULTILINE
        )
        
        code = re.sub(
            r'^(\s*class\s+\w+(?:\([^)]*\))?)\s*$',
            r'\1:',
            code,
            flags=re.MULTILINE
        )
        
        
        return code


def create_answer_validator() -> AnswerValidator:
    return AnswerValidator()


if __name__ == "__main__":
    validator = create_answer_validator()
    
    print("Testing GSM8K:")
    pred = "The answer is 18 dollars."
    truth = "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.\n#### 18"
    is_correct, details = validator.validate(pred, truth, 'gsm8k')
    print(f"  Correct: {is_correct}, Details: {details}")
    
    print("\nTesting MMLU:")
    pred = "The answer is B."
    truth = "B"
    is_correct, details = validator.validate(pred, truth, 'mmlu')
    print(f"  Correct: {is_correct}, Details: {details}")
    
    print("\nTesting HotpotQA:")
    pred = "AIA Group is a pan-Asian life insurance group."
    truth = "pan-Asian life insurance group"
    is_correct, details = validator.validate(pred, truth, 'hotpotqa')
    print(f"  Correct: {is_correct}, Details: {details}")
    
    print("\nTesting MATH:")
    pred = "The answer is (3, π/2)."
    truth = "\\left( 3, \\frac{\\pi}{2} \\right)"
    is_correct, details = validator.validate(pred, truth, 'math')
    print(f"  Correct: {is_correct}, Details: {details}")

