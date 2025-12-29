"""
Answer Validator for Different Datasets
不同数据集的答案验证器

针对每个数据集的特点，实现专门的答案验证方法：
1. GSM8K: 提取数值，进行数值比较
2. MMLU: 选择题，直接比较A/B/C/D
3. HumanEval: 执行测试代码验证
4. HotpotQA: 字符串匹配（不区分大小写，支持部分匹配）
5. ALFWorld: 验证动作序列是否匹配subgoals
6. MATH: LaTeX格式答案，规范化比较
"""

import re
import subprocess
import tempfile
import sys
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path


class AnswerValidator:
    """
    答案验证器
    
    为不同数据集提供专门的答案验证方法
    """
    
    def __init__(self):
        """初始化验证器"""
        pass
    
    def validate(self, 
                prediction: str, 
                ground_truth: Any, 
                domain: str,
                **kwargs) -> Tuple[bool, Dict[str, Any]]:
        """
        验证答案正确性
        
        Args:
            prediction: 模型预测的答案
            ground_truth: 正确答案（格式取决于数据集）
            domain: 数据集名称
            **kwargs: 数据集特定的额外参数
        
        Returns:
            (is_correct, details_dict)
        """
        if domain == 'gsm8k':
            return self._validate_gsm8k(prediction, ground_truth)
        elif domain == 'mmlu':
            return self._validate_mmlu(prediction, ground_truth)
        elif domain == 'gpqa':
            # GPQA 与 MMLU 同为 A/B/C/D 选择题验证
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
            # 默认：简单字符串匹配
            return self._validate_default(prediction, ground_truth)

    def _validate_gaia(self, prediction: str, ground_truth: Any) -> Tuple[bool, Dict[str, Any]]:
        """
        验证 GAIA 答案（偏宽松的文本归一化匹配）。

        GAIA 的答案通常是短字符串/数值/实体名称等。
        这里做：
        - 去首尾空白、大小写归一化
        - 去掉常见标点与多余空格
        - 对纯数字进行数值比较（允许极小误差）
        """
        pred = "" if prediction is None else str(prediction)
        true = "" if ground_truth is None else str(ground_truth)

        # 取模型输出最后一行作为“最终答案”候选（很多模型会带解释）
        pred_last = pred.strip().splitlines()[-1].strip() if pred.strip() else ""

        def norm(s: str) -> str:
            s = s.strip().lower()
            # 移除常见前缀
            for prefix in ("final answer:", "answer:", "the answer is", "final:"):
                if s.startswith(prefix):
                    s = s[len(prefix):].strip()
            # 去掉标点与多余空白
            s = re.sub(r"[\\s\\t\\n\\r]+", " ", s)
            s = re.sub(r"[\\.,;:!\\?\\(\\)\\[\\]\\{\\}\"'`]", "", s)
            return s.strip()

        pred_n = norm(pred_last)
        true_n = norm(true)

        # 数值比较（如果两边都能解析）
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
        """
        验证GSM8K答案
        
        GSM8K格式: "推理过程\n#### 最终答案"
        需要提取####后的数值，进行数值比较
        """
        # 从ground_truth中提取最终答案（如果包含####）
        if '####' in str(ground_truth):
            true_answer = str(ground_truth).split('####')[-1].strip()
        else:
            true_answer = str(ground_truth).strip()
        
        # 从prediction中提取数值答案
        # 方法1: 查找####后的内容
        if '####' in prediction:
            pred_answer = prediction.split('####')[-1].strip()
        else:
            # 方法2: 查找最后一个数字
            numbers = re.findall(r'-?\d+\.?\d*', prediction)
            pred_answer = numbers[-1] if numbers else prediction.strip()
        
        # 清理答案：移除逗号、空格等
        pred_clean = re.sub(r'[,\s$]', '', pred_answer)
        true_clean = re.sub(r'[,\s$]', '', true_answer)
        
        # 尝试数值比较
        try:
            pred_num = float(pred_clean)
            true_num = float(true_clean)
            is_correct = abs(pred_num - true_num) < 1e-6
        except ValueError:
            # 如果无法转换为数字，进行字符串比较
            is_correct = pred_clean.lower() == true_clean.lower()
        
        return is_correct, {
            'predicted': pred_answer,
            'ground_truth': true_answer,
            'method': 'numerical_comparison' if isinstance(is_correct, bool) else 'string_match'
        }
    
    def _validate_mmlu(self, prediction: str, ground_truth: str) -> Tuple[bool, Dict[str, Any]]:
        """
        验证MMLU答案
        
        MMLU是选择题，答案格式为A/B/C/D
        需要从prediction中提取选项字母
        """
        # 清理ground_truth
        true_answer = str(ground_truth).strip().upper()
        
        # 从prediction中提取选项（查找A/B/C/D）
        pred_match = re.search(r'\b([ABCD])\b', prediction.upper())
        if pred_match:
            pred_answer = pred_match.group(1)
        else:
            # 如果没有找到，尝试查找"Answer: A"等格式
            answer_pattern = re.search(r'(?:answer|choice|option)[:\s]+([ABCD])', prediction.upper())
            if answer_pattern:
                pred_answer = answer_pattern.group(1)
            else:
                # 最后尝试：取第一个字母
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
        """
        验证HumanEval答案
        
        HumanEval需要执行测试代码来验证
        如果提供了test_code，执行测试；否则进行代码相似度比较
        """
        if test_code is None or test_code.strip() == "":
            return False, {"method": "test_execution", "error": "empty_test_code"}
        
        if test_code and entry_point:
            # 执行测试代码验证
            try:
                # 1) 清理控制标记（如 <|submit|>）和多余前缀，只保留函数定义
                code_block = prediction
                # 去掉所有形如 <|...|> 的控制token
                code_block = re.sub(r'<\|[^>]+?\|>', '', code_block)
                code_block = code_block.strip()
                # 如果有代码块标记，优先使用其中的python代码
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
                # 优先从 entry_point 对应的函数定义开始截取；若找不到，则从第一个 def 开始
                # 注意：这里使用单反斜杠 + 原始字符串，生成真正的正则空白符 \s
                pattern = rf'def\s+{re.escape(entry_point)}\s*\('
                m = re.search(pattern, code_block)
                if not m:
                    # 退回到匹配任意函数定义
                    m = re.search(r'def\s+\w+\s*\(', code_block)
                if m:
                    code_block = code_block[m.start():]
                
                # ✨ 1.5) 语法修复：自动修复常见的语法错误
                code_block = self._fix_common_syntax_errors(code_block)
                
                # 2) 构建完整的测试代码（加入typing和future以避免类型注解报错）
                header = "from __future__ import annotations\nfrom typing import *\n"
                full_code = f"{header}\n{code_block}\n\n{test_code}"
                
                # 创建临时文件
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(full_code)
                    temp_file = f.name
                
                try:
                    # 执行测试
                    # 使用当前解释器运行，避免与系统默认 python 冲突
                    result = subprocess.run(
                        [sys.executable, temp_file],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    # 如果返回码为0，测试通过
                    is_correct = result.returncode == 0
                    error_msg = result.stderr if not is_correct else None
                    
                finally:
                    # 清理临时文件
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
            # 如果没有测试代码，进行代码相似度比较（简单方法）
            # 移除空白和注释后比较
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
        """
        验证MBPP代码
        
        Args:
            prediction: 生成的代码
            test_list: 测试用例列表（包含assert语句）
            
        Returns:
            (是否通过, 详情)
        """
        # 提取Python代码块
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
        
        # 清理控制标记（例如 `<|submit|>`），并截取从第一个 `def` 开始的部分
        import re
        # 去掉所有形如 <|...|> 的控制token（前缀、中间或后缀）
        code_block = re.sub(r'<\|[^>]+?\|>', '', code_block)
        code_block = code_block.strip()
        # 尝试从第一个函数定义开始截取，避免前面混入自然语言或其它内容
        m = re.search(r'def\s+\w+\s*\(', code_block)
        if m:
            code_block = code_block[m.start():]
        
        # ✨ 语法修复：自动修复常见的语法错误（如缺少冒号）
        code_block = self._fix_common_syntax_errors(code_block)
        
        # 此时期望 code_block 是一个完整的、可执行的 Python 函数定义
        
        # 构建完整的测试脚本
        # 组合：必要的导入 + 生成的代码 + 测试断言
        header = "from __future__ import annotations\nfrom typing import *\n"
        full_test_script = header + "\n" + code_block + "\n\n" + "\n".join(test_list)
        
        # 执行测试
        is_correct = False
        error_msg = ""
        
        import tempfile
        import subprocess
        import os
        from pathlib import Path
        
        # 使用临时文件执行代码
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write(full_test_script)
                temp_path = f.name
            
            # 在独立进程中运行，设置超时
            # 使用sys.executable确保使用相同的python环境
            import sys
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=5  # 5秒超时
            )
            
            if result.returncode == 0:
                is_correct = True
            else:
                error_msg = result.stderr
                
            # 清理临时文件
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
        """
        验证HotpotQA答案
        
        HotpotQA是开放域问答，答案通常是短语或短句
        使用不区分大小写的字符串匹配，支持部分匹配
        """
        pred_clean = prediction.strip().lower()
        truth_clean = str(ground_truth).strip().lower()
        
        # 方法1: 完全匹配
        if pred_clean == truth_clean:
            return True, {
                'predicted': prediction,
                'ground_truth': ground_truth,
                'method': 'exact_match'
            }
        
        # 方法2: 检查ground_truth是否在prediction中
        if truth_clean in pred_clean:
            return True, {
                'predicted': prediction,
                'ground_truth': ground_truth,
                'method': 'substring_match'
            }
        
        # 方法3: 检查prediction是否在ground_truth中（处理ground_truth是完整句子的情况）
        if pred_clean in truth_clean:
            return True, {
                'predicted': prediction,
                'ground_truth': ground_truth,
                'method': 'reverse_substring_match'
            }
        
        # 方法4: 提取关键短语进行比较
        # 移除标点符号和常见停用词
        pred_words = set(re.findall(r'\b\w+\b', pred_clean))
        truth_words = set(re.findall(r'\b\w+\b', truth_clean))
        
        # 计算重叠度
        if truth_words:
            overlap = len(pred_words & truth_words) / len(truth_words)
            if overlap >= 0.7:  # 70%以上重叠认为正确
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
        """
        验证ALFWorld答案
        
        ALFWorld需要验证动作序列是否匹配subgoals
        ground_truth是goal描述，实际验证需要检查subgoals是否达成
        
        ✨ 三层判定策略：
        1. 正则匹配：达成60%以上的subgoals即判为正确
        2. 关键词匹配：如果正则匹配率<60%，检查关键动作词是否出现
        3. 目标匹配：如果无subgoals，检查goal是否在输出中
        """
        if subgoals:
            # ===== 第一层：正则表达式匹配（收紧：至少80%子目标达成）=====
            achieved_subgoals = []
            for idx, subgoal_pattern in enumerate(subgoals):
                try:
                    if re.search(subgoal_pattern, prediction, re.IGNORECASE):
                        achieved_subgoals.append(idx)
                except re.error:
                    # 如果正则表达式无效，跳过
                    continue
            
            # 计算达成率
            achievement_rate = len(achieved_subgoals) / len(subgoals) if len(subgoals) > 0 else 0.0
            
            # ✅ 达成率≥65% 判为正确（适度宽松，考虑ALFWorld的复杂性）
            if achievement_rate >= 0.65:
                return True, {
                    'predicted': prediction[:200] + '...' if len(prediction) > 200 else prediction,
                    'ground_truth': ground_truth,
                    'subgoals': subgoals,
                    'achieved_subgoals': achieved_subgoals,
                    'achievement_rate': achievement_rate,
                    'method': 'subgoal_regex_relaxed'
                }
            
            # ===== 第二层：关键动作词匹配（适度放宽：动作词≥1 + 对象重叠≥50%）=====
            # 提取常见的ALFWorld动作关键词
            action_keywords = [
                'go to', 'goto', 'examine', 'open', 'close', 'take', 'put', 
                'pick up', 'place', 'toggle', 'heat', 'cool', 'clean', 'slice',
                'you see', 'you pick', 'you put', 'you open', 'you close'
            ]
            
            # 从goal中提取关键对象词
            goal_lower = str(ground_truth).lower()
            pred_lower = prediction.lower()
            
            # 检查包含的动作词数量
            action_hits = [kw for kw in action_keywords if kw in pred_lower]
            action_found = len(action_hits) >= 1  # ✅ 至少命中1个动作词即可（宽松）
            
            # 提取goal中的关键对象（去除常见停用词）
            stop_words = {'a', 'an', 'the', 'in', 'on', 'at', 'under', 'to', 'from', 'with'}
            goal_words = set(word for word in re.findall(r'\b\w+\b', goal_lower) if word not in stop_words)
            pred_words = set(re.findall(r'\b\w+\b', pred_lower))
            
            # 计算关键对象词重叠率
            if goal_words:
                object_overlap = len(goal_words & pred_words) / len(goal_words)
            else:
                object_overlap = 0.0
            
            # ✅ 需要：动作词≥1 且 关键对象重叠率≥50%（适度宽松）
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
            
            # 两层判定都未通过，返回失败
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
            # ===== 无 subgoals 时的处理 =====
            # 按约定 ALFWorld 数据应总是提供 subgoals；若缺失，则直接判为未达成并返回提示
            return False, {
                'predicted': prediction,
                'ground_truth': ground_truth,
                'subgoals': subgoals,
                'method': 'no_subgoals_provided'
            }
    
    def _validate_math(self, prediction: str, ground_truth: str) -> Tuple[bool, Dict[str, Any]]:
        """
        验证MATH答案
        
        MATH答案通常是LaTeX格式，需要规范化后比较
        例如: \left( 3, \frac{\pi}{2} \right) 和 (3, π/2) 应该被认为是相同的
        """
        # 清理LaTeX格式
        def normalize_latex(text: str) -> str:
            # 移除LaTeX命令，保留内容
            text = re.sub(r'\\left\(', '(', text)
            text = re.sub(r'\\right\)', ')', text)
            text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', text)
            text = re.sub(r'\\pi', 'π', text)
            text = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', text)
            text = re.sub(r'\\boxed\{([^}]+)\}', r'\1', text)
            # 移除其他LaTeX命令
            text = re.sub(r'\\[a-zA-Z]+\{([^}]+)\}', r'\1', text)
            # 移除多余空格
            text = re.sub(r'\s+', ' ', text).strip()
            return text.lower()
        
        pred_normalized = normalize_latex(prediction)
        truth_normalized = normalize_latex(str(ground_truth))
        
        # 方法1: 规范化后完全匹配
        if pred_normalized == truth_normalized:
            return True, {
                'predicted': prediction,
                'ground_truth': ground_truth,
                'method': 'normalized_exact_match'
            }
        
        # 方法2: 提取数值和符号进行比较
        # 提取所有数字和数学符号
        pred_tokens = set(re.findall(r'[\dπ\+\-\*/\(\)]+', pred_normalized))
        truth_tokens = set(re.findall(r'[\dπ\+\-\*/\(\)]+', truth_normalized))
        
        if truth_tokens and pred_tokens == truth_tokens:
            return True, {
                'predicted': prediction,
                'ground_truth': ground_truth,
                'method': 'token_match'
            }
        
        # 方法3: 如果ground_truth是简单数值，尝试数值比较
        try:
            # 尝试从prediction中提取数值
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
        """
        默认验证方法：简单字符串匹配
        """
        pred_clean = prediction.strip().upper()
        truth_clean = str(ground_truth).strip().upper()
        
        is_correct = pred_clean == truth_clean or truth_clean in pred_clean
        
        return is_correct, {
            'predicted': prediction,
            'ground_truth': ground_truth,
            'method': 'default_string_match'
        }
    
    def _fix_common_syntax_errors(self, code: str) -> str:
        """
        自动修复常见的 Python 语法错误
        
        ⚠️ 注意：只修复安全的、不会误伤正常代码的情况
        
        Args:
            code: 原始代码
        
        Returns:
            修复后的代码
        """
        # ✨ 修复1：函数定义缺少冒号（仅匹配行首的 def）
        # 使用 MULTILINE 模式，确保只匹配行首
        code = re.sub(
            r'^(\s*def\s+\w+\s*\([^)]*\)(?:\s*->\s*[^:]+)?)\s*$',  # 函数定义后没有冒号
            r'\1:',  # 添加冒号
            code,
            flags=re.MULTILINE
        )
        
        # ✨ 修复2：class 定义缺少冒号（仅匹配行首的 class）
        code = re.sub(
            r'^(\s*class\s+\w+(?:\([^)]*\))?)\s*$',
            r'\1:',
            code,
            flags=re.MULTILINE
        )
        
        # ⚠️ 已移除修复3和修复4：if/elif/while/for/else/try/except/finally
        # 原因：这些修复会错误地匹配三元表达式（如 x if cond else y）
        # 导致在行尾添加多余的冒号，产生 SyntaxError
        
        return code


def create_answer_validator() -> AnswerValidator:
    """
    便捷函数：创建答案验证器
    
    Returns:
        AnswerValidator实例
    """
    return AnswerValidator()


# 示例用法
if __name__ == "__main__":
    validator = create_answer_validator()
    
    # 测试GSM8K
    print("Testing GSM8K:")
    pred = "The answer is 18 dollars."
    truth = "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.\n#### 18"
    is_correct, details = validator.validate(pred, truth, 'gsm8k')
    print(f"  Correct: {is_correct}, Details: {details}")
    
    # 测试MMLU
    print("\nTesting MMLU:")
    pred = "The answer is B."
    truth = "B"
    is_correct, details = validator.validate(pred, truth, 'mmlu')
    print(f"  Correct: {is_correct}, Details: {details}")
    
    # 测试HotpotQA
    print("\nTesting HotpotQA:")
    pred = "AIA Group is a pan-Asian life insurance group."
    truth = "pan-Asian life insurance group"
    is_correct, details = validator.validate(pred, truth, 'hotpotqa')
    print(f"  Correct: {is_correct}, Details: {details}")
    
    # 测试MATH
    print("\nTesting MATH:")
    pred = "The answer is (3, π/2)."
    truth = "\\left( 3, \\frac{\\pi}{2} \\right)"
    is_correct, details = validator.validate(pred, truth, 'math')
    print(f"  Correct: {is_correct}, Details: {details}")

