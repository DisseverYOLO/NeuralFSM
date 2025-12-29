"""
基于执行效率和准确率的状态描述动态优化
State-Description Optimization Based on Efficiency and Accuracy

核心理念：
1. 准确率维度：追踪在失败问题中频繁出现的状态
2. 效率维度：追踪在达到最大转移次数时频繁访问的状态
3. 综合优化：同时考虑准确率和效率问题

优化触发：
- 某状态在失败问题中出现次数 >= 阈值 → 优化（提高准确性）
- 某状态在低效执行中出现次数 >= 阈值 → 优化（提高效率）
"""

from typing import Dict, Tuple, Optional, Set, List
from collections import defaultdict


class StateDescriptionOptimizer:
    """
    基于执行效率和准确率的状态描述优化器
    
    工作流程：
    1. 追踪失败问题中访问的状态（准确率问题）
    2. 追踪达到最大转移次数时访问的状态（效率问题）
    3. 对问题状态动态增强描述
    """
    
    def __init__(self, 
                 domain: str,
                 max_transitions_threshold: int = 3,
                 accuracy_enhancement_threshold: int = 5,
                 efficiency_enhancement_threshold: int = 3):
        """
        Args:
            domain: 领域名称
            max_transitions_threshold: 触发效率优化的次数（达到N次最大转移后优化）
            accuracy_enhancement_threshold: 触发准确率优化的失败次数
            efficiency_enhancement_threshold: 触发效率优化的低效次数
        """
        self.domain = domain
        self.max_transitions_threshold = max_transitions_threshold
        self.accuracy_enhancement_threshold = accuracy_enhancement_threshold
        self.efficiency_enhancement_threshold = efficiency_enhancement_threshold
        
        # 状态访问统计
        self.state_in_failures: Dict[str, int] = defaultdict(int)  # 在失败问题中的出现次数
        self.state_in_max_transitions: Dict[str, int] = defaultdict(int)  # 在低效执行中的出现次数
        self.state_total_visits: Dict[str, int] = defaultdict(int)  # 总访问次数
        
        # 整体统计
        self.total_questions = 0
        self.failed_questions = 0
        self.max_transitions_count = 0  # 达到最大转移次数的问题数
        
        # 增强状态缓存 {state_name: (enhancement_type, enhancement_text)}
        self.enhancement_cache: Dict[str, Tuple[str, str]] = {}
        
        print(f"✨ 状态描述优化器已启用（{domain}）")
        print(f"   准确率阈值: {accuracy_enhancement_threshold}次失败")
        print(f"   效率阈值: {max_transitions_threshold}次达到最大转移后优化")
    
    def get_enhanced_description(self, 
                                 state_name: str,
                                 original_description: str) -> Tuple[str, bool]:
        """
        获取增强后的状态描述
        
        Args:
            state_name: 状态名称
            original_description: 原始状态描述
        
        Returns:
            (增强后的描述, 是否被增强)
        """
        if state_name in self.enhancement_cache:
            enhancement_type, enhancement = self.enhancement_cache[state_name]
            enhanced_desc = f"{enhancement}\n\n{original_description}"
            return enhanced_desc, True
        
        return original_description, False
    
    def record_execution(self,
                        visited_states: List[str],
                        is_correct: bool,
                        reached_max_transitions: bool):
        """
        记录一次FSM执行的结果
        
        Args:
            visited_states: 访问过的状态名称序列（List，保留重复访问）
            is_correct: 最终答案是否正确
            reached_max_transitions: 是否达到最大转移次数
        """
        self._do_record(visited_states, is_correct, reached_max_transitions)
    
    def record_state_execution(self,
                              visited_states: List[str],
                              is_correct: bool,
                              reached_max_transitions: bool):
        """别名：与 record_execution() 完全相同，供训练代码调用"""
        self._do_record(visited_states, is_correct, reached_max_transitions)
    
    def _do_record(self,
                  visited_states: List[str],
                  is_correct: bool,
                  reached_max_transitions: bool):
        """实际记录逻辑"""
        self.total_questions += 1
        
        # ✨ 统计每个状态的访问次数（包括重复访问）
        # 例如：[A, B, C, B, D] → A访问1次, B访问2次, C访问1次, D访问1次
        for state_name in visited_states:
            self.state_total_visits[state_name] += 1
        
        # 记录失败情况（准确率维度）
        # ✨ 如果问题失败，状态序列中所有状态（包括重复）都计入失败统计
        if not is_correct:
            self.failed_questions += 1
            for state_name in visited_states:
                self.state_in_failures[state_name] += 1
        
        # 记录低效执行（效率维度）
        # ✨ 如果达到最大转移，状态序列中所有状态（包括重复）都计入低效统计
        if reached_max_transitions:
            self.max_transitions_count += 1
            for state_name in visited_states:
                self.state_in_max_transitions[state_name] += 1
        
        # 每10个问题评估一次是否需要优化
        if self.total_questions % 10 == 0:
            self._update_enhancements()
    
    def _update_enhancements(self):
        """更新增强策略"""
        # 1. 检查准确率问题（在失败中频繁出现的状态）
        for state_name, failure_count in self.state_in_failures.items():
            if failure_count >= self.accuracy_enhancement_threshold:
                # 计算失败率
                total_visits = self.state_total_visits[state_name]
                failure_rate = failure_count / total_visits if total_visits > 0 else 0
                
                if state_name not in self.enhancement_cache and failure_rate > 0.3:
                    # 添加准确率增强
                    enhancement = self._generate_accuracy_enhancement(
                        state_name, failure_count, failure_rate
                    )
                    self.enhancement_cache[state_name] = ('accuracy', enhancement)
                    print(f"  ⚠️  状态 '{state_name}' 在 {failure_count} 次失败中出现（失败率 {failure_rate:.1%}）- 添加准确性增强")
        
        # 2. 检查效率问题（在达到最大转移次数时频繁出现的状态）
        if self.max_transitions_count >= self.max_transitions_threshold:
            for state_name, max_trans_count in self.state_in_max_transitions.items():
                if max_trans_count >= self.efficiency_enhancement_threshold:
                    total_visits = self.state_total_visits[state_name]
                    inefficiency_rate = max_trans_count / total_visits if total_visits > 0 else 0
                    
                    # 如果还没有增强，或者效率问题更严重
                    if state_name not in self.enhancement_cache and inefficiency_rate > 0.2:
                        enhancement = self._generate_efficiency_enhancement(
                            state_name, max_trans_count, inefficiency_rate
                        )
                        self.enhancement_cache[state_name] = ('efficiency', enhancement)
                        print(f"  ⚠️  状态 '{state_name}' 在 {max_trans_count} 次低效执行中出现 - 添加效率增强")
                    elif state_name in self.enhancement_cache:
                        # 已有准确率增强，添加效率提示
                        current_type, current_enhancement = self.enhancement_cache[state_name]
                        if current_type == 'accuracy' and inefficiency_rate > 0.2:
                            # 组合增强
                            efficiency_hint = self._generate_efficiency_enhancement(
                                state_name, max_trans_count, inefficiency_rate
                            )
                            combined = f"{current_enhancement}\n{efficiency_hint}"
                            self.enhancement_cache[state_name] = ('combined', combined)
                            print(f"  ⚠️  状态 '{state_name}' 存在准确率和效率问题 - 组合增强")
    
    def _generate_accuracy_enhancement(self,
                                       state_name: str,
                                       failure_count: int,
                                       failure_rate: float) -> str:
        """生成准确率增强提示"""
        if failure_rate > 0.6:
            return (f"⚠️ ACCURACY ALERT: This state appears in {failure_count} failed attempts "
                   f"({failure_rate:.0%} failure rate). "
                   f"CRITICAL: Double-check your work and verify all details carefully.")
        else:
            return (f"⚠️ NOTICE: This state has appeared in {failure_count} failed attempts "
                   f"({failure_rate:.0%} failure rate). "
                   f"Please focus on accuracy and avoid common mistakes.")
    
    def _generate_efficiency_enhancement(self,
                                         state_name: str,
                                         max_trans_count: int,
                                         inefficiency_rate: float) -> str:
        """生成效率增强提示"""
        return (f"⚡ EFFICIENCY ALERT: This state appears in {max_trans_count} cases "
               f"where maximum transitions were reached ({inefficiency_rate:.0%} of visits). "
               f"Focus on being CONCISE and DIRECT - avoid unnecessary steps.")
    
    def get_performance_summary(self) -> Dict:
        """获取性能摘要"""
        # 找出问题最严重的状态
        problem_states = []
        
        for state_name in set(list(self.state_in_failures.keys()) + 
                             list(self.state_in_max_transitions.keys())):
            failure_count = self.state_in_failures.get(state_name, 0)
            max_trans_count = self.state_in_max_transitions.get(state_name, 0)
            total_visits = self.state_total_visits.get(state_name, 0)
            
            if total_visits >= 5:  # 至少访问5次
                failure_rate = failure_count / total_visits if total_visits > 0 else 0
                inefficiency_rate = max_trans_count / total_visits if total_visits > 0 else 0
                
                problem_score = failure_rate * 2 + inefficiency_rate  # 准确率权重更高
                
                problem_states.append({
                    'state_name': state_name,
                    'total_visits': total_visits,
                    'failure_count': failure_count,
                    'failure_rate': failure_rate,
                    'max_trans_count': max_trans_count,
                    'inefficiency_rate': inefficiency_rate,
                    'problem_score': problem_score,
                    'is_enhanced': state_name in self.enhancement_cache,
                    'enhancement_type': self.enhancement_cache[state_name][0] if state_name in self.enhancement_cache else None
                })
        
        # 按问题严重程度排序
        problem_states.sort(key=lambda x: x['problem_score'], reverse=True)
        
        return {
            'total_questions': self.total_questions,
            'failed_questions': self.failed_questions,
            'overall_accuracy': (self.total_questions - self.failed_questions) / self.total_questions if self.total_questions > 0 else 0,
            'max_transitions_count': self.max_transitions_count,
            'max_transitions_rate': self.max_transitions_count / self.total_questions if self.total_questions > 0 else 0,
            'total_states_tracked': len(self.state_total_visits),
            'enhanced_states': len(self.enhancement_cache),
            'problem_states': problem_states[:10]  # 前10个问题状态
        }


def create_prompt_optimizer(domain: str,
                           max_transitions_threshold: int = 3,
                           accuracy_enhancement_threshold: int = 5,
                           efficiency_enhancement_threshold: int = 3) -> StateDescriptionOptimizer:
    """
    创建状态描述优化器
    
    Args:
        domain: 领域名称
        max_transitions_threshold: 达到N次最大转移后触发优化
        accuracy_enhancement_threshold: 失败N次后触发准确率优化
        efficiency_enhancement_threshold: 低效N次后触发效率优化
    """
    return StateDescriptionOptimizer(
        domain,
        max_transitions_threshold,
        accuracy_enhancement_threshold,
        efficiency_enhancement_threshold
    )


__all__ = [
    'StateDescriptionOptimizer',
    'create_prompt_optimizer'
]
