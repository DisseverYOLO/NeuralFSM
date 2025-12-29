"""
Protected Neural Multi-Agent System Training Script
带保护机制的神经多智能体系统训练脚本

在原始train_neural_mas.py基础上集成保护机制
"""

import asyncio
import torch
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import json

# 导入原始训练器
from neural_fsm_mas.train_neural_mas import NeuralMASTrainer

# 导入保护机制
from neural_fsm_mas.defense_mechanisms import ProtectedTGN


class ProtectedNeuralMASTrainer(NeuralMASTrainer):
    """
    带保护机制的神经多智能体系统训练器
    
    扩展原始NeuralMASTrainer,添加保护机制:
    1. 使用ProtectedTGN包装原始TGN
    2. 计算保护损失
    3. 追踪异常分数和保护优先级
    4. 可视化保护效果
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 mmlu_data_path: str,
                 output_dir: str = "./neural_mas_outputs"):
        
        # 调用父类初始化
        super().__init__(config, mmlu_data_path, output_dir)
        
        # 保护机制配置
        self.protection_config = config.get('protection', {})
        self.use_protection = config.get('use_protection', False)
        
        # 保护统计
        self.protection_stats = {}
        self.learned_weights_history = {}
        
        print(f"🛡️  保护机制: {'已启用' if self.use_protection else '未启用'}")
    
    def create_domain_topology(self, domain: str, task_description: str):
        """
        创建带保护机制的拓扑结构
        
        重写父类方法,在TGN创建后包装为ProtectedTGN
        """
        # 调用父类创建拓扑
        topology_manager = super().create_domain_topology(domain, task_description)
        
        # 如果启用保护机制,包装TGN
        if self.use_protection and topology_manager.use_neural_temporal_graph:
            print(f"🛡️  为领域 {domain} 启用保护机制...")
            
            # 获取通信拓扑图
            communication_graph = topology_manager.communication_topology
            
            # 包装为ProtectedTGN
            protected_tgn = ProtectedTGN(
                tgn_model=topology_manager.neural_temporal_graph,
                feature_dim=self.config.get('feature_dim', 384),
                graph=communication_graph,
                w_betweenness=self.protection_config.get('w_betweenness', 0.6),
                w_pagerank=self.protection_config.get('w_pagerank', 0.4),
                lambda_freq=self.protection_config.get('lambda_freq', 0.3),
                lambda_semantic=self.protection_config.get('lambda_semantic', 0.7),
                weight_hidden_dim=self.protection_config.get('weight_hidden_dim', 64),
                lambda_protect=self.protection_config.get('lambda_protect', 0.1),
                lambda_reg=self.protection_config.get('lambda_reg', 0.01),
                learnable_weights=self.protection_config.get('learnable_weights', True)
            )
            
            # 替换原始TGN
            topology_manager.neural_temporal_graph = protected_tgn
            
            print(f"  ✅ 保护机制已集成到拓扑管理器")
        
        return topology_manager
    
    async def _train_epoch(self, 
                          topology_manager,
                          training_batches: List[Dict],
                          optimizer: torch.optim.Optimizer,
                          domain: str) -> tuple[float, float]:
        """
        训练一个epoch (带保护机制)
        
        重写父类方法,添加保护损失计算
        """
        
        # 检查是否使用保护机制
        is_protected = (self.use_protection and 
                       hasattr(topology_manager.neural_temporal_graph, 'compute_loss'))
        
        if is_protected:
            topology_manager.neural_temporal_graph.train()
        else:
            if topology_manager.use_neural_temporal_graph:
                topology_manager.neural_temporal_graph.train()
        
        topology_manager.compatibility_graph_network.train()
        
        total_loss = 0.0
        total_task_loss = 0.0
        total_protection_loss = 0.0
        total_correct = 0
        total_questions = 0
        
        # 初始化历史记录 (用于异常检测)
        message_counts = {}
        embeddings = {}
        
        for batch_idx, batch in enumerate(training_batches):
            batch_loss = 0.0
            batch_task_loss = 0.0
            batch_protect_loss = 0.0
            batch_correct = 0
            
            for question_data in batch['questions']:
                formatted_question = self.mmlu_processor.format_question_for_agents(question_data)
                
                try:
                    # 执行多智能体推理
                    agent_responses, log_probs = await topology_manager.execute_multi_agent_reasoning(
                        task_input=formatted_question,
                        num_interaction_rounds=self.config.get('num_rounds', 3)
                    )
                    
                    if agent_responses:
                        evaluation = self.mmlu_processor.evaluate_agent_response(
                            str(agent_responses[0]), 
                            question_data.get('answer', '')
                        )
                        
                        # 计算任务奖励
                        reward = 1.0 if evaluation['is_correct'] else 0.0
                        
                        # 确保log_probs是tensor
                        if not isinstance(log_probs, torch.Tensor):
                            log_probs = torch.tensor(log_probs, dtype=torch.float32, 
                                                    requires_grad=True)
                        
                        # 策略梯度损失
                        task_loss = -log_probs * reward
                        batch_task_loss += task_loss
                        
                        # 如果使用保护机制,添加保护损失
                        if is_protected:
                            try:
                                # 更新异常检测历史
                                for agent_id in range(topology_manager.num_agents):
                                    msg_count = np.random.randint(1, 5)  # 模拟消息数
                                    message_counts[agent_id] = msg_count
                                    
                                    # 更新保护机制的历史
                                    topology_manager.neural_temporal_graph.update_anomaly_history(
                                        agent_id=agent_id,
                                        message_count=msg_count
                                    )
                                
                                # 这里可以添加实际的保护损失计算
                                # 简化版: 使用L2正则化作为保护损失
                                protect_weight = self.protection_config.get('lambda_protect', 0.1)
                                protection_loss = torch.tensor(0.0, requires_grad=True)
                                for param in topology_manager.neural_temporal_graph.parameters():
                                    if param.requires_grad:
                                        protection_loss = protection_loss + torch.sum(param ** 2)
                                
                                protection_loss = protect_weight * protection_loss
                                batch_protect_loss += protection_loss.item()
                                batch_task_loss += protection_loss
                                
                            except Exception as e:
                                print(f"⚠️  保护损失计算失败: {e}")
                        
                        batch_loss = batch_task_loss
                        
                        if evaluation['is_correct']:
                            batch_correct += 1
                    
                    total_questions += 1
                    
                except Exception as e:
                    print(f"⚠️  训练错误: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # 反向传播
            if batch_loss != 0:
                optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in optimizer.param_groups[0]['params']], 
                    max_norm=1.0
                )
                optimizer.step()
                
                total_loss += batch_loss.item() if isinstance(batch_loss, torch.Tensor) else batch_loss
                total_task_loss += batch_task_loss if not isinstance(batch_task_loss, torch.Tensor) else batch_task_loss.item()
                total_protection_loss += batch_protect_loss
            
            total_correct += batch_correct
            
            if (batch_idx + 1) % 10 == 0:
                loss_val = batch_loss.item() if isinstance(batch_loss, torch.Tensor) else batch_loss
                print(f"    Batch {batch_idx + 1}/{len(training_batches)}, "
                      f"Loss: {loss_val:.4f} (Task: {batch_task_loss:.4f}, Protect: {batch_protect_loss:.4f}), "
                      f"Acc: {batch_correct}/{len(batch['questions'])}")
        
        avg_loss = total_loss / len(training_batches) if training_batches else 0.0
        avg_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
        
        # 打印损失分解
        if is_protected and len(training_batches) > 0:
            avg_task_loss = total_task_loss / len(training_batches)
            avg_protect_loss = total_protection_loss / len(training_batches)
            print(f"  💡 损失分解: 总损失={avg_loss:.4f}, 任务损失={avg_task_loss:.4f}, 保护损失={avg_protect_loss:.4f}")
        
        return avg_loss, avg_accuracy
    
    async def train_domain_topology(self, 
                                  domain: str, 
                                  topology_manager,
                                  num_epochs: int = 100) -> Dict[str, Any]:
        """
        训练特定领域的拓扑结构 (带保护统计)
        """
        print(f"🎯 开始训练领域 {domain} 的拓扑结构...")
        
        # 调用父类训练
        training_results = await super().train_domain_topology(
            domain, topology_manager, num_epochs
        )
        
        # 记录学习到的权重 (如果使用保护机制)
        if (self.use_protection and 
            hasattr(topology_manager.neural_temporal_graph, 'get_learned_weights')):
            
            learned_weights = topology_manager.neural_temporal_graph.get_learned_weights()
            self.learned_weights_history[domain] = learned_weights
            
            print(f"\n🔧 领域 {domain} 的学习权重:")
            print(f"  中心性: {learned_weights.get('centrality', {})}")
            print(f"  异常: {learned_weights.get('anomaly', {})}")
        
        return training_results
    
    def get_learned_weights(self) -> Dict[str, Dict]:
        """获取所有领域的学习权重"""
        return self.learned_weights_history
    
    def visualize_protection_priorities(self, output_dir: str):
        """
        可视化保护优先级
        
        Args:
            output_dir: 输出目录
        """
        if not self.use_protection:
            print("⚠️  保护机制未启用,无法可视化")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for domain, topology_manager in self.domain_topologies.items():
            if not hasattr(topology_manager.neural_temporal_graph, 'visualize_protection'):
                continue
            
            print(f"📊 可视化领域 {domain} 的保护优先级...")
            
            # 获取通信拓扑图
            graph = topology_manager.communication_topology
            
            # 可视化
            viz_path = output_path / f"protection_priority_{domain}.png"
            topology_manager.neural_temporal_graph.visualize_protection(
                graph=graph,
                save_path=str(viz_path)
            )
            
            print(f"  ✅ 已保存到 {viz_path}")


__all__ = ['ProtectedNeuralMASTrainer']

