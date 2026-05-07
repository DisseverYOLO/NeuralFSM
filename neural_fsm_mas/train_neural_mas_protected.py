"""
Protected Neural Multi-Agent System Training Script

"""

import asyncio
import torch
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import json

# Import the original trainer.
from neural_fsm_mas.train_neural_mas import NeuralMASTrainer

# Import protection mechanisms.
from neural_fsm_mas.defense_mechanisms import ProtectedTGN


class ProtectedNeuralMASTrainer(NeuralMASTrainer):
    """
    Neural multi-agent system trainer with protection mechanisms.
    
    Extends the original `NeuralMASTrainer` by adding:
    1. `ProtectedTGN` wrapping around the original TGN
    2. Protection-loss computation
    3. Tracking of anomaly scores and protection priorities
    4. Visualization of protection effects
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 mmlu_data_path: str,
                 output_dir: str = "./neural_mas_outputs"):
        
        # Call the parent initializer.
        super().__init__(config, mmlu_data_path, output_dir)
        
        # Protection configuration.
        self.protection_config = config.get('protection', {})
        self.use_protection = config.get('use_protection', False)
        
        # Protection statistics.
        self.protection_stats = {}
        self.learned_weights_history = {}
        
        print(f"🛡️  Protection mechanism: {'enabled' if self.use_protection else 'disabled'}")
    
    def create_domain_topology(self, domain: str, task_description: str):
        """
        Create a topology with protection mechanisms.
        
        Overrides the parent method and wraps the TGN with `ProtectedTGN` after creation.
        """
        # Call the parent method to create the topology.
        topology_manager = super().create_domain_topology(domain, task_description)
        
        # Wrap the TGN if protection is enabled.
        if self.use_protection and topology_manager.use_neural_temporal_graph:
            print(f"🛡️  Enabling protection for domain {domain}...")
            
            # Get the communication topology graph.
            communication_graph = topology_manager.communication_topology
            
            # Wrap it as `ProtectedTGN`.
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
            
            # Replace the original TGN.
            topology_manager.neural_temporal_graph = protected_tgn
            
            print("  ✅ Protection mechanism integrated into the topology manager")
        
        return topology_manager
    
    async def _train_epoch(self, 
                          topology_manager,
                          training_batches: List[Dict],
                          optimizer: torch.optim.Optimizer,
                          domain: str) -> tuple[float, float]:
        """
        Train one epoch with protection mechanisms.
        
        Overrides the parent method to add protection-loss computation.
        """
        
        # Check whether protection is being used.
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
        
        # Initialize history for anomaly detection.
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
                    # Execute multi-agent reasoning.
                    agent_responses, log_probs = await topology_manager.execute_multi_agent_reasoning(
                        task_input=formatted_question,
                        num_interaction_rounds=self.config.get('num_rounds', 3)
                    )
                    
                    if agent_responses:
                        evaluation = self.mmlu_processor.evaluate_agent_response(
                            str(agent_responses[0]), 
                            question_data.get('answer', '')
                        )
                        
                        # Compute the task reward.
                        reward = 1.0 if evaluation['is_correct'] else 0.0
                        
                        # Make sure `log_probs` is a tensor.
                        if not isinstance(log_probs, torch.Tensor):
                            log_probs = torch.tensor(log_probs, dtype=torch.float32, 
                                                    requires_grad=True)
                        
                        # Policy-gradient loss.
                        task_loss = -log_probs * reward
                        batch_task_loss += task_loss
                        
                        # Add protection loss if protection is enabled.
                        if is_protected:
                            try:
                                # Update anomaly-detection history.
                                for agent_id in range(topology_manager.num_agents):
                                    msg_count = np.random.randint(1, 5)  # Simulated message count.
                                    message_counts[agent_id] = msg_count
                                    
                                    # Update protection history.
                                    topology_manager.neural_temporal_graph.update_anomaly_history(
                                        agent_id=agent_id,
                                        message_count=msg_count
                                    )
                                
                                # Actual protection-loss computation can be added here.
                                # Simplified version: use L2 regularization as the protection loss.
                                protect_weight = self.protection_config.get('lambda_protect', 0.1)
                                protection_loss = torch.tensor(0.0, requires_grad=True)
                                for param in topology_manager.neural_temporal_graph.parameters():
                                    if param.requires_grad:
                                        protection_loss = protection_loss + torch.sum(param ** 2)
                                
                                protection_loss = protect_weight * protection_loss
                                batch_protect_loss += protection_loss.item()
                                batch_task_loss += protection_loss
                                
                            except Exception as e:
                                print(f"⚠️  Failed to compute protection loss: {e}")
                        
                        batch_loss = batch_task_loss
                        
                        if evaluation['is_correct']:
                            batch_correct += 1
                    
                    total_questions += 1
                    
                except Exception as e:
                    print(f"⚠️  Training error: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Backpropagation.
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
        
        # Print the loss breakdown.
        if is_protected and len(training_batches) > 0:
            avg_task_loss = total_task_loss / len(training_batches)
            avg_protect_loss = total_protection_loss / len(training_batches)
            print(f"  💡 Loss breakdown: total={avg_loss:.4f}, task={avg_task_loss:.4f}, protection={avg_protect_loss:.4f}")
        
        return avg_loss, avg_accuracy
    
    async def train_domain_topology(self, 
                                  domain: str, 
                                  topology_manager,
                                  num_epochs: int = 100) -> Dict[str, Any]:
        """
        Train the topology for a specific domain with protection statistics.
        """
        print(f"🎯 Starting topology training for domain {domain}...")
        
        # Call the parent training method.
        training_results = await super().train_domain_topology(
            domain, topology_manager, num_epochs
        )
        
        # Record learned weights if protection is enabled.
        if (self.use_protection and 
            hasattr(topology_manager.neural_temporal_graph, 'get_learned_weights')):
            
            learned_weights = topology_manager.neural_temporal_graph.get_learned_weights()
            self.learned_weights_history[domain] = learned_weights
            
            print(f"\n🔧 Learned weights for domain {domain}:")
            print(f"  Centrality: {learned_weights.get('centrality', {})}")
            print(f"  Anomaly: {learned_weights.get('anomaly', {})}")
        
        return training_results
    
    def get_learned_weights(self) -> Dict[str, Dict]:
        """Get learned weights for all domains."""
        return self.learned_weights_history
    
    def visualize_protection_priorities(self, output_dir: str):
        """
        Visualize protection priorities.
        
        Args:
            output_dir: Output directory.
        """
        if not self.use_protection:
            print("⚠️  Protection is not enabled, so visualization is unavailable")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for domain, topology_manager in self.domain_topologies.items():
            if not hasattr(topology_manager.neural_temporal_graph, 'visualize_protection'):
                continue
            
            print(f"📊 Visualizing protection priorities for domain {domain}...")
            
            # Get the communication topology graph.
            graph = topology_manager.communication_topology
            
            # Visualize.
            viz_path = output_path / f"protection_priority_{domain}.png"
            topology_manager.neural_temporal_graph.visualize_protection(
                graph=graph,
                save_path=str(viz_path)
            )
            
            print(f"  ✅ Saved to {viz_path}")


__all__ = ['ProtectedNeuralMASTrainer']
