"""
Neural Multi-Agent System Training Script

Learns state-transition rules and agent communication networks.

Note: this is a simplified collaborative MAS trainer.
- Uses predefined agent configurations from `DomainPromptManager`
- Multi-round interaction mode with `num_rounds` rounds of collaboration
- Policy-gradient loss optimization

If you need full automatic FSM generation, use `train_fsm_mas.py`.
"""

import asyncio
import argparse
import json
import os
import sys
import time
import torch
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

# Add paths.
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent))

from neural_fsm_mas.temporal_networks.neural_temporal_graph import NeuralTemporalGraph, MultiLayerPerceptron
from neural_fsm_mas.agent_topology.multi_agent_topology import MultiAgentTopologyManager
from neural_fsm_mas.training_data.unified_data_processor import UnifiedDataProcessor
from neural_fsm_mas.domain_prompts.prompt_manager import DomainPromptManager
from baseclass.FSM_Gen import Generate_Agent_Description, Generate_State_Description


class NeuralMASTrainer:
    """
    Neural multi-agent system trainer.
    
    Core features:
    1. Train the TGN network using MMLU data.
    2. Learn optimal agent communication topologies.
    3. Learn FSM state-transition rules.
    4. Support separate training and evaluation across multiple domains.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 dataset_root: str = "./datasets",
                 output_dir: str = "./neural_mas_outputs",
                 mmlu_data_path: str = None):  # Kept for compatibility.
        
        self.config = config
        self.dataset_root = dataset_root if not mmlu_data_path else Path(mmlu_data_path).parent.parent
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the unified data processor.
        self.data_processor = UnifiedDataProcessor(str(self.dataset_root))
        
        # Training state.
        self.training_history = []
        self.domain_topologies = {}
        self.best_models = {}
        
        # Device configuration.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def prepare_training_data(self, domains: List[str] = None):
        """
        Prepare training data.
        
        Args:
            domains: List of datasets to prepare, such as `['mmlu', 'gsm8k', 'humaneval']`.
                If `None`, default to `config['domains']` or only `mmlu`.
        """
        if domains is None:
            domains = self.config.get('domains', ['mmlu'])
        
        print(f"🔄 Preparing training data: {', '.join(domains)}")
        
        statistics = {}
        
        for domain in domains:
            try:
                print(f"\n  Processing {domain.upper()}...")
                
                # Create data splits.
                self.data_processor.create_domain_splits(
                    domain=domain,
                    train_ratio=self.config.get('train_ratio', 0.7),
                    val_ratio=self.config.get('val_ratio', 0.15),
                    test_ratio=self.config.get('test_ratio', 0.15),
                    random_seed=self.config.get('random_seed', 42)
                )
                
                # Get statistics.
                stats = self.data_processor.get_domain_statistics(domain)
                statistics[domain] = stats[domain]
                
                print(f"  ✅ {domain}: {stats[domain]}")
                
            except Exception as e:
                print(f"  ⚠️  Failed to prepare {domain}: {e}")
                statistics[domain] = {'error': str(e)}
        
        print("\n📊 Dataset statistics summary:")
        for domain, stats in statistics.items():
            if 'error' not in stats:
                print(f"  {domain.upper()}: Train={stats.get('train', 0)}, "
                      f"Val={stats.get('val', 0)}, Test={stats.get('test', 0)}")
            else:
                print(f"  {domain.upper()}: ❌ {stats['error']}")
        
        return statistics
    
    def create_domain_topology(self, domain: str, task_description: str = None) -> MultiAgentTopologyManager:
        """
        Create a multi-agent topology for a specific domain.
        
        Args:
            domain: Domain name (`mmlu`, `gsm8k`, `humaneval`).
            task_description: Optional task description. If `None`, it is generated automatically.
        """
        print(f"🏗️ Creating multi-agent topology for domain {domain}...")
        
        # If no task description is provided, get it from the data processor.
        if task_description is None:
            task_description = self.data_processor.get_task_description(domain)
        
        # Prefer agent configurations from the domain prompt set.
        try:
            prompt_set = DomainPromptManager.get_manager(domain)
            agent_names = prompt_set.get_available_roles()
            print(f"✅ Using predefined agent configuration: {agent_names}")
        except Exception as e:
            print("⚠️ Generating agents with the LLM...")
            # Use MetaAgent to generate agent descriptions.
            available_tools = self.config.get('available_tools', [
                'knowledge_retrieval', 'logical_reasoning', 'calculation', 'analysis'
            ])
            
            try:
                agents_description, _ = Generate_Agent_Description(task_description, available_tools)
                agent_names = [agent['name'] for agent in agents_description]
                print(f"✅ Generated {len(agent_names)} agents: {agent_names}")
            except Exception as e:
                print(f"⚠️ Agent generation failed, using the default configuration: {e}")
                # Use the default agent configuration.
                agent_names = self._get_default_agents_for_domain(domain)
        
        # Create the multi-agent topology manager.
        topology_manager = MultiAgentTopologyManager(
            task_domain=domain,
            language_model_name=self.config.get('llm_name', 'gpt-5-nano'),
            agent_role_names=agent_names,
            decision_strategy='final_decision',
            enable_spatial_optimization=True,
            enable_temporal_optimization=True,
            use_neural_temporal_graph=True,
            memory_bank_dimension=self.config.get('memory_dim', 128),
            temporal_encoding_dimension=self.config.get('time_dim', 32)
        )
        
        return topology_manager
    
    def _get_default_agents_for_domain(self, domain: str) -> List[str]:
        """Get the default agent configuration for a domain."""
        domain_agents = {
            "mmlu": ["Knowledge Expert", "Subject Specialist", "Critical Analyzer", "Mathematician"],
            "gsm8k": ["Math Problem Solver", "Problem Analyzer", "Calculation Verifier", "Solution Critic"],
            "humaneval": ["Code Designer", "Code Writer", "Code Reviewer", "Test Engineer"],
            "STEM": ["Mathematical Reasoner", "Scientific Analyzer", "Problem Solver", "Verifier"],
            "Humanities": ["Historical Analyst", "Cultural Expert", "Critical Thinker", "Synthesizer"],
            "Social_Sciences": ["Social Analyst", "Economic Reasoner", "Policy Expert", "Evaluator"],
            "Other": ["General Expert", "Domain Specialist", "Analyst", "Reviewer"]
        }
        return domain_agents.get(domain, ["Expert Agent", "Analyst Agent", "Critic Agent"])
    
    async def train_domain_topology(self, 
                                  domain: str, 
                                  topology_manager: MultiAgentTopologyManager,
                                  num_epochs: int = 100) -> Dict[str, Any]:
        """Train the topology structure for a specific domain."""
        print(f"🎯 Starting topology training for domain {domain}...")
        
        # Prepare training data.
        training_batches = self.data_processor.prepare_multi_agent_training_data(
            domain=domain,
            split="train",
            batch_size=self.config.get('batch_size', 16)
        )
        
        validation_batches = self.data_processor.prepare_multi_agent_training_data(
            domain=domain,
            split="val",
            batch_size=self.config.get('batch_size', 16)
        )
        
        # Set up the optimizer.
        if topology_manager.use_neural_temporal_graph and topology_manager.neural_temporal_graph:
            model_params = (
                list(topology_manager.neural_temporal_graph.parameters()) +
                list(topology_manager.decision_decoder.parameters())
            )
        else:
            model_params = list(topology_manager.compatibility_graph_network.parameters())
        
        optimizer = optim.Adam(model_params, lr=self.config.get('learning_rate', 0.001))
        
        # Training loop.
        training_results = {
            'domain': domain,
            'epoch_losses': [],
            'epoch_accuracies': [],
            'validation_accuracies': [],
            'best_accuracy': 0.0,
            'best_epoch': 0
        }
        
        for epoch in range(num_epochs):
            print(f"📚 Epoch {epoch + 1}/{num_epochs} for domain {domain}")
            
            # Training phase.
            epoch_loss, epoch_accuracy = await self._train_epoch(
                topology_manager, training_batches, optimizer, domain
            )
            
            # Validation phase.
            val_accuracy = await self._validate_epoch(
                topology_manager, validation_batches, domain
            )
            
            # Record results.
            training_results['epoch_losses'].append(epoch_loss)
            training_results['epoch_accuracies'].append(epoch_accuracy)
            training_results['validation_accuracies'].append(val_accuracy)
            
            # Save the best model.
            if val_accuracy > training_results['best_accuracy']:
                training_results['best_accuracy'] = val_accuracy
                training_results['best_epoch'] = epoch
                self._save_best_model(topology_manager, domain, epoch, val_accuracy)
            
            print(f"  📊 Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # Early-stopping check.
            if self._should_early_stop(training_results, patience=10):
                print(f"🛑 Early stopping at epoch {epoch + 1}")
                break
        
        print(f"✅ Finished training for domain {domain}; best validation accuracy: {training_results['best_accuracy']:.4f}")
        return training_results
    
    async def _train_epoch(self, 
                          topology_manager: MultiAgentTopologyManager,
                          training_batches: List[Dict],
                          optimizer: torch.optim.Optimizer,
                          domain: str) -> tuple[float, float]:
        """Train one epoch."""
        
        if topology_manager.use_neural_temporal_graph and topology_manager.neural_temporal_graph:
            topology_manager.neural_temporal_graph.train()
        topology_manager.compatibility_graph_network.train()
        
        total_loss = 0.0
        total_correct = 0
        total_questions = 0
        
        for batch_idx, batch in enumerate(training_batches):
            batch_loss = 0.0
            batch_correct = 0
            
            for question_data in batch['questions']:
                # Format the question.
                formatted_question = self.mmlu_processor.format_question_for_agents(question_data)
                
                try:
                    # Execute multi-agent reasoning.
                    agent_responses, log_probs = await topology_manager.execute_multi_agent_reasoning(
                        task_input=formatted_question,
                        num_interaction_rounds=self.config.get('num_rounds', 3)
                    )
                    
                    # Evaluate the response.
                    if agent_responses:
                        evaluation = self.mmlu_processor.evaluate_agent_response(
                            str(agent_responses[0]), 
                            question_data.get('answer', '')
                        )
                        
                        # Compute the reward.
                        reward = 1.0 if evaluation['is_correct'] else 0.0
                        
                        # Policy-gradient loss.
                        loss = -log_probs * reward
                        batch_loss += loss
                        
                        if evaluation['is_correct']:
                            batch_correct += 1
                    
                    total_questions += 1
                    
                except Exception as e:
                    print(f"⚠️ Training error: {e}")
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
                
                total_loss += batch_loss.item()
            
            total_correct += batch_correct
            
            if (batch_idx + 1) % 10 == 0:
                print(f"    Batch {batch_idx + 1}/{len(training_batches)}, "
                      f"Loss: {batch_loss:.4f}, Acc: {batch_correct}/{len(batch['questions'])}")
        
        avg_loss = total_loss / len(training_batches) if training_batches else 0.0
        avg_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
        
        return avg_loss, avg_accuracy
    
    async def _validate_epoch(self, 
                            topology_manager: MultiAgentTopologyManager,
                            validation_batches: List[Dict],
                            domain: str) -> float:
        """Validate one epoch."""
        
        if topology_manager.use_neural_temporal_graph and topology_manager.neural_temporal_graph:
            topology_manager.neural_temporal_graph.eval()
        topology_manager.compatibility_graph_network.eval()
        
        total_correct = 0
        total_questions = 0
        
        with torch.no_grad():
            for batch in validation_batches:
                for question_data in batch['questions']:
                    formatted_question = self.mmlu_processor.format_question_for_agents(question_data)
                    
                    try:
                        agent_responses, _ = await topology_manager.execute_multi_agent_reasoning(
                            task_input=formatted_question,
                            num_interaction_rounds=self.config.get('num_rounds', 3)
                        )
                        
                        if agent_responses:
                            evaluation = self.mmlu_processor.evaluate_agent_response(
                                str(agent_responses[0]), 
                                question_data.get('answer', '')
                            )
                            
                            if evaluation['is_correct']:
                                total_correct += 1
                        
                        total_questions += 1
                        
                    except Exception as e:
                        print(f"⚠️ Validation error: {e}")
                        continue
        
        accuracy = total_correct / total_questions if total_questions > 0 else 0.0
        return accuracy
    
    def _should_early_stop(self, training_results: Dict, patience: int = 10) -> bool:
        """Check whether early stopping should be triggered."""
        if len(training_results['validation_accuracies']) < patience:
            return False
        
        recent_accuracies = training_results['validation_accuracies'][-patience:]
        return all(acc <= training_results['best_accuracy'] for acc in recent_accuracies)
    
    def _save_best_model(self, 
                        topology_manager: MultiAgentTopologyManager,
                        domain: str,
                        epoch: int,
                        accuracy: float):
        """Save the best model."""
        model_dir = self.output_dir / "best_models" / domain
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state.
        model_state = {
            'epoch': epoch,
            'accuracy': accuracy,
            'domain': domain,
            'config': self.config
        }
        
        if topology_manager.use_neural_temporal_graph and topology_manager.neural_temporal_graph:
            model_state['neural_temporal_graph'] = topology_manager.neural_temporal_graph.state_dict()
        
        model_state['compatibility_graph_network'] = topology_manager.compatibility_graph_network.state_dict()
        model_state['decision_decoder'] = topology_manager.decision_decoder.state_dict()
        
        model_path = model_dir / f"best_model_epoch_{epoch}.pth"
        torch.save(model_state, model_path)
        
        # Save topology configuration.
        topology_config = {
            'agent_role_names': topology_manager.agent_role_names,
            'task_domain': topology_manager.task_domain,
            'num_agents': topology_manager.num_agents
        }
        
        config_path = model_dir / "topology_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(topology_config, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved the best model to {model_path}")
    
    async def train_all_domains(self) -> Dict[str, Dict[str, Any]]:
        """Train all domains."""
        print("🚀 Starting training for the neural multi-agent system across all domains...")
        
        # Prepare data.
        statistics = self.prepare_training_data()
        
        # Train the topology for each domain.
        all_results = {}
        
        for domain in statistics.keys():
            print(f"\n{'='*60}")
            print(f"🎯 Starting training for domain: {domain}")
            print(f"{'='*60}")
            
            # Create the task description.
            task_description = f"Solve {domain} questions from MMLU dataset with multiple choice answers"
            
            # Create the topology manager.
            topology_manager = self.create_domain_topology(domain, task_description)
            self.domain_topologies[domain] = topology_manager
            
            # Train.
            training_results = await self.train_domain_topology(
                domain, 
                topology_manager, 
                num_epochs=self.config.get('num_epochs', 50)
            )
            
            all_results[domain] = training_results
        
        # Save training history.
        self._save_training_results(all_results)
        
        return all_results
    
    def _save_training_results(self, results: Dict[str, Dict[str, Any]]):
        """Save training results."""
        results_path = self.output_dir / "training_results.json"
        
        # Convert NumPy arrays to lists for JSON serialization.
        serializable_results = {}
        for domain, result in results.items():
            serializable_results[domain] = {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in result.items()
            }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Training results saved to {results_path}")
    
    async def evaluate_all_domains(self) -> Dict[str, float]:
        """Evaluate all domains."""
        print("🧪 Starting evaluation for all domains...")
        
        evaluation_results = {}
        
        for domain, topology_manager in self.domain_topologies.items():
            print(f"📝 Evaluating domain: {domain}")
            
            # Prepare test data.
            test_batches = self.mmlu_processor.prepare_multi_agent_training_data(
                domain=domain,
                split="test",
                batch_size=self.config.get('batch_size', 16)
            )
            
            # Evaluate.
            test_accuracy = await self._validate_epoch(topology_manager, test_batches, domain)
            evaluation_results[domain] = test_accuracy
            
            print(f"  📊 {domain} test accuracy: {test_accuracy:.4f}")
        
        return evaluation_results


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Neural Multi-Agent System Training")
    
    parser.add_argument("--mmlu_data_path", type=str, required=True,
                       help="Path to MMLU dataset")
    parser.add_argument("--output_dir", type=str, default="./neural_mas_outputs",
                       help="Output directory for results")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration file")
    
    # Training parameters.
    parser.add_argument("--num_epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--num_rounds", type=int, default=3,
                       help="Number of interaction rounds")
    
    # Model parameters.
    parser.add_argument("--memory_dim", type=int, default=128,
                       help="Memory dimension for TGN")
    parser.add_argument("--time_dim", type=int, default=32,
                       help="Time encoding dimension")
    parser.add_argument("--llm_name", type=str, default="gpt-5-nano",
                       help="Language model name (default: gpt-5-nano)")
    
    return parser.parse_args()


async def main():
    """Main function."""
    args = parse_arguments()
    
    # Build the configuration.
    config = {
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_rounds': args.num_rounds,
        'memory_dim': args.memory_dim,
        'time_dim': args.time_dim,
        'llm_name': args.llm_name,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'random_seed': 42,
        'available_tools': [
            'knowledge_retrieval', 'logical_reasoning', 'calculation', 
            'analysis', 'critical_thinking', 'synthesis'
        ]
    }
    
    # Load configuration from file if provided.
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            file_config = json.load(f)
        config.update(file_config)
    
    print("🌟 Neural multi-agent system training started")
    print(f"📁 MMLU data path: {args.mmlu_data_path}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"⚙️ Configuration: {json.dumps(config, indent=2)}")
    
    # Create the trainer.
    trainer = NeuralMASTrainer(
        config=config,
        mmlu_data_path=args.mmlu_data_path,
        output_dir=args.output_dir
    )
    
    try:
        # Train all domains.
        training_results = await trainer.train_all_domains()
        
        # Evaluate all domains.
        evaluation_results = await trainer.evaluate_all_domains()
        
        # Print the final results.
        print("\n" + "="*80)
        print("🎉 Training complete! Final results:")
        print("="*80)
        
        for domain in training_results.keys():
            train_acc = training_results[domain]['best_accuracy']
            test_acc = evaluation_results.get(domain, 0.0)
            print(f"📊 {domain:15} | Best validation accuracy: {train_acc:.4f} | Test accuracy: {test_acc:.4f}")
        
        # Compute the average accuracy.
        avg_test_acc = np.mean(list(evaluation_results.values()))
        print(f"\n🏆 Average test accuracy: {avg_test_acc:.4f}")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
