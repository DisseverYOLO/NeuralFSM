"""
FSM Multi-Agent System Generator


"""

import json
import random
import numpy as np
import torch
import networkx as nx
from typing import Dict, List, Any, Tuple, Optional
import sys
from pathlib import Path

# Add path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from baseclass.FSM_Gen import Generate_Agent_Description, Generate_FSM
from baseclass.MultiAgent import MultiAgentSystem
from neural_fsm_mas.temporal_networks.neural_temporal_graph import NeuralTemporalGraph
from neural_fsm_mas.agent_topology.multi_agent_topology import MultiAgentTopologyManager
from neural_fsm_mas.embeddings import get_embedding_model


class FSMMultiAgentSystemGenerator:
    """
    FSM multi-agent system generator

    Core features:
    1. Use MetaAgent to generate agent descriptions and FSM states
    2. Randomly sample the state transition topology
    3. Randomly sample the listening-agent communication topology
    4. Use TGN to learn optimal state transition and communication paths
    """
    
    def __init__(self, 
                 use_neural_learning: bool = True,
                 memory_dimension: int = 128,
                 temporal_dimension: int = 32,
                 random_seed: int = 42,
                 embedding_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        
        self.use_neural_learning = use_neural_learning
        self.memory_dimension = memory_dimension
        self.temporal_dimension = temporal_dimension
        
        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Initialize the embedding model (following GDesigner)
        print(f"🔤 Initializing text embedding model...")
        self.embedding_model = get_embedding_model(embedding_model_name)
        self.embedding_dim = self.embedding_model.embedding_dim
        print(f"✅ Embedding model loaded, dimension: {self.embedding_dim}")
        
        # Store generated components
        self.generated_agents = None
        self.generated_fsm = None
        self.state_topology_graph = None
        self.listening_topology_graph = None
        self.neural_state_learner = None
        self.neural_communication_learner = None
        
    def generate_complete_fsm_mas(self, 
                                 task_description: str,
                                 available_tools: List[str] = None) -> Dict[str, Any]:
        """
        Generate a complete FSM multi-agent system.
        
        Args:
            task_description: Task description
            available_tools: List of available tools
            
        Returns:
            Complete FSM-MAS system configuration
        """
        print("🚀 Starting generation of FSM multi-agent system...")
        
        # Default tool list
        if available_tools is None:
            available_tools = [
                "code_interpreter", "web_search", "calculator", 
                "knowledge_retrieval", "logical_reasoning", "analysis"
            ]
        
        # Step 1: Use MetaAgent to generate agent descriptions
        print("🤖 Step 1: Generating agent role descriptions...")
        self.generated_agents, _ = Generate_Agent_Description(task_description, available_tools)
        print(f"✅ Generated {len(self.generated_agents)} agents")
        
        # Step 2: Use MetaAgent to generate FSM states
        print("🔄 Step 2: Generating finite-state machine...")
        self.generated_fsm, _ = Generate_FSM(task_description, self.generated_agents)
        if self.generated_fsm is None:
            raise ValueError("FSM generation failed")
        print(f"✅ Generated {len(self.generated_fsm['states'])} states")
        
        # Step 3: Randomly sample the state transition topology
        print("🕸️ Step 3: Randomly sampling the state transition topology...")
        self.state_topology_graph = self._sample_state_transition_topology()
        print(f"✅ Sampled the state transition topology with {len(self.state_topology_graph.edges)} edges")
        
        # Step 4: Randomly sample the listening-agent communication topology
        print("📡 Step 4: Randomly sampling the listening-agent communication topology...")
        self.listening_topology_graph = self._sample_listening_communication_topology()
        print(f"✅ Sampled the communication topology with {len(self.listening_topology_graph.edges)} edges")
        
        # Step 5: If neural learning is enabled, create TGN learners
        if self.use_neural_learning:
            print("🧠 Step 5: Initializing TGN neural learners...")
            self._initialize_neural_learners()
            print("✅ TGN learners initialized")
        
        # Build the complete system configuration
        fsm_mas_config = {
            "task_description": task_description,
            "agents": self.generated_agents,
            "fsm": self.generated_fsm,
            "state_topology": self._graph_to_dict(self.state_topology_graph),
            "listening_topology": self._graph_to_dict(self.listening_topology_graph),
            "neural_learning_enabled": self.use_neural_learning,
            "system_metadata": {
                "num_agents": len(self.generated_agents),
                "num_states": len(self.generated_fsm['states']),
                "num_transitions": len(self.generated_fsm['transitions']),
                "state_topology_edges": len(self.state_topology_graph.edges),
                "listening_topology_edges": len(self.listening_topology_graph.edges)
            }
        }
        
        print("🎉 FSM multi-agent system generation completed!")
        return fsm_mas_config
    
    def _sample_state_transition_topology(self) -> nx.DiGraph:
        """
        Randomly sample the state transition topology.
        
        Add random extra connections based on the original FSM transitions.
        """
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add all states as nodes
        states = self.generated_fsm['states']
        for state in states:
            G.add_node(state['state_id'], 
                      agent_id=state['agent_id'],
                      instruction=state['instruction'],
                      is_initial=state['is_initial'],
                      is_final=state['is_final'])
        
        # Add the original transition relations
        for transition in self.generated_fsm['transitions']:
            G.add_edge(transition['from_state'], 
                      transition['to_state'],
                      condition=transition['condition'],
                      edge_type='original')
        
        # Randomly add extra transition connections (for learning)
        state_ids = [state['state_id'] for state in states]
        num_additional_edges = random.randint(1, len(state_ids) // 2)
        
        for _ in range(num_additional_edges):
            from_state = random.choice(state_ids)
            to_state = random.choice(state_ids)
            
            # Avoid self-loops and duplicate edges
            if from_state != to_state and not G.has_edge(from_state, to_state):
                G.add_edge(from_state, to_state,
                          condition="learnable_transition",
                          edge_type='sampled')
        
        return G
    
    def _sample_listening_communication_topology(self) -> nx.Graph:
        """
        Randomly sample the listening-agent communication topology.
        
        Add random communication connections based on listener relations in the FSM.
        """
        # Create an undirected graph (communication is bidirectional)
        G = nx.Graph()
        
        # Add all agents as nodes
        for agent in self.generated_agents:
            G.add_node(agent['agent_id'], 
                      name=agent['name'],
                      system_prompt=agent['system_prompt'],
                      tools=agent['tools'])
        
        # Add communication edges based on listener relations in the FSM
        for state in self.generated_fsm['states']:
            current_agent = state['agent_id']
            listeners = state.get('listener', [])
            
            for listener_id in listeners:
                if listener_id != current_agent:  # Avoid self-connections
                    G.add_edge(current_agent, listener_id,
                             edge_type='listening',
                             state_context=state['state_id'])
        
        # Randomly add extra communication connections
        agent_ids = [agent['agent_id'] for agent in self.generated_agents]
        num_additional_edges = random.randint(1, len(agent_ids))
        
        for _ in range(num_additional_edges):
            agent1 = random.choice(agent_ids)
            agent2 = random.choice(agent_ids)
            
            # Avoid self-loops and duplicate edges
            if agent1 != agent2 and not G.has_edge(agent1, agent2):
                G.add_edge(agent1, agent2,
                          edge_type='sampled_communication',
                          weight=random.uniform(0.1, 1.0))
        
        return G
    
    def _initialize_neural_learners(self):
        """
        Initialize TGN neural learners using the embedding dimension.
        
        """
        # State transition learner
        num_states = len(self.generated_fsm['states'])
        print(f"🧠 Initializing state transition TGN (feature_dim={self.embedding_dim})...")
        self.neural_state_learner = NeuralTemporalGraph(
            agent_feature_dim=self.embedding_dim,  # Use the embedding dimension
            memory_dimension=self.memory_dimension,
            temporal_dimension=self.temporal_dimension,
            agent_count=num_states,
            network_layers=2
        )
        
        # Agent communication learner
        num_agents = len(self.generated_agents)
        print(f"🧠 Initializing agent communication TGN (feature_dim={self.embedding_dim})...")
        self.neural_communication_learner = NeuralTemporalGraph(
            agent_feature_dim=self.embedding_dim,  # Use the embedding dimension
            memory_dimension=self.memory_dimension,
            temporal_dimension=self.temporal_dimension,
            agent_count=num_agents,
            network_layers=2
        )
    
    def learn_optimal_topologies(self, 
                                training_episodes: int = 100,
                                learning_rate: float = 0.001) -> Dict[str, Any]:
        """
        Compute auxiliary MSE reconstruction loss for combination with policy gradients.
        
        Args:
            training_episodes: Number of training episodes (currently only a single loss is computed)
            learning_rate: Learning rate (unused, kept only for compatibility)
            
        Returns:
            A dictionary containing MSE loss values and topology structures
        """
        if not self.use_neural_learning:
            raise ValueError("Neural learning is not enabled")
        
        print(f"🎯 Computing auxiliary MSE reconstruction loss (for the combined loss)...")
        
        # Prepare features
        state_features = self._prepare_state_features()
        agent_features = self._prepare_agent_features()
        
        # Compute reconstruction loss (without updating parameters)
        state_reconstruction_loss = self._train_state_transitions(
            state_features, None, 0
        )
        communication_reconstruction_loss = self._train_communication_paths(
            agent_features, None, 0
        )
        
        # Generate optimized topology structures
        optimized_state_topology = self._generate_optimized_state_topology()
        optimized_communication_topology = self._generate_optimized_communication_topology()
        
        learning_results = {
            "training_episodes": 1,
            "final_state_loss": state_reconstruction_loss,  # Auxiliary MSE loss
            "final_communication_loss": communication_reconstruction_loss,  # Auxiliary MSE loss
            "state_loss_history": [state_reconstruction_loss],
            "communication_loss_history": [communication_reconstruction_loss],
            "optimized_state_topology": optimized_state_topology,
            "optimized_communication_topology": optimized_communication_topology,
            "optimization_method": "combined_policy_gradient_and_reconstruction"
        }
        
        print(f"✅ Auxiliary MSE loss computation finished: State={state_reconstruction_loss:.4f}, Comm={communication_reconstruction_loss:.4f}")
        return learning_results
    
    def _prepare_state_features(self) -> torch.Tensor:
        """
        
        Use Sentence Transformer to embed state descriptions as vectors.
        """
        if self.generated_fsm is None:
            raise ValueError("FSM not generated yet")
        
        # Convert state descriptions into vectors with the embedding model
        state_embeddings = self.embedding_model.encode_states(self.generated_fsm['states'])
        
        print(f"📊 State feature preparation completed: {state_embeddings.shape}")
        return state_embeddings
    
    def _prepare_agent_features(self) -> torch.Tensor:
        """

        
        Use Sentence Transformer to embed agent descriptions as vectors.
        """
        if self.generated_agents is None:
            raise ValueError("Agents not generated yet")
        
        # Convert agent descriptions into vectors with the embedding model
        agent_embeddings = self.embedding_model.encode_agents(self.generated_agents)
        
        print(f"📊 Agent feature preparation completed: {agent_embeddings.shape}")
        return agent_embeddings
    
    def _prepare_features_with_query(self, 
                                     node_features: torch.Tensor,
                                     query: str) -> torch.Tensor:
        """

        
        Args:
            node_features: Node features [num_nodes, feature_dim]
            query: Task query text
            
        Returns:
            Combined features [num_nodes, feature_dim + embedding_dim]
        """
        combined_features = self.embedding_model.combine_features_with_query(
            node_features, query
        )
        
        print(f"📊 Query feature combination completed: {combined_features.shape}")
        return combined_features
    
    def compute_auxiliary_reconstruction_loss(self, 
                                            node_features: torch.Tensor,
                                            edge_index: torch.Tensor,
                                            timestamps: torch.Tensor,
                                            learner_network: torch.nn.Module) -> float:
        """
        Compute auxiliary MSE reconstruction loss.
        
        This loss acts as a regularization term to help TGN learn stable and meaningful node representations.
        When combined with policy gradient loss, it can improve training stability.
        
        Args:
            node_features: Node features
            edge_index: Edge indices
            timestamps: Timestamps
            learner_network: TGN network
            
        Returns:
            Reconstruction loss value
        """
        # Forward pass
        evolved_features = learner_network(
            node_features, edge_index, timestamps=timestamps
        )
        
        # Compute reconstruction loss
        reconstruction_loss = torch.nn.functional.mse_loss(evolved_features, node_features)
        
        return reconstruction_loss.item()
    
    def _train_state_transitions(self, 
                               state_features: torch.Tensor,
                               optimizer: torch.optim.Optimizer,
                               episode: int) -> float:
        """
 
        """
        edge_index = self._build_state_edge_index()
        timestamps = torch.tensor([episode] * len(self.generated_fsm['states']), dtype=torch.float)
        
        return self.compute_auxiliary_reconstruction_loss(
            state_features, edge_index, timestamps, self.neural_state_learner
        )
    
    def _train_communication_paths(self, 
                                 agent_features: torch.Tensor,
                                 optimizer: torch.optim.Optimizer,
                                 episode: int) -> float:
        """

        """
        edge_index = self._build_communication_edge_index()
        timestamps = torch.tensor([episode] * len(self.generated_agents), dtype=torch.float)
        
        return self.compute_auxiliary_reconstruction_loss(
            agent_features, edge_index, timestamps, self.neural_communication_learner
        )
    
    def _build_state_edge_index(self) -> torch.Tensor:
        """Build edge indices for state transitions."""
        edges = []
        state_id_to_idx = {state['state_id']: i for i, state in enumerate(self.generated_fsm['states'])}
        
        for edge in self.state_topology_graph.edges():
            from_idx = state_id_to_idx[edge[0]]
            to_idx = state_id_to_idx[edge[1]]
            edges.append([from_idx, to_idx])
        
        if not edges:
            # If there are no edges, create a self-loop
            edges = [[0, 0]]
        
        return torch.tensor(edges).t().contiguous()
    
    def _build_communication_edge_index(self) -> torch.Tensor:
        """Build edge indices for communication."""
        edges = []
        agent_id_to_idx = {agent['agent_id']: i for i, agent in enumerate(self.generated_agents)}
        
        for edge in self.listening_topology_graph.edges():
            from_idx = agent_id_to_idx[edge[0]]
            to_idx = agent_id_to_idx[edge[1]]
            edges.append([from_idx, to_idx])
            edges.append([to_idx, from_idx])  # Undirected graph, add reverse edge
        
        if not edges:
            # If there are no edges, create a self-loop
            edges = [[0, 0]]
        
        return torch.tensor(edges).t().contiguous()
    
    def _generate_optimized_state_topology(self) -> Dict[str, Any]:
        """Generate optimized state topology."""
        # Use connection probabilities learned by TGN
        with torch.no_grad():
            state_features = self._prepare_state_features()
            communication_probs = self.neural_state_learner.compute_communication_probabilities(state_features)
        
        # Build optimized topology based on probabilities
        optimized_edges = []
        state_ids = [state['state_id'] for state in self.generated_fsm['states']]
        
        for i, from_state in enumerate(state_ids):
            for j, to_state in enumerate(state_ids):
                if i != j and communication_probs[i, j] > 0.5:  # Threshold filtering
                    optimized_edges.append({
                        "from_state": from_state,
                        "to_state": to_state,
                        "probability": communication_probs[i, j].item(),
                        "learned": True
                    })
        
        return {"edges": optimized_edges, "type": "optimized_state_transitions"}
    
    def _generate_optimized_communication_topology(self) -> Dict[str, Any]:
        """Generate optimized communication topology."""
        # Use connection probabilities learned by TGN
        with torch.no_grad():
            agent_features = self._prepare_agent_features()
            communication_probs = self.neural_communication_learner.compute_communication_probabilities(agent_features)
        
        # Build optimized topology based on probabilities
        optimized_edges = []
        agent_ids = [agent['agent_id'] for agent in self.generated_agents]
        
        for i, from_agent in enumerate(agent_ids):
            for j, to_agent in enumerate(agent_ids):
                if i != j and communication_probs[i, j] > 0.5:  # Threshold filtering
                    optimized_edges.append({
                        "from_agent": from_agent,
                        "to_agent": to_agent,
                        "probability": communication_probs[i, j].item(),
                        "learned": True
                    })
        
        return {"edges": optimized_edges, "type": "optimized_communication"}
    
    def _graph_to_dict(self, graph: nx.Graph) -> Dict[str, Any]:
        """Convert a NetworkX graph into dictionary format."""
        return {
            "nodes": list(graph.nodes(data=True)),
            "edges": list(graph.edges(data=True)),
            "type": "networkx_graph"
        }
    
    def save_fsm_mas_system(self, output_path: str):
        """Save the complete FSM-MAS system."""
        if not all([self.generated_agents, self.generated_fsm, 
                   self.state_topology_graph, self.listening_topology_graph]):
            raise ValueError("FSM-MAS system not fully generated")
        
        system_data = {
            "agents": self.generated_agents,
            "fsm": self.generated_fsm,
            "state_topology": self._graph_to_dict(self.state_topology_graph),
            "listening_topology": self._graph_to_dict(self.listening_topology_graph),
            "neural_learning_enabled": self.use_neural_learning
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(system_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 FSM-MAS system saved to {output_path}")
    
    def create_executable_system(self) -> MultiAgentSystem:
        """
        Create an executable multi-agent system.
        
        Convert the generated and optimized FSM-MAS into an executable MultiAgentSystem.
        
        Returns:
            MultiAgentSystem: Executable multi-agent system
        """
        if not all([self.generated_agents, self.generated_fsm]):
            raise ValueError("FSM-MAS system not fully generated. Please call generate_complete_fsm_mas first.")
        
        print("🔧 Creating executable multi-agent system...")
        
        # Create a MultiAgentSystem instance
        executable_system = MultiAgentSystem(
            agents_json=self.generated_agents,
            states_json=self.generated_fsm
        )
        
        print("✅ Executable system created")
        return executable_system
    
    def execute_task(self, task_input: str, max_transitions: int = 10) -> Tuple[str, float]:
        """
        Execute a task.
        
        Args:
            task_input: Task input
            max_transitions: Maximum number of state transitions
            
        Returns:
            Tuple[str, float]: (execution result, total cost)
        """
        if not all([self.generated_agents, self.generated_fsm]):
            raise ValueError("FSM-MAS system not fully generated. Please call generate_complete_fsm_mas first.")
        
        print(f"🚀 Starting task execution: {task_input}")
        
        # Create executable system
        executable_system = self.create_executable_system()
        
        # Execute task
        result, cost = executable_system.start(task_input, max_transitions)
        
        print(f"✅ Task execution completed, cost: {cost}")
        return result, cost
    
    def generate_learn_and_execute(self, 
                                 task_description: str,
                                 task_input: str,
                                 available_tools: List[str] = None,
                                 training_episodes: int = 50,
                                 max_transitions: int = 10) -> Dict[str, Any]:
        """
        Full generate-learn-execute workflow.
        
        Args:
            task_description: Task description (used to generate the FSM)
            task_input: Concrete task input (used for execution)
            available_tools: Available tools
            training_episodes: Number of TGN training episodes
            max_transitions: Maximum number of state transitions during execution
            
        Returns:
            Complete results containing all generation, learning, and execution information
        """
        print("🌟 Starting the full generate-learn-execute workflow...")
        
        # Step 1: Generate the FSM-MAS system
        fsm_mas_config = self.generate_complete_fsm_mas(task_description, available_tools)
        
        # Step 2: TGN learning optimization
        learning_results = self.learn_optimal_topologies(training_episodes)
        
        # Step 3: Execute the task
        execution_result, execution_cost = self.execute_task(task_input, max_transitions)
        
        # Build the complete result
        complete_results = {
            'task_description': task_description,
            'task_input': task_input,
            'fsm_mas_config': fsm_mas_config,
            'learning_results': learning_results,
            'execution_result': execution_result,
            'execution_cost': execution_cost,
            'workflow': 'generate_learn_execute'
        }
        
        print("🎉 Full workflow completed!")
        return complete_results
    
    def load_fsm_mas_system(self, input_path: str):
        """Load an FSM-MAS system."""
        with open(input_path, 'r', encoding='utf-8') as f:
            system_data = json.load(f)
        
        self.generated_agents = system_data['agents']
        self.generated_fsm = system_data['fsm']
        
        # Rebuild graph structures
        self.state_topology_graph = nx.DiGraph()
        state_topo = system_data['state_topology']
        self.state_topology_graph.add_nodes_from(state_topo['nodes'])
        self.state_topology_graph.add_edges_from(state_topo['edges'])
        
        self.listening_topology_graph = nx.Graph()
        listening_topo = system_data['listening_topology']
        self.listening_topology_graph.add_nodes_from(listening_topo['nodes'])
        self.listening_topology_graph.add_edges_from(listening_topo['edges'])
        
        self.use_neural_learning = system_data.get('neural_learning_enabled', False)
        
        if self.use_neural_learning:
            self._initialize_neural_learners()
        
        print(f"📂 FSM-MAS system loaded from {input_path}")


# Convenience functions
def create_fsm_mas_generator(use_neural_learning: bool = True,
                           memory_dimension: int = 128,
                           temporal_dimension: int = 32) -> FSMMultiAgentSystemGenerator:
    """Create an FSM-MAS generator."""
    return FSMMultiAgentSystemGenerator(
        use_neural_learning=use_neural_learning,
        memory_dimension=memory_dimension,
        temporal_dimension=temporal_dimension
    )


def generate_and_learn_fsm_mas(task_description: str,
                             available_tools: List[str] = None,
                             training_episodes: int = 100,
                             output_path: str = None) -> Dict[str, Any]:
    """Generate and learn an FSM-MAS system in one call."""
    
    # Create generator
    generator = create_fsm_mas_generator()
    
    # Generate the complete system
    fsm_mas_config = generator.generate_complete_fsm_mas(task_description, available_tools)
    
    # Learn optimal topologies
    learning_results = generator.learn_optimal_topologies(training_episodes)
    
    # Merge results
    complete_results = {
        **fsm_mas_config,
        "learning_results": learning_results
    }
    
    # Save results
    if output_path:
        generator.save_fsm_mas_system(output_path)
    
    return complete_results


def generate_learn_and_execute_fsm_mas(task_description: str,
                                     task_input: str,
                                     available_tools: List[str] = None,
                                     training_episodes: int = 50,
                                     max_transitions: int = 10,
                                     output_path: str = None) -> Dict[str, Any]:
    """Generate, learn, and execute an FSM-MAS system in one call."""
    
    # Create generator
    generator = create_fsm_mas_generator()
    
    # Execute the full workflow
    complete_results = generator.generate_learn_and_execute(
        task_description=task_description,
        task_input=task_input,
        available_tools=available_tools,
        training_episodes=training_episodes,
        max_transitions=max_transitions
    )
    
    # Save results
    if output_path:
        generator.save_fsm_mas_system(output_path)
    
    return complete_results


# Export main classes and functions
__all__ = [
    'FSMMultiAgentSystemGenerator',
    'create_fsm_mas_generator',
    'generate_and_learn_fsm_mas',
    'generate_learn_and_execute_fsm_mas'
]
