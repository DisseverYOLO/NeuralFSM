"""
Text embedding module
Text Embedding Module

Use Sentence Transformers to convert text (agent descriptions, state descriptions) into vector representations
Referenced from GDesigner's profile_embedding.py implementation
"""

import numpy as np
import torch
from typing import Union, List
from sentence_transformers import SentenceTransformer


class TextEmbeddingModel:
    """
    Text embedding model

    Use a pretrained Sentence Transformer model to convert text into vectors
    """
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Sentence Transformer model name
                       Default: 'sentence-transformers/all-MiniLM-L6-v2' (384 dimensions)
                       Other options:
                       - 'all-mpnet-base-v2' (768 dimensions, higher quality)
                       - 'paraphrase-multilingual-MiniLM-L12-v2' (384 dimensions, supports Chinese)
        """
        print(f"📦 Loading text embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded, embedding dimension: {self.embedding_dim}")
    
    def encode_text(self, text: Union[str, List[str]], 
                    normalize: bool = False,
                    convert_to_tensor: bool = False) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode text into vectors.
        
        Args:
            text: A single text string or a list of text strings
            normalize: Whether to normalize vectors (L2 norm)
            convert_to_tensor: Whether to return a PyTorch tensor
            
        Returns:
            Embedding vectors, shape: [embedding_dim] or [batch_size, embedding_dim]
        """
        embeddings = self.model.encode(
            text,
            normalize_embeddings=normalize,
            convert_to_tensor=convert_to_tensor,
            show_progress_bar=False
        )
        
        return embeddings
    
    def encode_agents(self, agents: List[dict]) -> torch.Tensor:
        """
        Generate embedding vectors for a list of agents.
        
        Args:
            agents: Agent configuration list, each containing 'name' and 'system_prompt'
            
        Returns:
            Agent embedding tensor, shape: [num_agents, embedding_dim]
        """
        # Extract key agent information: name + system prompt
        agent_texts = []
        for agent in agents:
            name = agent.get('name', 'UnnamedAgent')
            system_prompt = agent.get('system_prompt', '')
            # Combine name and prompt as the full description
            full_description = f"{name}: {system_prompt}"
            agent_texts.append(full_description)
        
        # Batch encode
        embeddings = self.encode_text(agent_texts, convert_to_tensor=True)
        
        return embeddings
    
    def encode_states(self, states: List[dict]) -> torch.Tensor:
        """
        Generate embedding vectors for a list of FSM states.
        
        Args:
            states: State configuration list, each containing 'name' and 'action'
            
        Returns:
            State embedding tensor, shape: [num_states, embedding_dim]
        """
        # Extract key state information: name + action description
        state_texts = []
        for state in states:
            name = state.get('name', 'UnnamedState')
            action = state.get('action', '')
            # Combine name and action as the full description
            full_description = f"{name}: {action}"
            state_texts.append(full_description)
        
        # Batch encode
        embeddings = self.encode_text(state_texts, convert_to_tensor=True)
        
        return embeddings
    
    def encode_query(self, query: str) -> torch.Tensor:
        """
        Generate an embedding vector for the task query.
        
        Args:
            query: Task description or question
            
        Returns:
            Query embedding tensor, shape: [embedding_dim]
        """
        embedding = self.encode_text(query, convert_to_tensor=True)
        return embedding
    
    def combine_features_with_query(self, 
                                    node_features: torch.Tensor,
                                    query: str) -> torch.Tensor:
        """
        Combine node features with query embeddings (following GDesigner's construct_new_features).
        
        Args:
            node_features: Node feature tensor, shape: [num_nodes, feature_dim]
            query: Query text
            
        Returns:
            Combined feature tensor, shape: [num_nodes, feature_dim + embedding_dim]
        """
        # Encode the query
        query_embedding = self.encode_query(query)
        
        # Expand the query embedding to all nodes
        num_nodes = node_features.size(0)
        query_embedding = query_embedding.unsqueeze(0).repeat(num_nodes, 1)
        
        # Concatenate node features and query embedding
        combined_features = torch.cat([node_features, query_embedding], dim=1)
        
        return combined_features


# Global singleton instance
_embedding_model_instance = None


def get_embedding_model(model_name: str = 'sentence-transformers/all-MiniLM-L6-v2') -> TextEmbeddingModel:
    """
    Get the global embedding model instance (singleton pattern).
    
    Args:
        model_name: Model name
        
    Returns:
        TextEmbeddingModel instance
    """
    global _embedding_model_instance
    
    if _embedding_model_instance is None:
        _embedding_model_instance = TextEmbeddingModel(model_name)
    
    return _embedding_model_instance


def get_sentence_embedding(sentence: str) -> np.ndarray:
    """
    Get text embeddings (compatible with GDesigner API).
    
    Args:
        sentence: Text string
        
    Returns:
        Embedding vector (NumPy array)
    """
    model = get_embedding_model()
    return model.encode_text(sentence, convert_to_tensor=False)


# Export functions
__all__ = [
    'TextEmbeddingModel',
    'get_embedding_model',
    'get_sentence_embedding'
]
