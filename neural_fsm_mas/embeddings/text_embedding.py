"""
文本嵌入模块
Text Embedding Module

使用Sentence Transformers将文本（智能体描述、状态描述）转换为向量表示
参考GDesigner的profile_embedding.py实现
"""

import numpy as np
import torch
from typing import Union, List
from sentence_transformers import SentenceTransformer


class TextEmbeddingModel:
    """
    文本嵌入模型
    
    使用预训练的Sentence Transformer模型将文本转换为向量
    """
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        初始化嵌入模型
        
        Args:
            model_name: Sentence Transformer模型名称
                       默认: 'sentence-transformers/all-MiniLM-L6-v2' (384维)
                       其他选项:
                       - 'all-mpnet-base-v2' (768维，更高质量)
                       - 'paraphrase-multilingual-MiniLM-L12-v2' (384维，支持中文)
        """
        print(f"📦 加载文本嵌入模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ 模型加载完成，嵌入维度: {self.embedding_dim}")
    
    def encode_text(self, text: Union[str, List[str]], 
                    normalize: bool = False,
                    convert_to_tensor: bool = False) -> Union[np.ndarray, torch.Tensor]:
        """
        将文本编码为向量
        
        Args:
            text: 单个文本字符串或文本列表
            normalize: 是否归一化向量（L2范数）
            convert_to_tensor: 是否返回PyTorch张量
            
        Returns:
            嵌入向量，shape: [embedding_dim] 或 [batch_size, embedding_dim]
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
        为智能体列表生成嵌入向量
        
        Args:
            agents: 智能体配置列表，每个包含 'name' 和 'system_prompt'
            
        Returns:
            智能体嵌入张量，shape: [num_agents, embedding_dim]
        """
        # 提取智能体的关键信息：名称 + 系统提示词
        agent_texts = []
        for agent in agents:
            name = agent.get('name', 'UnnamedAgent')
            system_prompt = agent.get('system_prompt', '')
            # 组合名称和提示词作为完整描述
            full_description = f"{name}: {system_prompt}"
            agent_texts.append(full_description)
        
        # 批量编码
        embeddings = self.encode_text(agent_texts, convert_to_tensor=True)
        
        return embeddings
    
    def encode_states(self, states: List[dict]) -> torch.Tensor:
        """
        为FSM状态列表生成嵌入向量
        
        Args:
            states: 状态配置列表，每个包含 'name' 和 'action'
            
        Returns:
            状态嵌入张量，shape: [num_states, embedding_dim]
        """
        # 提取状态的关键信息：名称 + 动作描述
        state_texts = []
        for state in states:
            name = state.get('name', 'UnnamedState')
            action = state.get('action', '')
            # 组合名称和动作作为完整描述
            full_description = f"{name}: {action}"
            state_texts.append(full_description)
        
        # 批量编码
        embeddings = self.encode_text(state_texts, convert_to_tensor=True)
        
        return embeddings
    
    def encode_query(self, query: str) -> torch.Tensor:
        """
        为任务查询生成嵌入向量
        
        Args:
            query: 任务描述或问题
            
        Returns:
            查询嵌入张量，shape: [embedding_dim]
        """
        embedding = self.encode_text(query, convert_to_tensor=True)
        return embedding
    
    def combine_features_with_query(self, 
                                    node_features: torch.Tensor,
                                    query: str) -> torch.Tensor:
        """
        将节点特征与查询嵌入结合（参考GDesigner的construct_new_features）
        
        Args:
            node_features: 节点特征张量，shape: [num_nodes, feature_dim]
            query: 查询文本
            
        Returns:
            组合特征张量，shape: [num_nodes, feature_dim + embedding_dim]
        """
        # 编码查询
        query_embedding = self.encode_query(query)
        
        # 将查询嵌入扩展到所有节点
        num_nodes = node_features.size(0)
        query_embedding = query_embedding.unsqueeze(0).repeat(num_nodes, 1)
        
        # 拼接节点特征和查询嵌入
        combined_features = torch.cat([node_features, query_embedding], dim=1)
        
        return combined_features


# 全局单例实例
_embedding_model_instance = None


def get_embedding_model(model_name: str = 'sentence-transformers/all-MiniLM-L6-v2') -> TextEmbeddingModel:
    """
    获取全局嵌入模型实例（单例模式）
    
    Args:
        model_name: 模型名称
        
    Returns:
        TextEmbeddingModel实例
    """
    global _embedding_model_instance
    
    if _embedding_model_instance is None:
        _embedding_model_instance = TextEmbeddingModel(model_name)
    
    return _embedding_model_instance


def get_sentence_embedding(sentence: str) -> np.ndarray:
    """
    获取文本嵌入（兼容GDesigner的API）
    
    Args:
        sentence: 文本字符串
        
    Returns:
        嵌入向量（numpy数组）
    """
    model = get_embedding_model()
    return model.encode_text(sentence, convert_to_tensor=False)


# 导出函数
__all__ = [
    'TextEmbeddingModel',
    'get_embedding_model',
    'get_sentence_embedding'
]

