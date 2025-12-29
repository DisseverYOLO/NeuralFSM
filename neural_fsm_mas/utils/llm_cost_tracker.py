"""
LLM API Cost Tracker
LLM API成本追踪器

功能:
1. 维护各种LLM模型的API价格清单
2. 追踪每次LLM调用的token使用和成本
3. 计算episode/batch的总成本
4. 支持成本损失函数计算
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class LLMPricing:
    """LLM模型定价信息"""
    model_name: str
    input_price_per_1k: float  # 美元/1K tokens
    output_price_per_1k: float  # 美元/1K tokens
    context_window: int = 0  # 上下文窗口大小
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """计算单次调用成本（美元）"""
        input_cost = (input_tokens / 1000.0) * self.input_price_per_1k
        output_cost = (output_tokens / 1000.0) * self.output_price_per_1k
        return input_cost + output_cost


@dataclass
class LLMCall:
    """单次LLM调用记录"""
    model_name: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    agent_id: Optional[str] = None
    state_id: Optional[str] = None
    timestamp: Optional[float] = None


class LLMPricingRegistry:
    """LLM定价注册表"""
    
    # 官方/公开定价清单（会随时间变化；以代码仓库当前版本为准）
    PRICING_TABLE = {
        # OpenAI Models
        'gpt-4o': LLMPricing(
            model_name='gpt-4o',
            input_price_per_1k=0.0025,
            output_price_per_1k=0.010,
            context_window=128000
        ),
        'gpt-5-nano': LLMPricing(
            model_name='gpt-5-nano',
            input_price_per_1k=0.00005,
            output_price_per_1k=0.0004,
            context_window=128000
        ),
        'gpt-4o-mini': LLMPricing(
            model_name='gpt-4o-mini',
            input_price_per_1k=0.00015,
            output_price_per_1k=0.0006,
            context_window=128000
        ),
        # GPT-4.1 Nano (OpenAI)
        # Pricing reference (public): input $0.10 / 1M, output $0.40 / 1M
        # -> per 1K: input 0.0001, output 0.0004
        'gpt-4.1-nano': LLMPricing(
            model_name='gpt-4.1-nano',
            input_price_per_1k=0.0001,
            output_price_per_1k=0.0004,
            context_window=128000
        ),
        # Alias form sometimes used in OpenAI docs / clients
        'gpt-4-1-nano': LLMPricing(
            model_name='gpt-4-1-nano',
            input_price_per_1k=0.0001,
            output_price_per_1k=0.0004,
            context_window=128000
        ),
        'gpt-4-turbo': LLMPricing(
            model_name='gpt-4-turbo',
            input_price_per_1k=0.010,
            output_price_per_1k=0.030,
            context_window=128000
        ),
        'gpt-4': LLMPricing(
            model_name='gpt-4',
            input_price_per_1k=0.030,
            output_price_per_1k=0.060,
            context_window=8192
        ),
        'gpt-3.5-turbo': LLMPricing(
            model_name='gpt-3.5-turbo',
            input_price_per_1k=0.0005,
            output_price_per_1k=0.0015,
            context_window=16385
        ),

        # xAI Grok Models
        # Pricing reference (public reporting): Grok 3 Mini (standard) input $0.30 / 1M, output $0.50 / 1M
        # -> per 1K: input 0.0003, output 0.0005
        'grok-3-mini': LLMPricing(
            model_name='grok-3-mini',
            input_price_per_1k=0.0003,
            output_price_per_1k=0.0005,
            context_window=128000
        ),
        
        # Anthropic Claude Models
        'claude-3-opus': LLMPricing(
            model_name='claude-3-opus',
            input_price_per_1k=0.015,
            output_price_per_1k=0.075,
            context_window=200000
        ),
        'claude-3-sonnet': LLMPricing(
            model_name='claude-3-sonnet',
            input_price_per_1k=0.003,
            output_price_per_1k=0.015,
            context_window=200000
        ),
        'claude-3-haiku': LLMPricing(
            model_name='claude-3-haiku',
            input_price_per_1k=0.00025,
            output_price_per_1k=0.00125,
            context_window=200000
        ),
        
        # Google Gemini Models
        'gemini-pro': LLMPricing(
            model_name='gemini-pro',
            input_price_per_1k=0.00025,
            output_price_per_1k=0.0005,
            context_window=32000
        ),
        'gemini-1.5-pro': LLMPricing(
            model_name='gemini-1.5-pro',
            input_price_per_1k=0.00125,
            output_price_per_1k=0.005,
            context_window=1000000
        ),
        'gemini-1.5-flash': LLMPricing(
            model_name='gemini-1.5-flash',
            input_price_per_1k=0.000075,
            output_price_per_1k=0.0003,
            context_window=1000000
        ),
    }
    
    @classmethod
    def get_pricing(cls, model_name: str) -> LLMPricing:
        """获取模型定价信息"""
        # 尝试精确匹配
        if model_name in cls.PRICING_TABLE:
            return cls.PRICING_TABLE[model_name]
        
        # 尝试模糊匹配（例如 gpt-4o-mini-2024-07-18 匹配 gpt-4o-mini）
        for key in cls.PRICING_TABLE.keys():
            if key in model_name:
                return cls.PRICING_TABLE[key]
        
        # 默认使用gpt-5-nano定价
        print(f"⚠️  Warning: Model '{model_name}' not found in pricing table, using gpt-5-nano as default")
        return cls.PRICING_TABLE['gpt-5-nano']
    
    @classmethod
    def list_all_models(cls) -> List[str]:
        """列出所有支持的模型"""
        return list(cls.PRICING_TABLE.keys())
    
    @classmethod
    def print_pricing_table(cls):
        """打印定价表"""
        print("\n" + "=" * 100)
        print("LLM API Pricing Table (USD per 1K tokens)")
        print("=" * 100)
        print(f"{'Model':<20} {'Input ($/1K)':<15} {'Output ($/1K)':<15} {'Context Window':<15}")
        print("-" * 100)
        
        for model_name, pricing in sorted(cls.PRICING_TABLE.items()):
            print(f"{model_name:<20} ${pricing.input_price_per_1k:<14.6f} ${pricing.output_price_per_1k:<14.6f} {pricing.context_window:<15,}")
        
        print("=" * 100 + "\n")


class LLMCostTracker:
    """LLM成本追踪器"""
    
    def __init__(self, default_model: str = 'gpt-5-nano'):
        """
        初始化成本追踪器
        
        Args:
            default_model: 默认模型名称
        """
        self.default_model = default_model
        self.call_history: List[LLMCall] = []
        self.episode_costs: List[float] = []  # 每个episode的成本
        
    def track_call(self, 
                   input_tokens: int, 
                   output_tokens: int,
                   model_name: Optional[str] = None,
                   agent_id: Optional[str] = None,
                   state_id: Optional[str] = None,
                   timestamp: Optional[float] = None) -> float:
        """
        追踪一次LLM调用
        
        Args:
            input_tokens: 输入token数
            output_tokens: 输出token数
            model_name: 模型名称（可选，默认使用default_model）
            agent_id: Agent ID（可选）
            state_id: State ID（可选）
            timestamp: 时间戳（可选）
        
        Returns:
            本次调用的成本（美元）
        """
        model_name = model_name or self.default_model
        pricing = LLMPricingRegistry.get_pricing(model_name)
        cost = pricing.calculate_cost(input_tokens, output_tokens)
        
        # 记录调用
        call = LLMCall(
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            agent_id=agent_id,
            state_id=state_id,
            timestamp=timestamp
        )
        self.call_history.append(call)
        
        return cost
    
    def start_episode(self):
        """开始一个新的episode"""
        self.episode_start_idx = len(self.call_history)
    
    def end_episode(self) -> float:
        """
        结束当前episode并计算成本
        
        Returns:
            本episode的总成本（美元）
        """
        episode_calls = self.call_history[self.episode_start_idx:]
        episode_cost = sum(call.cost_usd for call in episode_calls)
        self.episode_costs.append(episode_cost)
        return episode_cost
    
    def get_total_cost(self) -> float:
        """获取总成本（美元）"""
        return sum(call.cost_usd for call in self.call_history)
    
    def get_average_cost_per_episode(self) -> float:
        """获取每个episode的平均成本（美元）"""
        if not self.episode_costs:
            return 0.0
        return sum(self.episode_costs) / len(self.episode_costs)
    
    def get_cost_by_agent(self) -> Dict[str, float]:
        """按Agent统计成本"""
        agent_costs = {}
        for call in self.call_history:
            if call.agent_id:
                agent_costs[call.agent_id] = agent_costs.get(call.agent_id, 0.0) + call.cost_usd
        return agent_costs
    
    def get_cost_by_state(self) -> Dict[str, float]:
        """按State统计成本"""
        state_costs = {}
        for call in self.call_history:
            if call.state_id:
                state_costs[call.state_id] = state_costs.get(call.state_id, 0.0) + call.cost_usd
        return state_costs
    
    def get_statistics(self) -> Dict[str, any]:
        """获取统计信息"""
        return {
            'total_calls': len(self.call_history),
            'total_cost_usd': self.get_total_cost(),
            'total_input_tokens': sum(call.input_tokens for call in self.call_history),
            'total_output_tokens': sum(call.output_tokens for call in self.call_history),
            'episodes': len(self.episode_costs),
            'avg_cost_per_episode': self.get_average_cost_per_episode(),
            'cost_by_agent': self.get_cost_by_agent(),
            'cost_by_state': self.get_cost_by_state()
        }
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 80)
        print("LLM Cost Tracker Statistics")
        print("=" * 80)
        print(f"Total Calls:              {stats['total_calls']}")
        print(f"Total Cost:               ${stats['total_cost_usd']:.6f} USD")
        print(f"Total Input Tokens:       {stats['total_input_tokens']:,}")
        print(f"Total Output Tokens:      {stats['total_output_tokens']:,}")
        print(f"Episodes:                 {stats['episodes']}")
        print(f"Avg Cost per Episode:     ${stats['avg_cost_per_episode']:.6f} USD")
        
        if stats['cost_by_agent']:
            print("\nCost by Agent:")
            for agent_id, cost in sorted(stats['cost_by_agent'].items()):
                print(f"  Agent {agent_id}: ${cost:.6f} USD")
        
        if stats['cost_by_state']:
            print("\nCost by State:")
            for state_id, cost in sorted(stats['cost_by_state'].items()):
                print(f"  State {state_id}: ${cost:.6f} USD")
        
        print("=" * 80 + "\n")
    
    def save_to_file(self, filepath: str):
        """保存追踪数据到文件"""
        data = {
            'default_model': self.default_model,
            'statistics': self.get_statistics(),
            'call_history': [
                {
                    'model_name': call.model_name,
                    'input_tokens': call.input_tokens,
                    'output_tokens': call.output_tokens,
                    'cost_usd': call.cost_usd,
                    'agent_id': call.agent_id,
                    'state_id': call.state_id,
                    'timestamp': call.timestamp
                }
                for call in self.call_history
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_from_file(self, filepath: str):
        """从文件加载追踪数据"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.default_model = data['default_model']
        self.call_history = [
            LLMCall(**call_data)
            for call_data in data['call_history']
        ]
        
        # 重建episode_costs（简化版）
        self.episode_costs = []


# ========== 便捷函数 ==========

def create_cost_tracker(model_name: str = 'gpt-5-nano') -> LLMCostTracker:
    """创建成本追踪器"""
    return LLMCostTracker(default_model=model_name)


# ========== 全局追踪器（供LLM客户端直接写入） ==========

_GLOBAL_COST_TRACKER: Optional[LLMCostTracker] = None


def set_global_cost_tracker(tracker: Optional[LLMCostTracker]):
    """设置全局LLM成本追踪器（例如由集成模块在训练初始化时注入）"""
    global _GLOBAL_COST_TRACKER
    _GLOBAL_COST_TRACKER = tracker


def get_global_cost_tracker() -> Optional[LLMCostTracker]:
    """获取全局LLM成本追踪器"""
    return _GLOBAL_COST_TRACKER


def print_pricing_table():
    """打印LLM定价表"""
    LLMPricingRegistry.print_pricing_table()


def estimate_cost(input_tokens: int, output_tokens: int, model_name: str = 'gpt-5-nano') -> float:
    """估算成本"""
    pricing = LLMPricingRegistry.get_pricing(model_name)
    return pricing.calculate_cost(input_tokens, output_tokens)


# ========== 测试示例 ==========

if __name__ == "__main__":
    # 打印定价表
    print_pricing_table()
    
    # 创建追踪器
    tracker = create_cost_tracker('gpt-4o-mini')
    
    # 模拟一些调用
    print("Simulating LLM calls...")
    tracker.start_episode()
    tracker.track_call(input_tokens=500, output_tokens=150, agent_id="0", state_id="0")
    tracker.track_call(input_tokens=600, output_tokens=200, agent_id="1", state_id="1")
    tracker.track_call(input_tokens=450, output_tokens=100, agent_id="2", state_id="2")
    episode_cost_1 = tracker.end_episode()
    print(f"Episode 1 cost: ${episode_cost_1:.6f}")
    
    tracker.start_episode()
    tracker.track_call(input_tokens=550, output_tokens=180, agent_id="0", state_id="0")
    tracker.track_call(input_tokens=620, output_tokens=220, agent_id="1", state_id="1")
    episode_cost_2 = tracker.end_episode()
    print(f"Episode 2 cost: ${episode_cost_2:.6f}")
    
    # 打印统计
    tracker.print_statistics()
    
    # 成本估算示例
    print("\n" + "=" * 80)
    print("Cost Estimation Examples:")
    print("=" * 80)
    print(f"gpt-4o-mini (1000 input + 500 output):  ${estimate_cost(1000, 500, 'gpt-4o-mini'):.6f}")
    print(f"gpt-4o      (1000 input + 500 output):  ${estimate_cost(1000, 500, 'gpt-4o'):.6f}")
    print(f"gpt-4-turbo (1000 input + 500 output):  ${estimate_cost(1000, 500, 'gpt-4-turbo'):.6f}")
    print("=" * 80 + "\n")

