"""
LLM API Cost Tracker
LLM API cost tracker

Features:
1. Maintain the API pricing table for various LLM models
2. Track token usage and cost for each LLM call
3. Compute total cost per episode/batch
4. Support cost loss computation
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class LLMPricing:
    """LLM model pricing information."""
    model_name: str
    input_price_per_1k: float  # USD/1K tokens
    output_price_per_1k: float  # USD/1K tokens
    context_window: int = 0  # Context window size
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost of a single call in USD."""
        input_cost = (input_tokens / 1000.0) * self.input_price_per_1k
        output_cost = (output_tokens / 1000.0) * self.output_price_per_1k
        return input_cost + output_cost


@dataclass
class LLMCall:
    """Record for a single LLM call."""
    model_name: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    agent_id: Optional[str] = None
    state_id: Optional[str] = None
    timestamp: Optional[float] = None


class LLMPricingRegistry:
    """LLM pricing registry."""
    
    # Official/public pricing table (subject to change over time; the current repository version is authoritative)
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
        """Get pricing information for a model."""
        # Try exact matching
        if model_name in cls.PRICING_TABLE:
            return cls.PRICING_TABLE[model_name]
        
        # Try fuzzy matching (for example, gpt-4o-mini-2024-07-18 matches gpt-4o-mini)
        for key in cls.PRICING_TABLE.keys():
            if key in model_name:
                return cls.PRICING_TABLE[key]
        
        # Use gpt-5-nano pricing by default
        print(f"⚠️  Warning: Model '{model_name}' not found in pricing table, using gpt-5-nano as default")
        return cls.PRICING_TABLE['gpt-5-nano']
    
    @classmethod
    def list_all_models(cls) -> List[str]:
        """List all supported models."""
        return list(cls.PRICING_TABLE.keys())
    
    @classmethod
    def print_pricing_table(cls):
        """Print the pricing table."""
        print("\n" + "=" * 100)
        print("LLM API Pricing Table (USD per 1K tokens)")
        print("=" * 100)
        print(f"{'Model':<20} {'Input ($/1K)':<15} {'Output ($/1K)':<15} {'Context Window':<15}")
        print("-" * 100)
        
        for model_name, pricing in sorted(cls.PRICING_TABLE.items()):
            print(f"{model_name:<20} ${pricing.input_price_per_1k:<14.6f} ${pricing.output_price_per_1k:<14.6f} {pricing.context_window:<15,}")
        
        print("=" * 100 + "\n")


class LLMCostTracker:
    """LLM cost tracker."""
    
    def __init__(self, default_model: str = 'gpt-5-nano'):
        """
        Initialize the cost tracker.
        
        Args:
            default_model: Default model name
        """
        self.default_model = default_model
        self.call_history: List[LLMCall] = []
        self.episode_costs: List[float] = []  # Cost for each episode
        
    def track_call(self, 
                   input_tokens: int, 
                   output_tokens: int,
                   model_name: Optional[str] = None,
                   agent_id: Optional[str] = None,
                   state_id: Optional[str] = None,
                   timestamp: Optional[float] = None) -> float:
        """
        Track one LLM call.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model_name: Model name (optional; uses default_model by default)
            agent_id: Agent ID (optional)
            state_id: State ID (optional)
            timestamp: Timestamp (optional)
        
        Returns:
            Cost of this call in USD
        """
        model_name = model_name or self.default_model
        pricing = LLMPricingRegistry.get_pricing(model_name)
        cost = pricing.calculate_cost(input_tokens, output_tokens)
        
        # Record the call
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
        """Start a new episode."""
        self.episode_start_idx = len(self.call_history)
    
    def end_episode(self) -> float:
        """
        End the current episode and compute its cost.
        
        Returns:
            Total cost of the current episode in USD
        """
        episode_calls = self.call_history[self.episode_start_idx:]
        episode_cost = sum(call.cost_usd for call in episode_calls)
        self.episode_costs.append(episode_cost)
        return episode_cost
    
    def get_total_cost(self) -> float:
        """Get the total cost in USD."""
        return sum(call.cost_usd for call in self.call_history)
    
    def get_average_cost_per_episode(self) -> float:
        """Get the average cost per episode in USD."""
        if not self.episode_costs:
            return 0.0
        return sum(self.episode_costs) / len(self.episode_costs)
    
    def get_cost_by_agent(self) -> Dict[str, float]:
        """Get cost statistics by agent."""
        agent_costs = {}
        for call in self.call_history:
            if call.agent_id:
                agent_costs[call.agent_id] = agent_costs.get(call.agent_id, 0.0) + call.cost_usd
        return agent_costs
    
    def get_cost_by_state(self) -> Dict[str, float]:
        """Get cost statistics by state."""
        state_costs = {}
        for call in self.call_history:
            if call.state_id:
                state_costs[call.state_id] = state_costs.get(call.state_id, 0.0) + call.cost_usd
        return state_costs
    
    def get_statistics(self) -> Dict[str, any]:
        """Get summary statistics."""
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
        """Print summary statistics."""
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
        """Save tracking data to a file."""
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
        """Load tracking data from a file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.default_model = data['default_model']
        self.call_history = [
            LLMCall(**call_data)
            for call_data in data['call_history']
        ]
        
        # Rebuild episode_costs (simplified version)
        self.episode_costs = []


# ========== Convenience Functions ==========

def create_cost_tracker(model_name: str = 'gpt-5-nano') -> LLMCostTracker:
    """Create a cost tracker."""
    return LLMCostTracker(default_model=model_name)


# ========== Global Tracker (for direct writes from LLM clients) ==========

_GLOBAL_COST_TRACKER: Optional[LLMCostTracker] = None


def set_global_cost_tracker(tracker: Optional[LLMCostTracker]):
    """Set the global LLM cost tracker (for example, injected by the integration module during training initialization)."""
    global _GLOBAL_COST_TRACKER
    _GLOBAL_COST_TRACKER = tracker


def get_global_cost_tracker() -> Optional[LLMCostTracker]:
    """Get the global LLM cost tracker."""
    return _GLOBAL_COST_TRACKER


def print_pricing_table():
    """Print the LLM pricing table."""
    LLMPricingRegistry.print_pricing_table()


def estimate_cost(input_tokens: int, output_tokens: int, model_name: str = 'gpt-5-nano') -> float:
    """Estimate cost."""
    pricing = LLMPricingRegistry.get_pricing(model_name)
    return pricing.calculate_cost(input_tokens, output_tokens)


# ========== Test Example ==========

if __name__ == "__main__":
    # Print the pricing table
    print_pricing_table()
    
    # Create a tracker
    tracker = create_cost_tracker('gpt-4o-mini')
    
    # Simulate several calls
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
    
    # Print statistics
    tracker.print_statistics()
    
    # Cost estimation example
    print("\n" + "=" * 80)
    print("Cost Estimation Examples:")
    print("=" * 80)
    print(f"gpt-4o-mini (1000 input + 500 output):  ${estimate_cost(1000, 500, 'gpt-4o-mini'):.6f}")
    print(f"gpt-4o      (1000 input + 500 output):  ${estimate_cost(1000, 500, 'gpt-4o'):.6f}")
    print(f"gpt-4-turbo (1000 input + 500 output):  ${estimate_cost(1000, 500, 'gpt-4-turbo'):.6f}")
    print("=" * 80 + "\n")
