import os
from openai import OpenAI, AzureOpenAI
import json
# export AZURE_OPENAI_API_KEY="d56bd868ff56401595c6e74357c02f04"
# export AZURE_OPENAI_API_BASE="https://yaolun-west.openai.azure.com/"

def _ensure_api_env_from_config():
    """If OPENAI envs are missing, try loading them from project root config.yaml."""
    try:
        need_key = not os.getenv("OPENAI_API_KEY")
        need_base = not os.getenv("OPENAI_API_BASE")
        if not (need_key or need_base):
            return
        import yaml  # PyYAML is present per requirements/pip list
        base_dir = os.path.dirname(__file__)
        config_path = os.path.abspath(os.path.join(base_dir, "..", "config.yaml"))
        if not os.path.isfile(config_path):
            return
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if need_key and cfg.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = str(cfg["OPENAI_API_KEY"]).strip()
        if need_base and cfg.get("OPENAI_API_BASE"):
            os.environ["OPENAI_API_BASE"] = str(cfg["OPENAI_API_BASE"]).strip()
    except Exception:
        # Fail silently; caller will error if still unset
        pass

class LLM():
    def __init__(self, system_prompt="You are a helpful assistant", use_azure=False):
        _ensure_api_env_from_config()
        self.use_azure = use_azure
        # 允许通过环境变量统一控制默认模型（例如由实验脚本设置）
        self.default_model = os.getenv("NEURALFSM_LLM_MODEL", "gpt-5-nano")
        if self.use_azure:
            self.client = AzureOpenAI(
                api_key="d56bd868ff56401595c6e74357c02f04",
                #os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint="https://yaolun-west.openai.azure.com/",
                #os.getenv("AZURE_OPENAI_API_BASE"),
                api_version="2024-07-01-preview"
                #os.getenv("API_VERSION")
            )
        else:
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE")  # 启用 OpenRouter API 支持
            )
        self.messages = [{"role": "system", "content": system_prompt}]
        self.system_prompt = system_prompt
        self.token_cost = 0
        # 分别追踪 input 和 output tokens
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def chat(self, message, temperature=0.1, model=None):
        # 如果未显式指定模型，则使用实例的默认模型
        model = model or self.default_model
        self.messages.append({"role": "user", "content": message})
        
        # ✨ 某些模型不支持 temperature 参数，需要特殊处理
        models_no_temp = ['gpt-5-nano', 'o1-mini', 'o1-preview', 'o1']
        skip_temp = any(m in model.lower() for m in models_no_temp)
        
        if self.use_azure:
            try:
                if skip_temp:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=self.messages
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=self.messages,
                        temperature=temperature
                    )
            except Exception as e:
                print(self.messages[-1])
                print(e)
                response = self.client.chat.completions.create(
                    model=model,
                    messages=self.messages
                )
        else:
            if skip_temp:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=self.messages
                )
            else:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=self.messages,
                    temperature=temperature
                )
        rsp = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": rsp})

        # 统计本次调用的token使用情况
        input_tokens = 0
        output_tokens = 0
        usage = getattr(response, "usage", None)
        # ✨ 调试：打印usage对象类型
        # print(f"    [DEBUG] response.usage type: {type(usage)}, value: {usage}")
        if usage is not None:
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            # OpenAI SDK 有时使用 "completion_tokens" 或 "completion_tokens"
            output_tokens = getattr(usage, "completion_tokens", 0) or getattr(usage, "completion_tokens", 0) or 0
            self.token_cost += (input_tokens + output_tokens)
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens

            # 将实际token用量写入全局LLM成本追踪器（如果已配置）
            try:
                from neural_fsm_mas.utils.llm_cost_tracker import get_global_cost_tracker
                tracker = get_global_cost_tracker()
                if tracker is not None and (input_tokens or output_tokens):
                    cost = tracker.track_call(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        model_name=model,
                    )
                    # ✨ 调试信息：确认成本被正确追踪
                    # print(f"    💵 [CostTracker] {model}: {input_tokens}+{output_tokens} tokens = ${cost:.6f}")
            except Exception as e:
                # 成本追踪失败不应影响主逻辑
                # print(f"    ⚠️  成本追踪失败: {e}")
                pass

        return rsp

    def add_message(self, message):
        self.messages.append({"role": "user", "content": message})

    def add_tool_message(self, message):
        self.messages.append({"role": "user", "content": "[INFO] This is a tool message:\n" + message})

    def get_whole_message(self):
        return self.messages

    def get_token_cost(self):
        return self.token_cost
    
    def get_token_usage(self):
        """获取详细的 token 使用统计"""
        return {
            'total_tokens': self.token_cost,
            'input_tokens': self.total_input_tokens,
            'output_tokens': self.total_output_tokens
        }
    
    def calculate_cost_usd(self, model_name=None):
        """根据实际使用的 tokens 和模型计算 USD 成本"""
        if model_name is None:
            model_name = self.default_model
        
        try:
            from neural_fsm_mas.utils.llm_cost_tracker import LLMPricingRegistry
            pricing = LLMPricingRegistry.get_pricing(model_name)
            return pricing.calculate_cost(self.total_input_tokens, self.total_output_tokens)
        except Exception:
            # 如果无法获取定价，使用 gpt-5-nano 作为默认估算
            input_cost = (self.total_input_tokens / 1000.0) * 0.00015
            output_cost = (self.total_output_tokens / 1000.0) * 0.0006
            return input_cost + output_cost

    def recount_token(self):
        self.token_cost = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0