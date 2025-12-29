# LLM API 调用指南
## LLM API Usage Guide

---

## 📖 目录

1. [LLM调用架构](#llm调用架构)
2. [API配置方法](#api配置方法)
3. [LLM类使用说明](#llm类使用说明)
4. [支持的API平台](#支持的api平台)
5. [常见问题解答](#常见问题解答)

---

## 🏗️ LLM调用架构

### 核心文件位置

```
NeuralFSM/
├── baseclass/
│   ├── LLM.py              # 🔥 核心LLM调用类
│   ├── FSM_Gen.py          # 使用LLM生成FSM
│   └── MultiAgent.py       # 使用LLM执行智能体
└── config.yaml             # 🔥 API配置文件
```

### 调用层次结构

```
用户代码
    ↓
FSM_Gen.py / MultiAgent.py
    ↓
LLM.py (核心封装类)
    ↓
OpenAI Python SDK
    ↓
[OpenAI API / Azure OpenAI / OpenRouter / 本地VLLM]
```

---

## ⚙️ API配置方法

### 方法1: 使用config.yaml配置（推荐）

**位置**: `NeuralFSM/config.yaml`

```yaml
### For OpenAI
#OPENAI_API_KEY: sk-proj-xxxxx
#OPENAI_API_BASE: https://api.openai-proxy.org/v1

### For OpenRouter (OpenAI compatible) - 当前使用
OPENAI_API_KEY: sk-REDACTED
OPENAI_API_BASE: https://openrouter.ai/api/v1

### For Azure OpenAI
API_VERSION: 
AZURE_OPENAI_API_KEY:
AZURE_OPENAI_API_BASE: 

### For Serper (网络搜索)
SERPER_API_KEY: 1005592c05898cd38eb4b74d0fe9a03e0965d8c3
SERPER_ENDPOINT: https://google.serper.dev/search

### For VLLM (本地部署)
#OPENAI_API_BASE: http://localhost:8000/v1
#OPENAI_API_KEY: 1234
```

**使用说明**:
- 取消注释你要使用的API配置
- 填入你的API密钥
- `LLM.py`会自动从`config.yaml`加载配置

---

### 方法2: 使用环境变量配置

#### 在Linux/Mac上：

```bash
# OpenAI
export OPENAI_API_KEY="sk-xxxxx"
export OPENAI_API_BASE="https://api.openai.com/v1"

# Azure OpenAI
export AZURE_OPENAI_API_KEY="xxxxx"
export AZURE_OPENAI_API_BASE="https://your-endpoint.openai.azure.com/"
export API_VERSION="2024-07-01-preview"

# Serper搜索
export SERPER_API_KEY="xxxxx"
export SERPER_ENDPOINT="https://google.serper.dev/search"
```

#### 在Windows PowerShell上：

```powershell
# OpenAI
$env:OPENAI_API_KEY="sk-xxxxx"
$env:OPENAI_API_BASE="https://api.openai.com/v1"

# Azure OpenAI
$env:AZURE_OPENAI_API_KEY="xxxxx"
$env:AZURE_OPENAI_API_BASE="https://your-endpoint.openai.azure.com/"
$env:API_VERSION="2024-07-01-preview"

# Serper搜索
$env:SERPER_API_KEY="xxxxx"
$env:SERPER_ENDPOINT="https://google.serper.dev/search"
```

#### 使用setup.sh脚本（Linux/Mac）：

**位置**: `NeuralFSM/autodesign/setup.sh`

```bash
#!/bin/bash
export OPENAI_API_BASE=https://openrouter.ai/api/v1
export OPENAI_API_KEY=sk-REDACTED
export API_VERSION=2024-07-01-preview
export SERPER_API_KEY=1005592c05898cd38eb4b74d0fe9a03e0965d8c3
export SERPER_ENDPOINT=https://google.serper.dev/search
```

使用方法：
```bash
source autodesign/setup.sh
```

---

## 💻 LLM类使用说明

### LLM类定义

**位置**: `baseclass/LLM.py`

```python
class LLM():
    def __init__(self, system_prompt="You are a helpful assistant", use_azure=False):
        """
        初始化LLM客户端
        
        Args:
            system_prompt: 系统提示词
            use_azure: 是否使用Azure OpenAI（默认False，使用标准OpenAI API）
        """
        
    def chat(self, message, temperature=0.2, model="gpt-4o"):
        """
        发送聊天消息
        
        Args:
            message: 用户消息
            temperature: 采样温度（0.0-2.0）
            model: 模型名称（支持OpenAI/OpenRouter/Azure的所有模型）
            
        Returns:
            str: LLM的回复
        """
        
    def add_message(self, message):
        """添加用户消息到对话历史"""
        
    def add_tool_message(self, message):
        """添加工具输出消息到对话历史"""
        
    def get_whole_message(self):
        """获取完整对话历史"""
        
    def get_token_cost(self):
        """获取总token消耗"""
```

---

### 使用示例

#### 示例1: 基本使用

```python
from baseclass.LLM import LLM

# 创建LLM实例
llm = LLM(
    system_prompt="You are a helpful math tutor.",
    use_azure=False  # 使用标准OpenAI API（或OpenRouter）
)

# 发送消息
response = llm.chat(
    message="What is 2 + 2?",
    temperature=0.2,
    model="gpt-4o"
)

print(response)
# 输出: "2 + 2 equals 4."

# 查看token消耗
print(f"Total tokens used: {llm.get_token_cost()}")
```

---

#### 示例2: 多轮对话

```python
from baseclass.LLM import LLM

llm = LLM(system_prompt="You are a coding assistant.")

# 第一轮
response1 = llm.chat("Write a Python function to calculate factorial.")
print(response1)

# 第二轮（保留上下文）
response2 = llm.chat("Now add error handling to that function.")
print(response2)

# 查看完整对话历史
history = llm.get_whole_message()
for msg in history:
    print(f"{msg['role']}: {msg['content'][:50]}...")
```

---

#### 示例3: 在FSM生成中使用

**位置**: `baseclass/FSM_Gen.py: 40-41行`

```python
def Generate_Agent_Description(task, tools):
    prompt_template = '''You are the designer of a multi-agent system...'''
    
    # 创建LLM实例
    agent_generator = LLM(prompt_template, use_azure=False)
    
    # 生成智能体描述
    agents = agent_generator.chat(
        message=f"The General Task is {task} and the tools are {tools}."
    )
    
    # 解析JSON
    agent_json = agents.split("```")[-2].replace("json", "")
    agent_dict = json.loads(agent_json)
    
    return agent_dict, agent_generator
```

---

#### 示例4: 在MultiAgent中使用

**位置**: `baseclass/MultiAgent.py: 71行`

```python
class MultiAgentSystem:
    def initialize_agents(self):
        for agent_id, agent in self.agents.items():
            # 为每个智能体创建LLM实例
            self.llms[agent_id] = LLM(
                system_prompt=agent['system_prompt'], 
                use_azure=False
            )
    
    def execute_state(self, state_id, input_data):
        state = self.states[state_id]
        
        # 获取监听智能体列表
        listener_ids = state.get('listener', [])
        
        # 每个智能体使用自己的LLM执行
        for agent_id in listener_ids:
            response = self.llms[agent_id].chat(
                message=f"State: {state['action']}\nInput: {input_data}",
                temperature=0.2
            )
            # 处理响应...
```

---

## 🌐 支持的API平台

### 1. OpenAI 官方API

**配置**:
```yaml
OPENAI_API_KEY: sk-proj-xxxxx
OPENAI_API_BASE: https://api.openai.com/v1
```

**支持模型**:
- `gpt-4o`
- `gpt-4o-mini`
- `gpt-4-turbo`
- `gpt-3.5-turbo`

**使用**:
```python
llm = LLM(use_azure=False)
response = llm.chat("Hello", model="gpt-4o")
```

---

### 2. OpenRouter (推荐，当前使用)

**配置**:
```yaml
OPENAI_API_KEY: sk-or-v1-xxxxx
OPENAI_API_BASE: https://openrouter.ai/api/v1
```

**优势**:
- ✅ 兼容OpenAI API格式
- ✅ 支持多个模型提供商（OpenAI, Anthropic, Google等）
- ✅ 按需付费，无需订阅
- ✅ 自动负载均衡和故障切换

**支持模型**:
- `openai/gpt-4o`
- `anthropic/claude-3-opus`
- `google/gemini-pro-1.5`
- `meta-llama/llama-3-70b`

**使用**:
```python
llm = LLM(use_azure=False)
response = llm.chat("Hello", model="openai/gpt-4o")
```

---

### 3. Azure OpenAI

**配置**:
```yaml
AZURE_OPENAI_API_KEY: xxxxx
AZURE_OPENAI_API_BASE: https://your-endpoint.openai.azure.com/
API_VERSION: 2024-07-01-preview
```

**使用**:
```python
llm = LLM(use_azure=True)  # 🔥 注意：use_azure=True
response = llm.chat("Hello", model="gpt-4o")
```

**注意事项**:
- Azure中的模型名称可能与OpenAI官方不同
- 需要单独部署模型实例
- 企业用户推荐使用

---

### 4. 本地VLLM部署

**配置**:
```yaml
OPENAI_API_BASE: http://localhost:8000/v1
OPENAI_API_KEY: 1234  # 本地部署通常不需要真实密钥
```

**VLLM部署示例**:
```bash
# 安装VLLM
pip install vllm

# 启动VLLM服务器
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --port 8000
```

**使用**:
```python
llm = LLM(use_azure=False)
response = llm.chat("Hello", model="meta-llama/Llama-2-7b-chat-hf")
```

**优势**:
- ✅ 完全本地化，数据隐私
- ✅ 无API调用费用
- ✅ 可自定义模型

---

## 🔧 LLM.py核心实现解析

### API密钥加载机制

**位置**: `baseclass/LLM.py: 7-27行`

```python
def _ensure_api_env_from_config():
    """如果环境变量缺失，从config.yaml加载"""
    try:
        need_key = not os.getenv("OPENAI_API_KEY")
        need_base = not os.getenv("OPENAI_API_BASE")
        
        if not (need_key or need_base):
            return  # 环境变量已设置，无需加载
        
        import yaml
        config_path = os.path.abspath(os.path.join(base_dir, "..", "config.yaml"))
        
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        
        # 从config.yaml加载到环境变量
        if need_key and cfg.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = str(cfg["OPENAI_API_KEY"]).strip()
        
        if need_base and cfg.get("OPENAI_API_BASE"):
            os.environ["OPENAI_API_BASE"] = str(cfg["OPENAI_API_BASE"]).strip()
    
    except Exception:
        pass  # 静默失败，调用者会报错
```

**加载优先级**:
1. 环境变量（`os.getenv()`）
2. `config.yaml`配置文件
3. 代码中硬编码（Azure专用）

---

### OpenAI客户端初始化

**位置**: `baseclass/LLM.py: 29-46行`

```python
class LLM():
    def __init__(self, system_prompt="You are a helpful assistant", use_azure=False):
        _ensure_api_env_from_config()  # 加载配置
        
        self.use_azure = use_azure
        
        if self.use_azure:
            # Azure OpenAI客户端
            self.client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
                api_version=os.getenv("API_VERSION", "2024-07-01-preview")
            )
        else:
            # 标准OpenAI客户端（兼容OpenRouter、VLLM等）
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE")  # 支持自定义端点
            )
        
        self.messages = [{"role": "system", "content": system_prompt}]
        self.system_prompt = system_prompt
        self.token_cost = 0
```

---

### Chat方法实现

**位置**: `baseclass/LLM.py: 51-77行`

```python
def chat(self, message, temperature=0.2, model="gpt-4o"):
    # 添加用户消息到历史
    self.messages.append({"role": "user", "content": message})
    
    # 调用API
    if self.use_azure:
        response = self.client.chat.completions.create(
            model="gpt-4o",  # Azure中的部署名称
            messages=self.messages,
            temperature=temperature
        )
    else:
        response = self.client.chat.completions.create(
            model=model,  # 支持OpenRouter的各种模型
            messages=self.messages,
            temperature=temperature
        )
    
    # 提取回复
    rsp = response.choices[0].message.content
    
    # 添加助手回复到历史
    self.messages.append({"role": "assistant", "content": rsp})
    
    # 累计token消耗
    self.token_cost += response.usage.total_tokens
    
    return rsp
```

---

## ❓ 常见问题解答

### Q1: 如何切换不同的API提供商？

**A**: 修改`config.yaml`中的`OPENAI_API_KEY`和`OPENAI_API_BASE`

```yaml
# OpenAI官方
OPENAI_API_KEY: sk-proj-xxxxx
OPENAI_API_BASE: https://api.openai.com/v1

# OpenRouter（推荐）
OPENAI_API_KEY: sk-or-v1-xxxxx
OPENAI_API_BASE: https://openrouter.ai/api/v1

# 本地VLLM
OPENAI_API_KEY: 1234
OPENAI_API_BASE: http://localhost:8000/v1
```

---

### Q2: 如何使用Azure OpenAI？

**A**: 在创建LLM实例时设置`use_azure=True`

```python
llm = LLM(
    system_prompt="You are a helpful assistant",
    use_azure=True  # 🔥 关键参数
)
```

并确保Azure配置正确：
```yaml
AZURE_OPENAI_API_KEY: xxxxx
AZURE_OPENAI_API_BASE: https://your-endpoint.openai.azure.com/
API_VERSION: 2024-07-01-preview
```

---

### Q3: 如何查看API调用消耗的token数量？

**A**: 使用`get_token_cost()`方法

```python
llm = LLM()
response = llm.chat("Hello, how are you?")
print(f"Tokens used: {llm.get_token_cost()}")
```

---

### Q4: 如何使用不同的模型（如Claude、Gemini）？

**A**: 使用OpenRouter，然后在`model`参数中指定

```yaml
# config.yaml
OPENAI_API_KEY: sk-or-v1-xxxxx
OPENAI_API_BASE: https://openrouter.ai/api/v1
```

```python
# 使用Claude
llm = LLM(use_azure=False)
response = llm.chat("Hello", model="anthropic/claude-3-opus")

# 使用Gemini
response = llm.chat("Hello", model="google/gemini-pro-1.5")

# 使用Llama
response = llm.chat("Hello", model="meta-llama/llama-3-70b")
```

---

### Q5: 如何在项目中统一配置API？

**A**: 推荐使用`config.yaml`方法，所有脚本自动加载

1. 编辑`config.yaml`
2. 填入你的API密钥
3. 所有使用`LLM`类的代码自动生效

---

### Q6: API密钥暴露在config.yaml中安全吗？

**A**: 建议使用环境变量或`.env`文件（不提交到git）

```bash
# 创建.env文件（不提交到git）
echo "OPENAI_API_KEY=sk-xxxxx" > .env
echo "OPENAI_API_BASE=https://api.openai.com/v1" >> .env

# 在代码中加载
from dotenv import load_dotenv
load_dotenv()
```

或者在`.gitignore`中添加：
```
config.yaml
.env
```

---

### Q7: 如何处理API调用错误？

**A**: 使用try-except捕获异常

```python
from baseclass.LLM import LLM

llm = LLM()

try:
    response = llm.chat("Hello")
    print(response)
except Exception as e:
    print(f"API调用失败: {e}")
    # 可以实现重试逻辑
```

---

### Q8: 如何控制LLM的输出长度和风格？

**A**: 通过`system_prompt`和`temperature`参数

```python
# 控制输出风格
llm = LLM(
    system_prompt="""You are a concise assistant. 
    Always provide brief, direct answers without elaboration."""
)

# 控制随机性
response = llm.chat(
    message="Write a story",
    temperature=0.8  # 0.0=确定性, 2.0=随机性
)
```

---

## 📚 相关文档

- [OpenAI API 文档](https://platform.openai.com/docs/api-reference)
- [OpenRouter 文档](https://openrouter.ai/docs)
- [Azure OpenAI 文档](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [VLLM 文档](https://docs.vllm.ai/)

---

## 🔗 项目中的使用位置

| 文件 | 用途 | LLM调用方式 |
|-----|------|------------|
| `baseclass/FSM_Gen.py` | 生成智能体描述和FSM | `LLM(prompt_template).chat(message)` |
| `baseclass/MultiAgent.py` | 执行多智能体系统 | `LLM(system_prompt).chat(message)` |
| `autodesign/building_pipeline.py` | 自动设计流程 | `LLM(prompt).chat(task)` |
| `baselines/baselines.py` | 基线方法对比 | `LLM().chat(question)` |

---

## 📞 技术支持

如有问题，请参考：
1. 本文档的常见问题解答部分
2. `baseclass/LLM.py`的源代码和注释
3. OpenAI/OpenRouter的官方文档

---

**文档版本**: v1.0  
**最后更新**: 2025-01-XX  
**适用项目**: NeuralFSM

