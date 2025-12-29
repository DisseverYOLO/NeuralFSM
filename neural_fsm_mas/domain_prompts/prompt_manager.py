"""
Domain Prompt Manager for Multi-Agent System
领域提示管理器

适配自原始PromptSetRegistry实现，专门为MetaAgent项目优化
支持不同领域的提示管理和角色定义
"""

from typing import Type, Dict, Any, List, Tuple, Optional
import sys
from pathlib import Path

# 添加MetaAgent路径
sys.path.append(str(Path(__file__).parent.parent.parent))


class DomainPromptSet:
    """
    领域提示集合基类
    
    定义了特定领域的提示模板、角色描述和连接关系
    """
    
    def __init__(self, domain_name: str):
        self.domain_name = domain_name
        self.role_descriptions = {}
        self.role_connections = []
        self.answer_prompts = {}
        self.constraints = {}
    
    def get_role_description(self, role_name: str) -> str:
        """获取角色描述"""
        return self.role_descriptions.get(role_name, f"You are a {role_name} agent.")
    
    def get_role_connections(self) -> List[Tuple[str, str]]:
        """获取角色连接关系"""
        return self.role_connections
    
    def get_answer_prompt(self, question: str, role: str) -> str:
        """获取回答提示"""
        base_prompt = self.answer_prompts.get(role, "Please answer the following question: {question}")
        return base_prompt.format(question=question)
    
    def get_constraint(self, role: str) -> str:
        """获取角色约束"""
        return self.constraints.get(role, "Please provide a helpful and accurate response.")
    
    def get_available_roles(self) -> List[str]:
        """获取可用角色列表"""
        return list(self.role_descriptions.keys())


class MMLUDomainPromptSet(DomainPromptSet):
    """
    MMLU领域提示集合
    
    专门为MMLU数据集优化的提示集合
    """
    
    def __init__(self):
        super().__init__("mmlu")
        
        # 定义MMLU相关的角色
        self.role_descriptions = {
            "Knowledge Expert": """
You are a knowledgeable expert in question answering across multiple domains.
Your role is to provide comprehensive analysis and identify key concepts needed to solve problems.
You excel at breaking down complex questions and providing structured reasoning.
""",
            
            "Subject Specialist": """
You are a subject matter specialist with deep expertise in specific academic domains.
You provide detailed, accurate information within your area of expertise.
You can explain complex concepts clearly and relate them to the given questions.
""",
            
            "Critical Analyzer": """
You are an excellent critical analyzer and reviewer.
Your role is to examine other agents' responses, identify potential issues, and provide constructive feedback.
You look for logical inconsistencies, factual errors, and gaps in reasoning.
""",
            
            "Mathematician": """
You are a mathematician skilled in mathematical reasoning, calculations, and problem-solving.
You excel at quantitative analysis, logical deduction, and mathematical modeling.
You can handle complex mathematical concepts and computations accurately.
""",
            
            "Scientist": """
You are a scientist with broad knowledge across natural sciences.
You understand scientific principles, experimental methods, and empirical reasoning.
You can apply scientific thinking to analyze problems systematically.
""",
            
            "Humanities Scholar": """
You are a humanities scholar with expertise in literature, history, philosophy, and cultural studies.
You provide insights into human culture, historical contexts, and philosophical perspectives.
You excel at interpretive analysis and contextual understanding.
""",
            
            "Social Scientist": """
You are a social scientist specializing in psychology, sociology, economics, and political science.
You understand human behavior, social systems, and economic principles.
You can analyze social phenomena and human decision-making processes.
"""
        }
        
        # 定义角色连接关系
        self.role_connections = [
            ("Knowledge Expert", "Subject Specialist"),
            ("Knowledge Expert", "Critical Analyzer"),
            ("Subject Specialist", "Mathematician"),
            ("Subject Specialist", "Scientist"),
            ("Subject Specialist", "Humanities Scholar"),
            ("Subject Specialist", "Social Scientist"),
            ("Critical Analyzer", "Knowledge Expert"),
            ("Mathematician", "Critical Analyzer"),
            ("Scientist", "Critical Analyzer"),
            ("Humanities Scholar", "Critical Analyzer"),
            ("Social Scientist", "Critical Analyzer")
        ]
        
        # 定义回答提示模板
        self.answer_prompts = {
            "Knowledge Expert": """
Question: {question}

As a knowledge expert, please:
1. Analyze the question and identify the key concepts involved
2. Determine what domain knowledge is required
3. Provide a structured approach to solving this question
4. Give your best answer with clear reasoning
""",
            
            "Subject Specialist": """
Question: {question}

As a subject specialist, please:
1. Apply your domain expertise to analyze this question
2. Provide detailed explanations of relevant concepts
3. Give your expert opinion with supporting evidence
4. Explain any technical terms or specialized knowledge
""",
            
            "Critical Analyzer": """
Question: {question}

As a critical analyzer, please:
1. Examine the question for potential ambiguities or complexities
2. Consider multiple perspectives and possible interpretations
3. Identify what additional information might be needed
4. Provide a balanced analysis with your conclusion
""",
            
            "Mathematician": """
Question: {question}

As a mathematician, please:
1. Identify any mathematical concepts or calculations involved
2. Apply logical reasoning and mathematical principles
3. Show your work step-by-step if calculations are needed
4. Provide a precise and accurate answer
""",
            
            "Scientist": """
Question: {question}

As a scientist, please:
1. Apply scientific principles and empirical reasoning
2. Consider evidence-based approaches to the problem
3. Use systematic analysis and logical deduction
4. Provide a scientifically sound answer
""",
            
            "Humanities Scholar": """
Question: {question}

As a humanities scholar, please:
1. Consider historical, cultural, and philosophical contexts
2. Apply interpretive analysis and critical thinking
3. Draw on knowledge of human culture and expression
4. Provide a thoughtful, well-reasoned response
""",
            
            "Social Scientist": """
Question: {question}

As a social scientist, please:
1. Consider human behavior and social dynamics
2. Apply knowledge of psychology, sociology, or economics as relevant
3. Analyze the question from a social science perspective
4. Provide insights based on social science principles
"""
        }
        
        # 定义角色约束
        self.constraints = {
            "Knowledge Expert": "Provide comprehensive and accurate analysis. Focus on identifying key concepts and structuring the problem-solving approach.",
            "Subject Specialist": "Leverage your specialized knowledge to provide expert-level insights. Ensure accuracy within your domain of expertise.",
            "Critical Analyzer": "Maintain objectivity and provide constructive criticism. Focus on improving the overall quality of reasoning.",
            "Mathematician": "Ensure mathematical accuracy and logical consistency. Show clear reasoning for any calculations or logical deductions.",
            "Scientist": "Apply scientific rigor and evidence-based reasoning. Maintain objectivity and systematic analysis.",
            "Humanities Scholar": "Provide thoughtful interpretation and contextual understanding. Draw on cultural and historical knowledge appropriately.",
            "Social Scientist": "Apply social science principles accurately. Consider human factors and social dynamics in your analysis."
        }


class GSM8KDomainPromptSet(DomainPromptSet):
    """
    GSM8K领域提示集合
    
    专门为GSM8K数学问题优化的提示集合
    """
    
    def __init__(self):
        super().__init__("gsm8k")
        
        self.role_descriptions = {
            "Math Problem Solver": """
You are an expert mathematical problem solver specializing in grade school math word problems.
You excel at breaking down complex word problems into step-by-step solutions.
You always show your work clearly and verify your calculations.
""",
            
            "Problem Analyzer": """
You are a problem analysis expert who helps break down mathematical word problems.
You identify key information, unknown variables, and the mathematical operations needed.
You help structure the problem-solving approach systematically.
""",
            
            "Calculation Verifier": """
You are a calculation verification specialist.
Your role is to double-check mathematical calculations and identify any errors.
You ensure accuracy and consistency in mathematical reasoning.
""",
            
            "Solution Critic": """
You are a solution critic who reviews mathematical solutions for correctness and clarity.
You identify potential errors, suggest improvements, and ensure the solution addresses the original problem.
"""
        }
        
        self.role_connections = [
            ("Problem Analyzer", "Math Problem Solver"),
            ("Math Problem Solver", "Calculation Verifier"),
            ("Calculation Verifier", "Solution Critic"),
            ("Solution Critic", "Math Problem Solver")
        ]


class HumanEvalDomainPromptSet(DomainPromptSet):
    """
    HumanEval领域提示集合
    
    专门为HumanEval代码生成任务优化的提示集合
    """
    
    def __init__(self):
        super().__init__("humaneval")
        
        self.role_descriptions = {
            "Code Designer": """
You are an expert software architect and code designer.
You excel at understanding programming requirements and designing clean, efficient solutions.
You break down complex coding tasks into clear implementation steps.
""",
            
            "Code Writer": """
You are a skilled programmer proficient in multiple programming languages.
You write clean, efficient, and well-documented code that follows best practices.
You ensure code correctness, readability, and maintainability.
""",
            
            "Code Reviewer": """
You are a meticulous code reviewer with expertise in identifying bugs and code quality issues.
You check for logical errors, edge cases, performance issues, and adherence to coding standards.
You provide constructive feedback to improve code quality.
""",
            
            "Test Engineer": """
You are a test engineer specializing in writing comprehensive test cases.
You identify edge cases, corner cases, and potential failure modes.
You ensure code correctness through thorough testing and validation.
""",
            
            "Algorithm Expert": """
You are an algorithm expert with deep knowledge of data structures and algorithms.
You analyze time and space complexity and suggest optimal algorithmic approaches.
You help optimize code for performance and efficiency.
"""
        }
        
        self.role_connections = [
            ("Code Designer", "Code Writer"),
            ("Code Designer", "Algorithm Expert"),
            ("Code Writer", "Code Reviewer"),
            ("Code Writer", "Test Engineer"),
            ("Algorithm Expert", "Code Writer"),
            ("Code Reviewer", "Code Designer"),
            ("Test Engineer", "Code Reviewer")
        ]
        
        self.answer_prompts = {
            "Code Designer": """
Programming Task: {question}

As a code designer, please:
1. Analyze the programming requirements and constraints
2. Design a high-level solution approach
3. Identify key data structures and algorithms needed
4. Outline the implementation steps
""",
            
            "Code Writer": """
Programming Task: {question}

As a code writer, please:
1. Implement the required function based on the specification
2. Write clean, readable, and well-commented code
3. Handle edge cases and error conditions
4. Follow coding best practices and conventions
""",
            
            "Code Reviewer": """
Programming Task: {question}

As a code reviewer, please:
1. Review the code for correctness and potential bugs
2. Check for edge cases and error handling
3. Verify that the code meets the requirements
4. Suggest improvements for code quality and efficiency
""",
            
            "Test Engineer": """
Programming Task: {question}

As a test engineer, please:
1. Design comprehensive test cases to verify correctness
2. Identify edge cases and corner cases to test
3. Consider potential failure modes and error conditions
4. Ensure the code passes all required tests
""",
            
            "Algorithm Expert": """
Programming Task: {question}

As an algorithm expert, please:
1. Analyze the algorithmic requirements of the task
2. Suggest optimal data structures and algorithms
3. Evaluate time and space complexity
4. Recommend performance optimizations if applicable
"""
        }
        
        self.constraints = {
            "Code Designer": "Focus on creating a clear, logical design that addresses all requirements.",
            "Code Writer": "Write correct, efficient code that follows best practices and handles edge cases.",
            "Code Reviewer": "Provide thorough, constructive review focusing on correctness and quality.",
            "Test Engineer": "Design comprehensive tests that thoroughly validate code correctness.",
            "Algorithm Expert": "Provide algorithmic insights to optimize performance and efficiency."
        }


class HotpotQADomainPromptSet(DomainPromptSet):
    """
    HotpotQA领域提示集合
    
    专门为HotpotQA多跳问答任务优化的提示集合
    """
    
    def __init__(self):
        super().__init__("hotpotqa")
        
        self.role_descriptions = {
            "Question Analyzer": """
You are a question analysis expert specializing in multi-hop question answering.
You identify question types (bridge or comparison), extract key entities, and determine information needs.
You excel at understanding complex questions that require reasoning across multiple documents.
""",
            
            "Document Retriever": """
You are a document retrieval specialist.
Your role is to identify and select relevant documents from the provided context.
You rank documents by relevance to the question and filter out irrelevant information.
""",
            
            "Fact Extractor": """
You are a fact extraction expert.
You extract key facts and information from documents, identifying potential supporting facts.
You structure extracted information clearly for downstream reasoning.
""",
            
            "Multi-hop Reasoner": """
You are a multi-hop reasoning specialist.
You connect facts across multiple documents to build reasoning chains.
You handle both bridge reasoning (A→B→C) and comparison reasoning (A vs B).
You identify the logical connections between different pieces of information.
""",
            
            "Answer Synthesizer": """
You are an answer synthesis expert.
You combine information from multiple sources to formulate comprehensive answers.
You ensure answers are complete, accurate, and well-supported by evidence.
""",
            
            "Consistency Verifier": """
You are a consistency verification specialist.
You validate that answers are consistent with supporting facts.
You check for contradictions and ensure answer quality.
"""
        }
        
        self.role_connections = [
            ("Question Analyzer", "Document Retriever"),
            ("Document Retriever", "Fact Extractor"),
            ("Fact Extractor", "Multi-hop Reasoner"),
            ("Multi-hop Reasoner", "Answer Synthesizer"),
            ("Answer Synthesizer", "Consistency Verifier"),
            ("Consistency Verifier", "Multi-hop Reasoner")
        ]


class ALFWorldDomainPromptSet(DomainPromptSet):
    """
    ALFWorld领域提示集合
    
    专门为ALFWorld具身智能体交互任务优化的提示集合
    """
    
    def __init__(self):
        super().__init__("alfworld")
        
        self.role_descriptions = {
            "Task Parser": """
You are a task parsing specialist for embodied AI agents.
You understand goal descriptions and decompose them into actionable subgoals.
You identify task types (pick_and_place, pick_clean_then_place, etc.) and requirements.
""",
            
            "Environment Explorer": """
You are an environment exploration expert.
You explore and understand the current state of the simulated environment.
You identify relevant objects, locations, and their relationships.
""",
            
            "Action Planner": """
You are an action planning specialist.
You plan sequences of actions to achieve subgoals.
You consider current environment state, goal requirements, and action constraints.
""",
            
            "Action Executor": """
You are an action execution expert.
You execute planned actions and parse environment feedback.
You understand action vocabulary (go, take, put, open, close, etc.) and their effects.
""",
            
            "State Monitor": """
You are a state monitoring specialist.
You track environment state changes and subgoal progress.
You verify when subgoals are achieved and determine next steps.
""",
            
            "Recovery Agent": """
You are a recovery and error handling specialist.
You detect action failures and replan when needed.
You handle unexpected situations and adapt plans dynamically.
""",
            
            "Goal Verifier": """
You are a goal verification expert.
You confirm when all subgoals are completed and the main goal is achieved.
You validate task completion and provide final confirmation.
"""
        }
        
        self.role_connections = [
            ("Task Parser", "Environment Explorer"),
            ("Environment Explorer", "Action Planner"),
            ("Action Planner", "Action Executor"),
            ("Action Executor", "State Monitor"),
            ("State Monitor", "Action Planner"),
            ("State Monitor", "Goal Verifier"),
            ("Recovery Agent", "Action Planner"),
            ("Goal Verifier", "Recovery Agent")
        ]


class MATHDomainPromptSet(DomainPromptSet):
    """
    MATH领域提示集合
    
    专门为MATH高级数学竞赛题优化的提示集合
    """
    
    def __init__(self):
        super().__init__("math")
        
        self.role_descriptions = {
            "Problem Analyzer": """
You are a mathematical problem analysis expert.
You identify problem types, subjects (Algebra, Geometry, etc.), and key mathematical concepts.
You extract given information and determine what needs to be found.
""",
            
            "Math Concept Expert": """
You are a mathematical concept specialist with deep knowledge across all mathematical domains.
You recall relevant theorems, formulas, definitions, and techniques.
You provide mathematical foundations needed for problem-solving.
""",
            
            "Strategy Designer": """
You are a mathematical strategy design expert.
You choose appropriate solution approaches (direct proof, construction, algebraic manipulation, etc.).
You outline step-by-step solution plans.
""",
            
            "Step-by-step Solver": """
You are a step-by-step mathematical problem solver.
You execute solution steps with detailed reasoning and explanations.
You perform calculations and derivations accurately.
""",
            
            "Symbolic Calculator": """
You are a symbolic computation specialist.
You perform algebraic manipulations, symbolic calculations, and mathematical transformations.
You handle complex expressions and equations accurately.
""",
            
            "Solution Verifier": """
You are a mathematical solution verification expert.
You check answer correctness and solution validity.
You verify logical steps and mathematical reasoning.
""",
            
            "Alternative Approach Agent": """
You are an alternative solution method specialist.
You explore different approaches to solve problems.
You compare methods and identify the most elegant solutions.
""",
            
            "LaTeX Formatter": """
You are a LaTeX formatting specialist.
You format mathematical answers in proper LaTeX notation.
You ensure correct mathematical typesetting and notation.
"""
        }
        
        self.role_connections = [
            ("Problem Analyzer", "Math Concept Expert"),
            ("Math Concept Expert", "Strategy Designer"),
            ("Strategy Designer", "Step-by-step Solver"),
            ("Step-by-step Solver", "Symbolic Calculator"),
            ("Symbolic Calculator", "Solution Verifier"),
            ("Solution Verifier", "Alternative Approach Agent"),
            ("Alternative Approach Agent", "LaTeX Formatter"),
            ("LaTeX Formatter", "Solution Verifier")
        ]


class DomainPromptRegistry:
    """
    领域提示注册表
    
    管理不同领域的提示集合
    """
    _prompt_sets: Dict[str, DomainPromptSet] = {}
    
    @classmethod
    def register_domain(cls, domain_name: str, prompt_set: DomainPromptSet):
        """注册领域提示集合"""
        cls._prompt_sets[domain_name] = prompt_set
    
    @classmethod
    def get_prompt_set(cls, domain_name: str) -> Optional[DomainPromptSet]:
        """获取领域提示集合"""
        return cls._prompt_sets.get(domain_name)
    
    @classmethod
    def get_available_domains(cls) -> List[str]:
        """获取可用领域列表"""
        return list(cls._prompt_sets.keys())


class DomainPromptManager:
    """
    领域提示管理器
    
    提供统一的领域提示管理接口
    """
    
    @staticmethod
    def get_manager(domain_name: str) -> DomainPromptSet:
        """获取领域提示管理器"""
        prompt_set = DomainPromptRegistry.get_prompt_set(domain_name)
        
        if prompt_set is None:
            # 如果没有找到特定领域的提示集合，创建默认的
            prompt_set = DomainPromptSet(domain_name)
            prompt_set.role_descriptions = {
                "General Agent": f"You are a general-purpose agent working on {domain_name} tasks.",
                "Analyzer": f"You are an analyzer specializing in {domain_name} domain.",
                "Critic": f"You are a critic providing feedback on {domain_name} solutions."
            }
            prompt_set.role_connections = [
                ("General Agent", "Analyzer"),
                ("Analyzer", "Critic"),
                ("Critic", "General Agent")
            ]
            DomainPromptRegistry.register_domain(domain_name, prompt_set)
        
        return prompt_set
    
    @staticmethod
    def register_custom_domain(domain_name: str, 
                             role_descriptions: Dict[str, str],
                             role_connections: List[Tuple[str, str]] = None,
                             answer_prompts: Dict[str, str] = None,
                             constraints: Dict[str, str] = None):
        """注册自定义领域"""
        prompt_set = DomainPromptSet(domain_name)
        prompt_set.role_descriptions = role_descriptions
        prompt_set.role_connections = role_connections or []
        prompt_set.answer_prompts = answer_prompts or {}
        prompt_set.constraints = constraints or {}
        
        DomainPromptRegistry.register_domain(domain_name, prompt_set)


# =====================================================================
# FSM模式提示集（新增）
# =====================================================================

class FSMPromptMixin:
    """
    FSM模式提示混入类
    
    为智能体提供FSM状态转移指导
    """
    
    @staticmethod
    def get_fsm_instruction(state_id: int, 
                           state_name: str,
                           is_final: bool = False,
                           possible_next_states: List[Dict[str, Any]] = None) -> str:
        """
        获取FSM状态转移指导
        
        Args:
            state_id: 当前状态ID
            state_name: 当前状态名称
            is_final: 是否为最终状态
            possible_next_states: 可能的下一个状态列表
        
        Returns:
            FSM指导文本
        """
        if is_final:
            return f"""
You are currently in the FINAL state (State {state_id}: {state_name}).
After completing your analysis, provide your final answer using the following format:

<|submit|> YOUR_FINAL_ANSWER

Replace YOUR_FINAL_ANSWER with your actual answer.
"""
        
        instruction = f"""
You are currently in State {state_id}: {state_name}.
After completing your analysis, you must specify the next state to transition to.
"""
        
        if possible_next_states:
            instruction += "\n\nPossible next states:\n"
            for next_state in possible_next_states:
                instruction += f"  - State {next_state['id']}: {next_state['name']} - {next_state.get('description', '')}\n"
        
        instruction += """
To transition to the next state, use the following format at the end of your response:

<STATE_TRANS>: X

Replace X with the state ID you want to transition to.
For example: <STATE_TRANS>: 2
"""
        
        return instruction
    
    @staticmethod
    def get_listener_context(predecessor_messages: List[str]) -> str:
        """
        获取前序智能体消息的上下文
        
        Args:
            predecessor_messages: 前序智能体的消息列表
        
        Returns:
            上下文文本
        """
        if not predecessor_messages:
            return ""
        
        context = "\n--- Messages from Previous States ---\n\n"
        for i, msg in enumerate(predecessor_messages, 1):
            context += f"Message {i}:\n{msg}\n\n"
        context += "--- End of Previous Messages ---\n\n"
        
        return context


class FSMGSMPromptSet(GSM8KDomainPromptSet):
    """
    FSM模式的GSM8K提示集
    
    扩展GSM8K提示集以支持FSM状态转移
    """
    
    def __init__(self):
        super().__init__()
        
        # 定义FSM状态
        self.fsm_states = {
            0: {
                'name': 'Problem Understanding',
                'description': 'Understand the problem and identify key information',
                'agent_role': 'Problem Analyzer',
                'is_initial': True
            },
            1: {
                'name': 'Solution Planning',
                'description': 'Design the solution approach and break down steps',
                'agent_role': 'Reasoning Expert'
            },
            2: {
                'name': 'Calculation',
                'description': 'Perform mathematical calculations',
                'agent_role': 'Math Solver'
            },
            3: {
                'name': 'Verification',
                'description': 'Verify the answer and provide final result',
                'agent_role': 'Solution Verifier',
                'is_final': True
            }
        }
    
    def get_answer_prompt(self, question: str, role: str, 
                         state_id: Optional[int] = None,
                         predecessor_messages: Optional[List[str]] = None) -> str:
        """获取FSM模式的回答提示"""
        base_prompt = super().get_answer_prompt(question, role)
        
        # 添加前序消息上下文
        if predecessor_messages:
            context = FSMPromptMixin.get_listener_context(predecessor_messages)
            base_prompt = context + base_prompt
        
        # 添加FSM状态转移指导
        if state_id is not None and state_id in self.fsm_states:
            state_info = self.fsm_states[state_id]
            is_final = state_info.get('is_final', False)
            
            # 获取可能的下一个状态
            possible_next = []
            for sid, sinfo in self.fsm_states.items():
                if sid > state_id:  # 只允许前向转移
                    possible_next.append({
                        'id': sid,
                        'name': sinfo['name'],
                        'description': sinfo.get('description', '')
                    })
            
            fsm_instruction = FSMPromptMixin.get_fsm_instruction(
                state_id, state_info['name'], is_final, possible_next
            )
            base_prompt += "\n\n" + fsm_instruction
        
        return base_prompt


class FSMMMLUPromptSet(MMLUDomainPromptSet):
    """FSM模式的MMLU提示集"""
    
    def __init__(self):
        super().__init__()
        
        self.fsm_states = {
            0: {
                'name': 'Question Analysis',
                'description': 'Analyze the question and identify the domain',
                'agent_role': 'Knowledge Expert',
                'is_initial': True
            },
            1: {
                'name': 'Knowledge Retrieval',
                'description': 'Retrieve relevant knowledge and concepts',
                'agent_role': 'Subject Specialist'
            },
            2: {
                'name': 'Option Evaluation',
                'description': 'Evaluate each option systematically',
                'agent_role': 'Critical Analyzer'
            },
            3: {
                'name': 'Final Decision',
                'description': 'Make the final choice and provide justification',
                'agent_role': 'Final Decision Maker',
                'is_final': True
            }
        }
    
    def get_answer_prompt(self, question: str, role: str,
                         state_id: Optional[int] = None,
                         predecessor_messages: Optional[List[str]] = None) -> str:
        """获取FSM模式的回答提示"""
        base_prompt = super().get_answer_prompt(question, role)
        
        if predecessor_messages:
            context = FSMPromptMixin.get_listener_context(predecessor_messages)
            base_prompt = context + base_prompt
        
        if state_id is not None and state_id in self.fsm_states:
            state_info = self.fsm_states[state_id]
            is_final = state_info.get('is_final', False)
            
            possible_next = [
                {'id': sid, 'name': sinfo['name'], 'description': sinfo.get('description', '')}
                for sid, sinfo in self.fsm_states.items() if sid > state_id
            ]
            
            fsm_instruction = FSMPromptMixin.get_fsm_instruction(
                state_id, state_info['name'], is_final, possible_next
            )
            base_prompt += "\n\n" + fsm_instruction
        
        return base_prompt


class FSMHumanEvalPromptSet(HumanEvalDomainPromptSet):
    """FSM模式的HumanEval提示集"""
    
    def __init__(self):
        super().__init__()
        
        self.fsm_states = {
            0: {
                'name': 'Requirement Analysis',
                'description': 'Understand the function requirements and constraints',
                'agent_role': 'Code Analyzer',
                'is_initial': True
            },
            1: {
                'name': 'Algorithm Design',
                'description': 'Design the algorithm and data structures',
                'agent_role': 'Algorithm Designer'
            },
            2: {
                'name': 'Code Implementation',
                'description': 'Write the actual code implementation',
                'agent_role': 'Code Writer'
            },
            3: {
                'name': 'Code Review',
                'description': 'Review code for correctness and edge cases',
                'agent_role': 'Code Reviewer',
                'is_final': True
            }
        }
    
    def get_answer_prompt(self, question: str, role: str,
                         state_id: Optional[int] = None,
                         predecessor_messages: Optional[List[str]] = None) -> str:
        """获取FSM模式的回答提示"""
        base_prompt = super().get_answer_prompt(question, role)
        
        if predecessor_messages:
            context = FSMPromptMixin.get_listener_context(predecessor_messages)
            base_prompt = context + base_prompt
        
        if state_id is not None and state_id in self.fsm_states:
            state_info = self.fsm_states[state_id]
            is_final = state_info.get('is_final', False)
            
            possible_next = [
                {'id': sid, 'name': sinfo['name'], 'description': sinfo.get('description', '')}
                for sid, sinfo in self.fsm_states.items() if sid > state_id
            ]
            
            fsm_instruction = FSMPromptMixin.get_fsm_instruction(
                state_id, state_info['name'], is_final, possible_next
            )
            base_prompt += "\n\n" + fsm_instruction
        
        return base_prompt


# 初始化默认领域
DomainPromptRegistry.register_domain("mmlu", MMLUDomainPromptSet())
DomainPromptRegistry.register_domain("gsm8k", GSM8KDomainPromptSet())
DomainPromptRegistry.register_domain("humaneval", HumanEvalDomainPromptSet())
DomainPromptRegistry.register_domain("hotpotqa", HotpotQADomainPromptSet())  # ✨ NEW
DomainPromptRegistry.register_domain("alfworld", ALFWorldDomainPromptSet())  # ✨ NEW
DomainPromptRegistry.register_domain("math", MATHDomainPromptSet())          # ✨ NEW

# 注册FSM模式提示集
DomainPromptRegistry.register_domain("fsm_mmlu", FSMMMLUPromptSet())
DomainPromptRegistry.register_domain("fsm_gsm8k", FSMGSMPromptSet())
DomainPromptRegistry.register_domain("fsm_humaneval", FSMHumanEvalPromptSet())

# 导出主要类
__all__ = [
    'DomainPromptSet', 
    'MMLUDomainPromptSet', 
    'GSM8KDomainPromptSet',
    'HumanEvalDomainPromptSet',
    'HotpotQADomainPromptSet',  # ✨ NEW
    'ALFWorldDomainPromptSet',  # ✨ NEW
    'MATHDomainPromptSet',      # ✨ NEW
    'FSMPromptMixin',
    'FSMGSMPromptSet',
    'FSMMMLUPromptSet',
    'FSMHumanEvalPromptSet',
    'DomainPromptRegistry', 
    'DomainPromptManager'
]
