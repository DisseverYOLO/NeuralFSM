

import json
import re
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from baseclass.LLM import LLM


class EnhancedFSMGenerator:
    """
    Enhanced FSM Generator - three-step method
    
    Step 1: generate_state_plan() - LLM generates the state plan
    Step 2: generate_agents_from_states() - LLM generates one agent for each state
    Step 3: generate_fsm_states() - directly combine Step 1 + Step 2 to build the full FSM without calling the LLM
    """
    
    def __init__(self, use_azure: bool = False):
        self.use_azure = use_azure
        self.dataset_templates = {
            'gsm8k': self._get_gsm8k_template(),
            'mmlu': self._get_mmlu_template(),
            'gpqa': self._get_gpqa_template(),
            'gaia': self._get_gaia_template(),
            'humaneval': self._get_humaneval_template(),
            'hotpotqa': self._get_hotpotqa_template(),
            'alfworld': self._get_alfworld_template(),
            'math': self._get_math_template(),
            'mbpp': self._get_mbpp_template(),
            'general': self._get_general_template()
        }
    
    def _get_gsm8k_template(self) -> Dict[str, str]:
        return {
            'task_description': """Grade School Math Word Problem Solving Task
Characteristics:
- Problems require multi-step reasoning
- Requires precise numerical calculations
- Answer is a specific numerical value
- Need to verify calculation logic

Requirements:
1. States should decompose problem-solving process
2. Each state needs clear completion criteria
3. Agents need mathematical reasoning and calculation capabilities
4. Support iterative improvement and error correction""",
            
            'state_granularity_guide': """State Decomposition Guide:
1. Initial Understanding Phase (1-2 states):
   - Problem text parsing
   - Key information extraction
   
2. Solution Planning Phase (1-2 states):
   - Solution strategy formulation
   - Calculation step planning
   
3. Calculation Execution Phase (2-3 states):
   - Intermediate value calculation
   - Final result calculation
   - Unit conversion handling
   
4. Verification Phase (1-2 states):
   - Answer reasonableness check
   - Error correction iteration
   
5. Final Submission Phase (1 state):
   - Answer formatting
   - Result submission

Total states: 6-10 (fine-grained for TGN optimization)"""
        }
    
    def _get_mmlu_template(self) -> Dict[str, str]:
        return {
            'task_description': """Multi-Domain Knowledge Question-Answering Task
Characteristics:
- Covers 57 academic subject areas
- Multiple-choice format (ABCD four options)
- Requires domain-specific professional knowledge
- Tests conceptual understanding and reasoning

Requirements:
1. Agents need to cover multiple knowledge domains
2. States should reflect analysis-reasoning-decision flow
3. Support cross-domain knowledge integration
4. Need confidence evaluation mechanism""",
            
            'state_granularity_guide': """State Decomposition Guide:
1. Problem Classification Phase (1 state):
   - Identify the subject domain
   
2. Knowledge Retrieval Phase (1-2 states):
   - Extract relevant concepts
   - Activate domain knowledge
   
3. Option Analysis Phase (2-3 states):
   - Analyze each option individually
   - Identify distractors
   
4. Reasoning and Decision Phase (1-2 states):
   - Logical reasoning
   - Option elimination
   
5. Confidence Evaluation Phase (1 state):
   - Assess answer credibility
   
6. Final Submission Phase (1 state):
   - Select and submit answer

Total states: 7-10"""
        }

    def _get_gpqa_template(self) -> Dict[str, str]:
        return {
            'task_description': """GPQA (Graduate-Level Multiple-Choice QA) Task
Characteristics:
- High-difficulty, graduate-level science questions (often multi-step reasoning)
- Multiple-choice format (A/B/C/D)
- Requires careful option elimination and consistency checks

Global Output Contract (CRITICAL):
- FINAL answer MUST be a SINGLE LETTER: A / B / C / D
- NO explanations, NO extra text, NO markdown
- If the prompt contains both a)/b)/c)/d) and a mapping to A/B/C/D, FOLLOW the final A/B/C/D mapping.
- If unsure, still choose the best option letter based on reasoning""",

            'state_granularity_guide': """State Decomposition Guide (7-10 states):
1. Question Parsing (1 state):
   - Extract what is being asked, key constraints, units

2. Concept Activation (1-2 states):
   - Recall relevant domain concepts / formulas

3. Option-by-Option Analysis (2-4 states):
   - Evaluate each option, identify contradictions/distractors

4. Cross-check & Elimination (1-2 states):
   - Sanity check, dimensional check, consistency with known facts

5. Final Decision (1 state):
   - Output ONLY the final option letter (A/B/C/D)"""
        }

    def _get_gaia_template(self) -> Dict[str, str]:
        return {
            'task_description': """GAIA (General AI Assistant) Benchmark Task
Characteristics:
- Mixed tool-using / multimodal / document-grounded reasoning
- Questions may reference external files (e.g., .xlsx/.pdf/.png/.docx/.csv/.txt) via a file_path
- Requires step-by-step planning, extraction from the given file when present, and final short answers

Global Output Contract:
- Provide ONLY the final answer (short), no extra commentary unless explicitly requested.
- If a numerical answer is required, return the number in the requested format.
- If the task requires reading an attached file, treat the attachment as authoritative evidence.

FSM Design Requirements:
1. Separate planning from evidence extraction (especially for file-backed questions).
2. Include an explicit verification / sanity-check step.
3. Keep states compact (6-10) and robust to cases with/without file attachments.""",

            'state_granularity_guide': """State Decomposition Guide (6-10 states):
1. Task Parsing (1 state):
   - Parse question, constraints, required output format
   - Detect whether an attachment is required (file_path exists)

2. Plan & Tool Selection (1 state):
   - Decide how to solve: web lookup vs file reading vs pure reasoning
   - Define intermediate quantities to compute

3. Evidence Acquisition (1-2 states):
   - If file_path exists: read/interpret the file (table/image/pdf/doc)
   - Otherwise: gather key facts from internal reasoning or allowed tools

4. Reasoning / Computation (1-3 states):
   - Perform calculations, logic, or multi-step synthesis

5. Verification (1 state):
   - Cross-check units, constraints, and consistency with evidence

6. Final Answer (1 state):
   - Emit final answer strictly in requested format"""
        }
    
    def _get_humaneval_template(self) -> Dict[str, str]:
        return {
            'task_description': """Python Code Generation Task (HumanEval)
Characteristics:
- Implement a single Python function based on signature and docstring
- Must pass hidden test suite
- Primary objective: correct, executable function

Global Output Contract (CRITICAL):
- FINAL answer MUST be single, syntactically valid Python FUNCTION
- No explanations, no markdown, no extra code
- Function name/signature MUST match given problem
- Use proper newlines/indentation

FSM Design Requirements:
1. Most states (except first understanding and final submission) SHOULD be code-producing
2. Almost every non-final state should output FULL updated function
3. Minimize purely analytical states
4. Model iterative code-writing/refinement process""",
            
            'agent_design_hints': """Agent Design Recommendations (code-centric):

1. Problem Reader (non-coding):
   - Reads the function signature and docstring.
   - Summarizes the requirements and edge cases in natural language.
   - MUST NOT output executable code.
   - Explicitly instructs coding agents to "output only a single valid Python function definition".

2. Code Writer (coding agent):
   - Main agent responsible for writing and rewriting the function implementation.
   - In EVERY state where this agent is responsible (except the final state), it MUST output a COMPLETE function definition.
   - Output format: ONLY the function definition, no explanation, no tests, no markdown.

3. Code Refiner / Debugger (coding agent):
   - Takes previous code and updates it to fix logical or edge-case issues.
   - ALWAYS outputs a FULL updated function definition (not patches, not comments).

4. Test Reasoner (non-coding):
   - Thinks about failing cases / edge cases.
   - Outputs ONLY natural-language analysis and instructions for coding agents; NEVER emits code.

5. Finalizer (coding agent) - 🚨 CRITICAL:
   - Runs only in the final submission state.
   - MUST output the FINAL production-ready Python function definition ONLY.
   - Output format: ONLY executable Python code wrapped in ```python and ``` markers.
   - ABSOLUTELY NO explanations, NO verification plans, NO natural language - ONLY CODE.
   - The output will be directly executed for test verification.
   - If Finalizer does not output code, the entire task FAILS.""",
            
            'state_granularity_guide': """State Decomposition Guide (code-heavy):

1. Initial Understanding (1 state, non-coding):
   - Read signature + docstring
   - Summarize requirements/edge cases
   - Output: natural-language notes

2. First Implementation Draft (1 state, coding):
   - Code Writer produces FIRST full implementation
   - Output: complete function definition

3. Guided Refinement Loops (3–5 states, mostly coding):
   - Alternate between:
     a) Test Reasoner: imagines failing cases
     b) Code Refiner: updates FULL function
   - Each coding state outputs entire updated function

4. Pre-Submission Check (optional, 1 state, non-coding):
   - Quick check for obvious mistakes
   
5. Final Submission (1 state, coding):
   - Finalizer outputs final polished function
   - Output: single function definition

Total states: 6–9, majority code-producing states"""
        }
    
    def _get_mbpp_template(self) -> Dict[str, str]:
        return {
            'task_description': """Basic Python Programming Task (MBPP)
Characteristics:
- Small self-contained Python tasks
- Function signature + assertion-based test cases
- Goal: implement function passing all tests

Global Output Contract (CRITICAL):
- FINAL answer MUST be single, syntactically valid Python FUNCTION
- Function name/parameters MUST match signature exactly
- Function body MUST be executable; no input()/print()
- Output pure code (no explanations, no markdown)
- **FORBID** multiple statements on one line (e.g., def f(): a=0; b=1)

FSM Design Requirements:
1. Most states SHOULD be code-writing/refinement states
2. FSM represents iterative coding: draft → refine → fix → finalize
3. Avoid long non-coding sequences
4. Each transition brings solution closer to passing tests""",

            'agent_design_hints': """Agent Design Recommendations (code-focused):

🚨 CRITICAL - FUNCTION SIGNATURE IS PROVIDED IN THE PROMPT:
The problem prompt ALREADY includes the exact function signature like:
  "Implement the function with this signature:
   def check_equilateral(x,y,z)"

You MUST use this EXACT function name and signature from the prompt!
NEVER rename functions to 'solve' or any other name.

1. Requirement_Analyst (non-coding):
   - Read the MBPP problem and the PROVIDED function signature.
   - The signature is given in the prompt as "Implement the function with this signature: def xxx(...)".
   - Summarize the requirements and edge cases.
   - MUST NOT output any executable code.
   - Remind coding agents to use the EXACT function signature from the prompt.

2. Code_Writer (coding agent, used in MANY states):
   - 🚨 Use the EXACT function signature provided in the prompt (e.g., "def check_equilateral(x,y,z)").
   - Output a COMPLETE function definition with the correct name and parameters.
   - Do NOT include explanations, comments, tests, or print statements — only the function code.
   - Wrap code in ```python and ``` markers.

3. Test_Reasoner (non-coding):
   - Propose test cases based on the provided assertions.
   - Verify the function name matches the signature in the prompt.
   - MUST NOT output executable code; only natural-language analysis.

4. Code_Refiner (coding agent, used in MULTIPLE states):
   - Take feedback and update the function while KEEPING the EXACT function name from the prompt.
   - Every time you act, output the FULL updated function definition (not a diff).
   - Wrap code in ```python and ``` markers.

5. Finalizer (coding agent, final state) - 🚨 CRITICAL:
   - MUST output the final Python function using the EXACT signature from the prompt.
   - Output format: ONLY executable Python code wrapped in ```python and ``` markers.
   - ABSOLUTELY NO explanations, NO 'ANSWER', NO natural language - ONLY CODE.
   - The output will be directly executed with the test assertions.
   - Wrong function name = ALL TESTS FAIL.

General Agent Principles:
- 🚨 The function signature is ALREADY provided in the prompt - USE IT EXACTLY.
- NEVER use generic names like 'solve', 'solution', 'func', etc.
- Coding roles always output a full function body wrapped in ```python and ```.""",
            
            'state_granularity_guide': """State Decomposition Guide (code-centric, 5–8 states):

1. Understand_Problem (1 state, non-coding):
   - Read MBPP prompt, restate goal/constraints
   - Output: brief summary and plan (no code)

2. Initial_Implementation (1–2 states, coding):
   - Code_Writer creates FIRST complete implementation
   - Output: full function definition

3. Test_Design_and_Bug_Hunting (1–2 states, non-coding):
   - Test_Reasoner inspects implementation + assertions
   - Proposes edge cases, identifies failure points
   - Output: detailed modification instructions

4. Refinement_Iterations (2–3 states, coding):
   - Code_Refiner updates function for edge cases
   - In EACH state, output ENTIRE updated function

5. Final_Submission (1 state, coding):
   - Finalizer outputs final clean function
   - Output ONLY function definition

Total states: 5–8, majority dedicated to executable code

Code formatting rules (MUST follow):
- GOOD: Multi-line statements with proper indentation
  ```python
  def add(a, b):
      result = a + b
      return result
  ```
- BAD: Multiple statements on one line
  ```python
  def add(a, b): result = a + b; return result
  ```"""
        }
    
    def _get_hotpotqa_template(self) -> Dict[str, str]:
        return {
            'task_description': """Multi-hop Question Answering Task
Characteristics:
- Requires reasoning across multiple documents
- Need to identify and connect supporting facts
- Questions involve bridge or comparison reasoning
- Long context with multiple document paragraphs

Requirements:
1. States should decompose multi-hop reasoning process
2. Agents need document retrieval and cross-document reasoning
3. Support extraction and verification of supporting facts
4. Handle both bridge (A→B→C) and comparison (A vs B) questions""",
            
            'state_granularity_guide': """State Decomposition Guide:
1. Question Analysis Phase (1-2 states):
   - Parse question type (bridge/comparison)
   - Identify required information
   
2. Document Retrieval Phase (2-3 states):
   - Extract relevant passages
   - Identify key entities
   
3. Multi-hop Reasoning Phase (2-4 states):
   - First hop: find initial fact
   - Intermediate hops: connect facts
   - Final hop: derive answer
   
4. Answer Synthesis Phase (1-2 states):
   - Combine evidence
   - Verify supporting facts
   
5. Final Submission Phase (1 state):
   - Format and submit answer

Total states: 7-11"""
        }
    
    def _get_alfworld_template(self) -> Dict[str, str]:
        return {
            'task_description': """Embodied AI Interactive Tasks (ALFWorld)
Characteristics:
- Text-based environment simulation in household scenarios
- Sequential action planning with state changes
- Goal-oriented navigation and object manipulation
- Requires: locate → pick up → (transform) → place/examine

Dataset Format (test.jsonl):
- goal: Natural language task description (e.g., "put a hot mug in coffeemachine")
- subgoals: Regex patterns to match expected action outcomes (2-3 patterns)
- difficulty: "easy" (simple pick-place) or "hard" (multi-step with state changes)
- Task types identified from additional_info description

Task Types (from test.jsonl, 80 samples):
1. pick_and_place_simple (difficulty: easy, ~15% of tasks)
   - Direct object transfer without state change
   - Examples: "put some saltshaker on drawer", "put a pencil in shelf"
   - Subgoals: [locate, pick up, place]

2. pick_two_obj_and_place (difficulty: hard, ~10%)
   - Find and place TWO identical objects in same location
   - Examples: "put two soapbar in garbagecan", "put two pillow in sofa"
   - Subgoals: [locate, pick up first, place first, locate second, pick up second, place second]

3. pick_heat_then_place (difficulty: hard, ~20%)
   - Pick → heat using microwave/stove → place
   - Examples: "put a hot mug in coffeemachine", "heat some apple and put it in fridge"
   - Subgoals: [locate, pick up, heat, place]

4. pick_cool_then_place (difficulty: hard, ~20%)
   - Pick → cool using fridge → place
   - Examples: "cool some lettuce and put it in countertop", "put a cool tomato in microwave"
   - Subgoals: [locate, pick up, cool, place]

5. pick_clean_then_place (difficulty: hard, ~25%)
   - Pick → clean using sink/sinkbasin → place
   - Examples: "clean some pan and put it in countertop", "put a clean mug in coffeemachine"
   - Subgoals: [locate, pick up, clean, place]

6. look_at_obj_in_light (difficulty: hard, ~10%)
   - Pick object → navigate to desklamp → examine under light
   - Examples: "examine the bowl with the desklamp", "look at pencil under the desklamp"
   - Subgoals: [locate object, pick up, locate desklamp, examine]

Available Actions:
- Navigation: go to [location]
- Manipulation: take [object] from [receptacle], put [object] in/on [receptacle]
- Interaction: open [receptacle], close [receptacle], toggle [appliance]
- State Change: use [receptacle] (for cleaning/heating/cooling)
- Observation: examine [object], look

Action Feedback Patterns:
- Success: "You pick up the [object] [id]", "You put the [object] in/on [receptacle]"
- State Change: "You heat/cool/clean the [object] [id] using the [appliance]"
- Failure: "Nothing happens", "There is no [object] here", "[Receptacle] is closed"

Evaluation:
- Subgoals are regex patterns that must be matched in action feedback
- Example: "^(?=.* you see)(?=.*a bowl \\d+)" → must see "you see" and "a bowl [number]"
- All subgoals must be satisfied sequentially to complete task

Requirements:
1. Parse goal to identify: target object(s), state transformation, destination
2. Plan action sequence based on task type
3. Execute actions and parse feedback for state confirmation
4. Match feedback against subgoal regex patterns
5. Handle failures: retry with alternative receptacles/paths
6. Output final action sequence as answer""",
            
            'state_granularity_guide': """State Decomposition Guide (Embodied AI - ALFWorld):

Based on test.jsonl task analysis (80 tasks):

Recommended FSM Structure (7 states):

1. Goal_Parsing_and_Planning (1 state) [INITIAL]:
   - Parse goal string to extract:
     * Target object type and count (one vs two)
     * Required state transformation (heat/cool/clean/none)
     * Destination receptacle
     * Task type (from 6 categories)
   - Output: Structured plan

2. Environment_Exploration (1 state):
   - Navigate rooms to locate target object
   - Look around, open receptacles if needed
   - Record object ID and location
   - Output: "Found [object] [id] in/on [receptacle]"

3. Object_Acquisition (1 state):
   - Navigate to object location
   - Execute "take [object] from [receptacle]"
   - Verify pickup with feedback: "You pick up the [object] [id]"
   - Output: Confirmation of successful pickup

4. State_Transformation (1 state) [OPTIONAL]:
   - Navigate to transformation appliance:
     * Heat: microwave, stove
     * Cool: fridge
     * Clean: sink, sinkbasin
   - Execute transformation action
   - Verify with feedback: "You [heat/cool/clean] the [object]"
   - Output: Transformed object ready

5. Object_Placement (1 state):
   - Navigate to destination receptacle
   - Execute "put [object] in/on [receptacle]"
   - Verify placement feedback
   - Output: Placement confirmation

6. Subgoal_Verification (1 state):
   - Match action feedback against subgoal regex patterns
   - Verify all required patterns satisfied
   - Output: Completion status

7. Final_Action_Sequence_Submission (1 state) [FINAL]:
   - Compile complete action sequence
   - Format: "action1 ; action2 ; action3 ; ..."
   - Submit using <|submit|> [action sequence]

Note: State transitions are learned by TGN, not predefined rules.
Total states: 7
Complexity: High (multi-step with state tracking)"""
        }
    
    def _get_math_template(self) -> Dict[str, str]:
        return {
            'task_description': """Competition Mathematics Task
Characteristics:
- Advanced mathematical reasoning
- Requires proof and derivation
- Symbolic manipulation and theorem application
- Multiple solution approaches possible

Requirements:
1. Deep problem analysis and strategy selection
2. Rigorous derivation and proof construction
3. Symbolic manipulation capabilities
4. Solution verification and alternative approaches""",
            
            'state_granularity_guide': """State Decomposition Guide:
1. Problem Analysis Phase (1-2 states):
   - Identify problem type and key concepts
   - Extract given conditions
   
2. Strategy Selection Phase (1-2 states):
   - Choose solution approach
   - Identify applicable theorems
   
3. Solution Derivation Phase (3-5 states):
   - Step-by-step derivation
   - Symbolic manipulation
   - Intermediate results
   
4. Verification Phase (1-2 states):
   - Check solution validity
   - Try alternative approaches
   
5. Final Submission Phase (1 state):
   - Format final answer

Total states: 8-12"""
        }
    
    def _get_general_template(self) -> Dict[str, str]:
        return {
            'task_description': """General Problem Solving Task""",
            'state_granularity_guide': """Recommended 6-10 states"""
        }
    
    def generate_state_plan(self, dataset: str, task_description: Optional[str] = None) -> Tuple[Dict, LLM]:
        """
        Step 1: Generate the state plan.
        """
        template = self.dataset_templates.get(dataset.lower())
        if not template:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
        prompt = f'''You are a workflow designer for {dataset.upper()} problem-solving.

🚨 OUTPUT REQUIREMENT: Output ONLY pure JSON, NO Python code (no def, no return, no functions).

📋 TASK CONTEXT:
{template['task_description']}

🏗️ STATE GRANULARITY GUIDE:
{template['state_granularity_guide']}

🎯 YOUR MISSION:
Design a workflow of states (6-10 states) for solving {dataset.upper()} problems.
Each state represents a distinct processing phase.

📝 OUTPUT FORMAT (pure JSON only):
```json
{{
  "workflow_reasoning": "<Explain your workflow design>",
  "states": [
    {{
      "state_name": "<Phase Name>",
      "description": "<What happens in this phase>",
      "required_capabilities": ["<capability1>", "<capability2>"],
      "is_initial": true/false,
      "is_final": true/false
    }},
    ...
  ]
}}
```

REQUIREMENTS:
- 6-10 states total
- Exactly 1 initial state
- At least 1 final state
- Each state describes a clear processing phase
- Focus on WHAT needs to be done, not WHO does it
'''
        
        llm = LLM(system_prompt=prompt, use_azure=self.use_azure)
        response = llm.chat(message=f"Design workflow for {dataset.upper()}")
        
        try:
            state_plan_json = self._extract_json(response)
            state_plan = json.loads(state_plan_json)
            print(f"✅ Generated workflow plan with {len(state_plan['states'])} states")
            return state_plan, llm
        except Exception as e:
            print(f"❌ Step 1 failed: {e}")
            print(f"📄 Response (first 500 characters): {response[:500]}")
            raise
    
    def generate_agents_from_states(self, dataset: str, state_plan: Dict, task_description: Optional[str] = None) -> Tuple[List[Dict], LLM]:
        """
        Step 2: Generate agents from states.
        """
        state_summary = "\n".join([
            f"  {i}. {s['state_name']}: {s['description']}\n     Needs: {', '.join(s['required_capabilities'])}"
            for i, s in enumerate(state_plan['states'])
        ])
        template = self.dataset_templates.get(dataset.lower(), {})
        agent_hints = template.get('agent_design_hints', "")
        
        prompt = f'''You are designing agents for a {dataset.upper()} problem-solving system.

🚨 OUTPUT REQUIREMENT: Output ONLY pure JSON array, NO Python code (no def, no return, no functions).

📋 WORKFLOW STATES:
{state_summary}

🧠 AGENT DESIGN HINTS (if provided by the dataset template):
{agent_hints}

🎯 YOUR MISSION:
Design agents that can fulfill ALL the capabilities needed across these states.
- Number of agents: {len(state_plan['states'])} (one primary agent per state, but agents can be reused)
- Each agent should have clear expertise matching state requirements

📝 OUTPUT FORMAT (pure JSON only):
```json
[
  {{
    "agent_id": "0",
    "name": "<Agent Name>",
    "role": "<Brief role>",
    "system_prompt": "<Detailed capabilities and responsibilities>",
    "expertise": ["<expertise1>", "<expertise2>"]
  }},
  ...
]
```
'''
        
        llm = LLM(system_prompt=prompt, use_azure=self.use_azure)
        response = llm.chat(message=f"Design agents for workflow with {len(state_plan['states'])} states")
        
        try:
            agent_json = self._extract_json(response)
            agent_dict = json.loads(agent_json)
            print(f"✅ Generated {len(agent_dict)} agents for {len(state_plan['states'])} states")
            return agent_dict, llm
        except Exception as e:
            print(f"❌ Step 2 failed: {e}")
            print(f"📄 Response (first 500 characters): {response[:500]}")
            raise
    
    def generate_fsm_states(self, dataset: str, state_plan: Dict, agent_dict: List[Dict], task_description: Optional[str] = None) -> Tuple[Dict, LLM]:
        """
        Step 3: Directly combine Step 1 + Step 2 to build the complete FSM.
        
        ✨ Core improvement: no longer call the LLM here; assemble the FSM directly from state_plan and agent_dict.

        Compatibility notes:
        - Older example scripts may call `generate_fsm_states(dataset, agent_dict)`
        - In that case, call `generate_state_plan` externally to obtain state_plan first,
          then call this method; internally, this training framework only uses this method via `generate_complete_mas`
        """
        print("📝 Step 3: Assembling FSM from state plan + agents...")
        
        # Build the states list
        states = []
        for i, state_info in enumerate(state_plan['states']):
            # Find the corresponding agent
            agent = agent_dict[i] if i < len(agent_dict) else agent_dict[-1]
            
            state = {
                "state_id": str(i),
                "state_name": state_info['state_name'],
                "agent_id": agent['agent_id'],
                "instruction": f"{state_info['description']}. Output format: complete your task and include completion marker.",
                "completion_condition": f"Output contains task completion marker or {state_info['state_name']} objectives met",
                "is_initial": state_info.get('is_initial', i == 0),
                "is_final": state_info.get('is_final', False),
                "listeners": self._generate_listeners(i, len(state_plan['states']), agent_dict)
            }
            states.append(state)
        
        # Ensure there is an initial state and a final state
        if not any(s['is_initial'] for s in states):
            states[0]['is_initial'] = True
        if not any(s['is_final'] for s in states):
            states[-1]['is_final'] = True
        
        # Build the transition list (sequential chain)
        transitions = []
        for i in range(len(states) - 1):
            transitions.append({
                "from_state": str(i),
                "to_state": str(i + 1),
                "condition": f"State {i} completion_condition is met",
                "priority": 1
            })
        
        # Add retry transitions for errors
        for i in range(1, len(states) - 1):
            transitions.append({
                "from_state": str(i),
                "to_state": str(max(0, i - 1)),
                "condition": f"State {i} detected errors requiring retry",
                "priority": 2
            })
        
        fsm = {
            "states": states,
            "transitions": transitions
        }
        
        # Validate the FSM
        if self._validate_fsm(fsm, agent_dict):
            print(f"✅ Assembled FSM with {len(states)} states and {len(transitions)} transitions")
            # Return a dummy LLM instance (used for accounting)
            dummy_llm = LLM(system_prompt="", use_azure=self.use_azure)
            return fsm, dummy_llm
        else:
            raise ValueError("FSM validation failed")
    
    def _generate_listeners(self, state_idx: int, total_states: int, agent_dict: List[Dict]) -> List[str]:
        """Generate the listener list."""
        listeners = []
        # Earlier states can listen
        for j in range(max(0, state_idx - 2), state_idx):
            if j < len(agent_dict):
                listeners.append(agent_dict[j]['agent_id'])
        # Later states can also listen (for broadcasting)
        for j in range(state_idx + 1, min(state_idx + 3, len(agent_dict))):
            if j < len(agent_dict):
                listeners.append(agent_dict[j]['agent_id'])
        return listeners
    
    def _validate_fsm(self, fsm: Dict, agent_dict: List[Dict]) -> bool:
        """Validate the FSM structure."""
        if 'states' not in fsm or 'transitions' not in fsm:
            return False
        
        states = fsm['states']
        if not states:
            return False
        
        # Check the initial and final states
        has_initial = any(s.get('is_initial', False) for s in states)
        has_final = any(s.get('is_final', False) for s in states)
        
        return has_initial and has_final
    
    def _extract_json(self, response: str) -> str:
        """Extract JSON from an LLM response (robust version)."""
        if not response:
            return ""
        
        # Check whether this is an HTML error page
        if response.strip().startswith('<!DOCTYPE') or response.strip().startswith('<html'):
            raise ValueError("LLM API returned HTML error page (likely timeout or server error)")
        
        # First priority: extract a ```json ... ``` code block
        if "```json" in response:
            parts = response.split("```json")
            if len(parts) > 1:
                json_part = parts[1].split("```")[0]
                json_part = json_part.strip()
                if json_part:
                    return self._clean_json(json_part)
        
        # Second priority: extract any ``` ... ``` block starting with { or [
        if "```" in response:
            parts = response.split("```")
            for part in parts:
                candidate = part.strip()
                if candidate.startswith("{") or candidate.startswith("["):
                    return self._clean_json(candidate)
        
        # Handle JSON wrapped inside Python code (for example: def func(): return {...})
        python_return_match = re.search(r'return\s*(\{|\[)', response, re.MULTILINE | re.DOTALL)
        if python_return_match:
            start = python_return_match.start() + len('return')
            while start < len(response) and response[start].isspace():
                start += 1
            if start < len(response):
                candidate = response[start:]
                json_str = self._extract_balanced_json(candidate)
                if json_str:
                    return self._clean_json(json_str)
        
        # Next: search the full text for the first { or [
        text = response.strip()
        if not text:
            return ""
        
        first_curly = text.find("{")
        first_square = text.find("[")
        indices = [i for i in (first_curly, first_square) if i != -1]
        if indices:
            start = min(indices)
            candidate = text[start:]
            json_str = self._extract_balanced_json(candidate)
            if json_str:
                return self._clean_json(json_str)
        
        # Final fallback
        return self._clean_json(text)
    
    def _extract_balanced_json(self, text: str) -> str:
        """Extract complete JSON using bracket matching."""
        if not text:
            return ""
        
        start_char = text[0]
        if start_char not in ('{', '['):
            return ""
        
        # Use a stack to track bracket matching
        stack = [start_char]
        in_string = False
        escape_next = False
        
        i = 1
        while i < len(text) and stack:
            char = text[i]
            
            if escape_next:
                escape_next = False
                i += 1
                continue
            
            if char == '\\':
                escape_next = True
                i += 1
                continue
            
            if char == '"':
                in_string = not in_string
                i += 1
                continue
            
            if in_string:
                i += 1
                continue
            
            # Process brackets
            if char in ('{', '['):
                stack.append(char)
            elif char == '}':
                if stack and stack[-1] == '{':
                    stack.pop()
            elif char == ']':
                if stack and stack[-1] == '[':
                    stack.pop()
            
            i += 1
        
        # Verify the stack is empty and the last character is a closing bracket
        if not stack:
            result = text[:i]
            last_char = result.rstrip()[-1] if result.rstrip() else ''
            if last_char in ('}', ']'):
                return result
        
        return ""
    
    def _clean_json(self, json_str: str) -> str:
        """Clean common issues in a JSON string."""
        if not json_str:
            return json_str
        
        # Remove BOM and other invisible characters
        json_str = json_str.strip('\ufeff\u200b\u200c\u200d')
        
        # Remove // single-line comments
        lines = json_str.split('\n')
        cleaned_lines = []
        for line in lines:
            if '//' in line:
                parts = line.split('//')
                if len(parts) > 1:
                    before_comment = parts[0]
                    quote_count = before_comment.count('"') - before_comment.count('\\"')
                    if quote_count % 2 == 0:  # Quotes are balanced
                        line = parts[0].rstrip()
            cleaned_lines.append(line)
        json_str = '\n'.join(cleaned_lines)
        
        # Remove trailing commas
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        return json_str
    
    def generate_complete_mas(self, dataset: str, save_path: Optional[str] = None, task_description: Optional[str] = None) -> Tuple[Dict, float]:
        """
        Generate a MAS using the full three-step process.
        """
        print(f"🚀 Generating Multi-Agent System for {dataset.upper()}...")
        print("   Using 3-Step Assembly: State Plan → Agents → FSM Assembly ✨")
        
        # Step 1: Generate the state plan
        print("\n📝 Step 1: Designing workflow states...")
        state_plan, state_llm = self.generate_state_plan(dataset, task_description)
        
        # Step 2: Generate agents
        print("\n📝 Step 2: Generating agents for workflow...")
        agent_dict, agent_llm = self.generate_agents_from_states(dataset, state_plan, task_description)
        
        # Step 3: Assemble the FSM (without calling the LLM)
        print("\n📝 Step 3: Assembling FSM...")
        fsm, _ = self.generate_fsm_states(dataset, state_plan, agent_dict, task_description)
        
        # Assemble the complete system
        complete_system = {
            "dataset": dataset,
            "agents": agent_dict,
            "fsm": fsm,
            "metadata": {
                "num_agents": len(agent_dict),
                "num_states": len(fsm['states']),
                "num_transitions": len(fsm['transitions'])
            }
        }
        
        # Compute the cost (only Step 1 and Step 2 called the LLM)
        state_usage = state_llm.get_token_usage()
        agent_usage = agent_llm.get_token_usage()
        
        total_input_tokens = state_usage['input_tokens'] + agent_usage['input_tokens']
        total_output_tokens = state_usage['output_tokens'] + agent_usage['output_tokens']
        total_tokens = total_input_tokens + total_output_tokens
        
        state_cost = state_llm.calculate_cost_usd()
        agent_cost = agent_llm.calculate_cost_usd()
        total_cost_usd = state_cost + agent_cost
        
        # Save
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(complete_system, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Saved to: {save_path}")
        
        print(f"\n✅ Generation Complete!")
        print(f"   - Agents: {len(agent_dict)}")
        print(f"   - States: {len(fsm['states'])}")
        print(f"   - Transitions: {len(fsm['transitions'])}")
        print(f"   - Input Tokens: {total_input_tokens}")
        print(f"   - Output Tokens: {total_output_tokens}")
        print(f"   - Total Tokens: {total_tokens}")
        print(f"   - Cost: ${total_cost_usd:.4f} USD")
        
        return complete_system, total_cost_usd


def generate_enhanced_mas(dataset: str,
                          save_path: Optional[str] = None,
                          use_azure: bool = False,
                          task_description: Optional[str] = None) -> Tuple[Dict, float]:
    """
    Compatibility helper for existing example scripts:
    - Wraps EnhancedFSMGenerator.generate_complete_mas
    - Called directly by scripts such as generate_enhanced_mas_examples.py
    """
    generator = EnhancedFSMGenerator(use_azure=use_azure)
    system, cost = generator.generate_complete_mas(
        dataset=dataset,
        save_path=save_path,
        task_description=task_description,
    )
    return system, cost
