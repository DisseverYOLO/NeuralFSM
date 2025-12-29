# NeuralFSM: Neural Finite State Machine for Multi-Agent Collaboration and Protection

This repository contains the **NeuralFSM** codebase for training and evaluating a **finite-state, task-conditioned multi-agent coordination** framework with a **Temporal Graph Network (TGN)** controller and an **optional dual-defense protection layer** (robustness under frequency/semantic attacks).


## Project structure

- `run_experiment_1_fsm_complete.py`: main entry to run Experiment 1
- `run_experiment_2_fsm_protected_v2.py`: robustness/protection experiments (attack vs. protected)
- `neural_fsm_mas/`: NeuralFSM training/inference implementation
- `fsm_cache/`: cached master FSM scaffolds (may be ignored depending on your needs)

## Setup

### Requirements

- Python **3.9–3.11** recommended

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quickstart

### Experiment 1 (example)

```bash
python run_experiment_1_fsm_complete.py --domains gsm8k --dataset_root ./datasets --output_dir ./results/experiment1_fsm --num_epochs 1 --batch_size 4 --llm_name gpt-5-nano --max_transitions 8
```

### Robustness / protection (example)

```bash
python run_experiment_2_fsm_protected_v2.py --domains gsm8k --dataset_root ./datasets --output_dir ./results/experiment2_fsm --llm_name gpt-5-nano
```

## Notes on datasets and outputs

- `datasets/` is intentionally ignored. Put your datasets under `./datasets` following the expected layout used by the scripts.
- `results/` is ignored. Outputs (summaries, checkpoints) will be written there by default.

## License

MIT License

