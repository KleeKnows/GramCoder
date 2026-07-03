# GramCoder

## Project Overview

GramCoder is a research framework for AI-powered code generation using Large Language Models. It implements and benchmarks various prompting strategies for solving programming problems, with the **GramCoder** method featuring graph-based reasoning, retrieval-augmented generation, and multi-stage refinement.

## Key Commands

### Running the Main System
```bash
# Basic usage
python script/main.py --method GramCoder --model gpt4o --dataset MBPP --language Python3 --temperature 0.2 --pass_at_k 1

# Available methods: GramCoder, MapCoder, Direct, CoT, SelfPlanning, Analogical
# Available models: gpt4, gpt4o, O1, gpt5, llama
# Available datasets: HumanEval, MBPP, APPS, CC (CodeContest)
```