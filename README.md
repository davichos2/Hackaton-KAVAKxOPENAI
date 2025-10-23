# 🤖 LLM Parameter Optimizer with Genetic Algorithm

An automated system for optimizing Large Language Model (LLM) parameters using genetic algorithms and LLM-as-Judge evaluation.

## 📋 Overview

This system automatically finds the best configuration for your chatbot by:
1. **Testing** your chatbot with a set of prompts
2. **Evaluating** responses using multiple metrics (quality, latency, cost, safety)
3. **Optimizing** parameters using a genetic algorithm
4. **Visualizing** progress with automatic plots

### Key Features
- 🧬 **Genetic Algorithm** optimization for 7 LLM parameters
- 📊 **Multi-Metric Evaluation**: Quality, latency, cost, and safety scores
- 🎨 **Automatic Visualization**: Real-time progress plots
- 💾 **Comprehensive Logging**: CSV and JSON output for analysis
- 🔄 **Iterative Improvement**: Converges to optimal parameters over time

## 🏗️ Architecture

```
┌─────────────────┐
│  prompts.txt    │ (20 test prompts)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   agent_evaluator.py (Main Loop)   │
│  • Loads prompts & config           │
│  • Calls V3.py chatbot              │
│  • Evaluates with LLM-as-Judge      │
│  • Logs results                     │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│      heuristic.py (Optimizer)       │
│  • Genetic Algorithm                │
│  • Population: 20 individuals       │
│  • Tournament selection             │
│  • Crossover + Mutation             │
│  • Returns new params to test       │
└─────────────────────────────────────┘
```

## 📁 Project Structure

```
your_project/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .env                         # API keys (create this)
│
├── agent_evaluator.py          # Main optimization loop
├── heuristic.py                # Genetic algorithm
├── V3.py                       # Your chatbot (original)
├── V3_optimized.py            # Modified version with params
│
├── prompts.txt                 # 20 test prompts
├── config.json                 # Initial parameters
│
├── plot_results.py             # Standalone plotting tool
│
└── outputs/                    # Created during run
    ├── results.csv                      # Per-prompt results
    ├── optimization_progress.csv        # Iteration summaries
    ├── optimization_progress.png        # Automatic plots
    ├── optimization_summary.png         # Summary visualization
    ├── optimization_history.json        # Complete history
    ├── iteration_1.json                 # Detailed per iteration
    ├── iteration_2.json
    ├── ...
    ├── proposal_1.json                  # Heuristic proposals
    └── proposal_2.json
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd your_project

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create `.env` file with your OpenAI API key:
```env
OPENAI_API_KEY=sk-your-actual-key-here
```

Verify you have these files:
- ✅ `prompts.txt` (20 prompts, one per line)
- ✅ `config.json` (initial parameters)
- ✅ `agent_evaluator.py`
- ✅ `heuristic.py`
- ✅ `V3.py` or `V3_optimized.py`

### 3. Run Optimization

```bash
# Basic run (5 iterations)
python agent_evaluator.py --prompts prompts.txt --config config.json

# Custom iterations
python agent_evaluator.py --prompts prompts.txt --config config.json --iterations 10

# With multiple repeats (reduce variance)
python agent_evaluator.py --prompts prompts.txt --config config.json --iterations 10 --repeats 2
```

### 4. View Results

Results are automatically generated during the run:
- `optimization_progress.png` - Progress visualization
- `optimization_progress.csv` - Data for custom analysis

Or regenerate plots anytime:
```bash
python plot_results.py
```

## 📊 Output Files Explained

### `results.csv`
Complete log of every prompt evaluation:
```csv
timestamp,prompt,response,quality_score,latency_ms,params,...
```

### `optimization_progress.csv`
High-level iteration summaries (used for plotting):
```csv
iteration,mean_quality,median_quality,std_quality,mean_latency_ms,...
```

### `optimization_history.json`
Complete history including:
- Parameters tested each iteration
- All prompt results
- Aggregate statistics

### `iteration_N.json`
Full details for iteration N:
```json
{
  "iteration": 1,
  "params": {...},
  "aggregate": {"mean_quality": 72.5, ...},
  "per_prompt": [...]
}
```

## 🎯 Parameters Being Optimized

The system optimizes these 7 parameters:

| Parameter | Range | Description |
|-----------|-------|-------------|
| `temperature` | 0.2 - 1.5 | Creativity vs consistency |
| `top_p` | 0.6 - 1.0 | Nucleus sampling threshold |
| `max_tokens` | 20 - 200 | Maximum response length |
| `presence_penalty` | -2.0 - 2.0 | Penalize token presence |
| `frequency_penalty` | -2.0 - 2.0 | Penalize token frequency |
| `retrieval_top_k` | 10 - 100 | Number of retrieval results |
| `num_beams` | 1 - 5 | Beam search width |

## 📈 Evaluation Metrics

### Quality Score (0-100)
Weighted combination:
- 30% **Embedding Relevance** - Semantic similarity via embeddings
- 10% **LLM Relevance** - Rated by LLM-as-Judge
- 30% **Factuality** - Accuracy of information
- 15% **Clarity** - Ease of understanding
- 10% **Tone Adherence** - Appropriate tone
- 5% **Hallucination Check** - No false information

### Latency Score (0-100)
- Target: 800ms (configurable)
- Score decreases as latency increases

### Cost Score (0-100)
- Target: $0.001 per call (configurable)
- Based on token usage

### Safety Score (0-100)
- Based on hallucination detection
- 100 = no hallucinations

## 🧬 Genetic Algorithm Details

### Population
- Size: 20 individuals
- Each individual = one parameter configuration

### Selection
- **Tournament selection** (k=3)
- Best individual from 3 random candidates

### Crossover
- **Two-point crossover**
- Creates 2 children from 2 parents

### Mutation
- Rate: 20%
- Random perturbations within valid ranges

### Elitism
- Best individuals preserved across generations

## 🎨 Visualization

The system generates two plots:

### `optimization_progress.png`
6-panel detailed view:
1. Quality score with confidence bands
2. Best score cumulative
3. All metrics comparison
4. Quality variability
5. Latency trend
6. Iteration-to-iteration improvement

### `optimization_summary.png`
2-panel summary:
1. Quality progress over time
2. Parameter evolution heatmap

## ⚙️ Configuration Options

### Environment Variables

```bash
# Set custom targets
export TARGET_LATENCY_MS=1000
export TARGET_COST_PER_CALL=0.002

# Then run
python agent_evaluator.py --prompts prompts.txt --config config.json
```

### Initial Config (`config.json`)

```json
{
  "temperature": 0.7,
  "top_p": 0.9,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "max_tokens": 100,
  "retrieval_top_k": 50,
  "num_beams": 1
}
```

### Genetic Algorithm (`heuristic.py`)

```python
# Adjust these constants
POPULATION_SIZE = 20      # Population size
MUTATION_RATE = 0.2       # Mutation probability
```

## 📝 Example Run

```bash
$ python agent_evaluator.py --prompts prompts.txt --config config.json --iterations 5

📝 Progress log initialized: optimization_progress.csv
✓ Loaded 20 prompts
✓ Loaded initial config with 7 parameters
✓ V3 Agent ready

============================================================
🔄 ITERATION 1/5
============================================================
Testing with parameters:
{
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 100,
  ...
}
============================================================

Evaluating prompts: 100%|████████████████| 20/20 [02:30<00:00]

📊 Results Summary:
  Mean Quality: 68.45
  Median Quality: 70.00
  Std Quality: 9.12
  Mean Latency: 1450ms

🧬 Calling heuristic for next parameters...
🧬 Genetic Algorithm - Generation 0
   Initializing population of 20 individuals...
   Proposing untested individual 5
   Best fitness so far: 68.45

📊 Progress plots saved to: optimization_progress.png

============================================================
🔄 ITERATION 2/5
...

============================================================
✅ Optimization complete!
============================================================
Results saved to:
  • results.csv - Detailed per-prompt results
  • optimization_progress.csv - Iteration summary (for plotting)
  • optimization_progress.png - Progress visualization
  • iteration_*.json - Full iteration data
  • optimization_history.json - Complete run history
============================================================

🏆 Best Iteration: 4
   Quality Score: 78.92
   Parameters: {
     "temperature": 0.65,
     "top_p": 0.87,
     "max_tokens": 125,
     ...
   }
```

## 🔧 Troubleshooting

### "Could not import V3 agent"
```bash
pip install agents
# Verify .env file exists with OPENAI_API_KEY
```

### "Embedding error"
- Check OpenAI API key has embeddings access
- Verify internet connection
- Try: `pip install --upgrade openai`

### "No JSON found in evaluator output"
- LLM-as-Judge sometimes fails to return valid JSON
- Script uses fallback scores (50 for all metrics)
- This won't break the optimization

### Plots not generating
```bash
pip install matplotlib
```

### Slow execution
- Reduce `--iterations` for testing
- Remove `create_progress_plots()` call inside loop
- Use `--repeats 1` (default)

## 📊 Analyzing Results

### Find Best Parameters

```python
import json

with open('optimization_history.json') as f:
    history = json.load(f)

best = max(history, key=lambda x: x['aggregate']['mean_quality'])
print(f"Best iteration: {best['iteration']}")
print(f"Quality: {best['aggregate']['mean_quality']:.2f}")
print("Parameters:", json.dumps(best['params'], indent=2))
```

### Plot Custom Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('optimization_progress.csv')

plt.figure(figsize=(10, 6))
plt.plot(df['iteration'], df['mean_quality'], 'bo-', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Mean Quality Score')
plt.title('Optimization Progress')
plt.grid(True, alpha=0.3)
plt.savefig('my_custom_plot.png', dpi=300)
```

### Compare Parameter Configurations

```python
import pandas as pd
import json

df = pd.read_csv('results.csv')
df['params_dict'] = df['params'].apply(json.loads)

# Group by unique parameter configurations
grouped = df.groupby('params')['final_quality_score'].agg(['mean', 'std', 'count'])
print(grouped.sort_values('mean', ascending=False).head(10))
```

## 🎓 Best Practices

1. **Start with 5-10 iterations** to verify everything works
2. **Use diverse prompts** that cover your use cases
3. **Monitor quality trends** - should improve over iterations
4. **Test final params** on held-out validation prompts
5. **Run multiple times** to ensure consistency
6. **Adjust parameter ranges** if hitting boundaries frequently
7. **Use repeats** (`--repeats 2`) for more stable results

## 🚀 Advanced Usage

### Custom Objective Function

Edit `aggregate_scores()` in `agent_evaluator.py` to change weights:

```python
quality_score = (
    0.40 * embedding_relevance +  # Increased weight
    0.20 * llm_rel +               # Increased weight
    0.20 * factual,
    0.10 * clarity,
    0.05 * tone,
    0.05 * halluc
)
```

### Different Heuristic

Replace `heuristic.py` with your own optimizer:
- Bayesian Optimization
- Simulated Annealing
- Particle Swarm Optimization
- Random Search

Just implement `propose(params, iteration_summary) -> params`

### Batch Evaluation

Modify `evaluate_batch()` to parallelize prompt evaluation

## 📚 Dependencies

```
openai>=1.0.0          # OpenAI API
numpy>=1.24.0          # Numerical computations
pandas>=2.0.0          # Data handling
tqdm>=4.65.0           # Progress bars
python-dotenv>=1.0.0   # Environment variables
agents>=0.1.0          # Agent framework
matplotlib>=3.7.0      # Plotting
```

## 📄 License

MIT License - Feel free to modify and use for your projects!

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional optimization algorithms
- More evaluation metrics
- Parallel evaluation
- Multi-objective optimization
- Real-time monitoring dashboard

## 📧 Support

If you encounter issues:
1. Check the Troubleshooting section
2. Review output logs in `results.csv`
3. Try with fewer iterations first
4. Verify all dependencies are installed

## 🎉 Acknowledgments

Built with:
- OpenAI API for LLM capabilities
- Genetic Algorithm for optimization
- LLM-as-Judge for evaluation
