# Creative Writing Benchmark v3

🎨 Welcome to the Creative Writing Benchmark v3 repository! This benchmark evaluates the creative writing capabilities of large language models using a hybrid rubric and Elo scoring system, designed for enhanced discrimination, especially at the top end of model performance. This is the system used for the Creative Writing leaderboard on [EQ-Bench.com](https://eqbench.com/creative_writing.html).

## How the Benchmark Works

The evaluation process involves several steps:

1.  **Generation:** The model under test generates responses to 32 distinct writing prompts across 3 iterations (96 items total). Generation uses a temperature of 0.7 and min_p of 0.1 to encourage creativity while maintaining some consistency.
2.  **Rubric Scoring:** Each generated piece is individually assessed by a judge model (Anthropic's Claude Sonnet 4.6 recommended for leaderboard parity) against a comprehensive rubric.
3.  **Initial Elo Inference:** The aggregate rubric score is used to estimate an initial Elo rating for the model being evaluated relative to existing models.
4.  **Pairwise Matchups (Sparse):** The model is compared against neighboring models on the leaderboard using pairwise matchups. The judge determines the better output across several criteria, assigning a score margin (using `+` symbols).
5.  **Glicko Calculation:** Elo scores are calculated using the Glicko-2 rating system, modified to incorporate the win margin (number of `+`'s) from pairwise comparisons. This process loops until model positions stabilize.
6.  **Pairwise Matchups (Comprehensive):** More thorough pairwise comparisons are conducted with the model's final neighbors.
7.  **Final Elo Calculation:** The definitive leaderboard Elo score is computed based on all comparisons.
8.  **Normalization:** Raw Elo scores are normalized by anchoring specific models (e.g., `deepseek/deepseek-r1` to 1500, `mistralai/ministral-3b` to 200) to ensure comparability over time.

More info here: [https://eqbench.com/about.html#creative-writing-v3](https://eqbench.com/about.html#creative-writing-v3)

## Key Features

*   **Hybrid Scoring:** Combines isolated rubric scoring with more discriminative pairwise Elo comparisons.
*   **Glicko-2 System:** Uses a robust rating system that accounts for rating uncertainty and volatility, adapted to weight win margins.
*   **Discriminative Prompts:** Prompts are designed to challenge models in areas like humor, romance, spatial awareness, and unique perspectives.
*   **Bias Mitigation:** Incorporates strategies to reduce known LLM judge biases (Length, Position, Verbosity, Poetic Incoherence).
*   **Iteration-Based:** Runs multiple iterations per prompt to account for generation variability.

## Usage

### Prerequisites

*   Python 3.x
*   API keys for the test and judge models (compatible with OpenAI/OpenRouter API format).
*   Required Python packages.

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/EQ-bench/creative-writing-bench.git
    cd creative-writing-bench
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    # Or manually:
    # pip install requests python-dotenv numpy scipy tqdm glicko2 nltk joblib
    ```
    You also need to download NLTK data:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('cmudict')
    ```

3.  **Configure API Keys:**
    *   Copy the example environment file: `cp .env.example .env`
    *   Edit the `.env` file and add your API keys and desired endpoint URLs for the test and judge models. You can also adjust timeouts and retries here.

### Running the Benchmark

Execute the main script with your desired parameters. For a leaderboard-comparable score, use the recommended judge model and the provided runs file:

```bash
python3 creative_writing_bench.py \
    --test-model "your-model-provider/your-model-name" \
    --judge-model "claude-sonnet-4-6" \
    --runs-file "creative_bench_runs.json" \
    --creative-prompts-file "data/creative_writing_prompts_v3.json" \
    --run-id "my_model_run_1" \
    --threads 500 \
    --verbosity "INFO" \
    --iterations 3
```

**Important Arguments:**

*   `--test-model`: Identifier for the model you want to evaluate.
*   `--judge-model`: Identifier for the judge model (use `claude-sonnet-4-6` for leaderboard scores).
*   `--runs-file`: Path to the JSON file storing run data. **Crucially, to get an Elo score comparable to the EQ-Bench leaderboard, you *must* use the `creative_bench_runs.json` file provided in this repository**, as it contains the necessary historical data for Elo calculation. Start with the one provided here. Subsequent runs will update this file.
*   `--iterations`: Number of generation iterations per prompt (default and recommended: 3).
*   `--run-id`: A unique prefix for this specific run attempt. Helps organize results if you run the same model multiple times.
*   `--threads`: Number of parallel threads for generation and judging (adjust based on your API rate limits and system). High numbers like 500 assume generous rate limits.
*   `--verbosity`: Logging level (e.g., `DEBUG`, `INFO`).


### Canonical Leaderboard Results

Leaderboard results are saved in `creative_bench_runs.zip` and `elo_results.zip`. If you would to compare a result against the leaderboard models, unzip these into the root repository dir and the eval pipeline will use them in ELO matchups (assuming you are using default run file paths), giving you a leaderboard-comparable result.

These canonical zip files may not be always updated, so if you need the latest results, ping contact@eqbench.com.

Quick example for leaderboard reproduction:

```bash
# ensure .env is set up for your api, then:
unzip creative_bench_runs.zip
unzip elo_results.zip
python3 creative_writing_bench.py \
    --test-model "your-model-provider/your-model-name" \
    --judge-model "claude-sonnet-4-6" \
    --runs-file "creative_bench_runs.json" \
    --run-id "my_model_run_1" \
    --iterations 3
```



### Understanding the Output

*   Progress will be logged to the console.
*   Detailed run data, including generated text and judge scores, is saved in the specified `--runs-file` (e.g., `creative_bench_runs.json`).
*   Elo analysis results, including pairwise comparisons and final ratings, are stored in `elo_results.json`.
*   The final normalized Elo score (`elo_norm`) for your test model will be printed at the end and saved in `elo_results.json`. This is the score comparable to the EQ-Bench leaderboard.

## Benchmark Philosophy

Evaluating creative writing is inherently subjective. This benchmark aims to provide a reliable *relative* ranking by:

*   Using a judge model (Sonnet 4.6) known for decent literary assessment.
*   Employing pairwise comparisons for better discrimination than rubric scores alone.
*   Choosing prompts that deliberately expose model weaknesses, creating a steeper evaluation gradient.
*   Acknowledging and attempting to mitigate known LLM judge biases.

However, no benchmark is perfect. Always supplement scores by reading sample outputs and forming your own judgment.

## Scoring System: Rubric vs. Elo

*   **Rubric Score:** An aggregate score based on judging each piece in isolation against a detailed rubric. Provides insight into specific criteria but can saturate at high performance levels.
*   **Elo Score:** A relative rating derived from pairwise comparisons against other models. More discriminative, especially at the top end, but sensitive to the pool of compared models.

These scores measure different aspects and may not always align perfectly due to judging methodology differences and criteria variations. The **normalized Elo score (`elo_norm`)** is the primary metric used for the leaderboard ranking.

## Bias Mitigation

We attempt to control for several biases common in pairwise LLM judging:

*   **Length Bias:** Mitigated by truncating outputs to 4000 characters.
*   **Position Bias:** Mitigated by running comparisons in both A/B and B/A orders and averaging.
*   **Verbosity/Poetic Incoherence Bias:** Addressed through specific judging criteria penalizing excessive or incoherent stylistic choices.

Biases **not** explicitly controlled for include potential judge self-bias, positivity/negativity bias, NSFW content aversion (smut bias), stylistic preferences, and "slop" bias (favoring overused tropes). Be mindful of these when interpreting results.

## Limitations

*   **Subjectivity:** Creative quality is subjective; the judge's assessment may differ from human preferences.
*   **Judge Limitations:** Sonnet 4.6 is good but not infallible; it may miss nuances humans perceive.
*   **Not a Roleplay Eval:** The benchmark doesn't assess conversational RP skills.
*   **English Only:** Currently evaluates English language writing only.
*   **Cost:** Running the benchmark involves API costs (approx. $10 per model using Sonnet 4.6 as judge).

**Always view benchmark scores as a guide, not absolute truth. Read the sample outputs!**

## Citation

If you use this benchmark in your work, please cite the repository:

```bibtex
@misc{creative-writing-bench-v3,
  author = {Samuel J Paech},
  title = {EQ-Bench Creative Writing Benchmark v3},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/EQ-bench/creative-writing-bench}}
}
```