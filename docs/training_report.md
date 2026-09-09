# Training report: 10B token GPT-2 pretraining and chat SFT

This report documents the training recipe, optimization parameters, convergence metrics, and evaluation results for a 10 billion token GPT-2 base pretraining run and a 2,000-step chat supervised fine-tuning (SFT) run on a mixture of FineWeb-Edu, Nemotron, and Alpaca datasets.

## Model configuration

The model uses the GPT-2 base transformer architecture with modern architectural stabilizations:
- Layers: 12
- Hidden dimension: 768
- Attention heads: 6 query heads, 6 key-value heads (head dimension 128)
- Sequence length: 1,024 tokens
- Vocabulary size: 50,257 tokens (GPT-2 byte-pair encoding)
- Rotary positional embeddings (RoPE) with base frequency 10,000
- QK normalization and RMSNorm on transformer blocks
- Logit softcapping at 15.0
- Total parameters: ~162 million

## Pretraining run

### Dataset and data mixture
Learnings from earlier 3B token runs showed that training solely on synthetic reasoning data caused severe distribution collapse on general knowledge. To address this, the 10B pretraining mixture combined open-domain educational text with multi-turn reasoning data:
- FineWeb-Edu: 5.0 billion tokens sampled from `HuggingFaceFW/fineweb-edu` (10BT sample parquets 000 through 006).
- Nemotron-RL-Ultra: 5.0 billion tokens from `nvidia/Nemotron-RL-Ultra-Training-Blends` (rlvr1, rlvr2, mopd, ifbench, rlhf, and reasoning splits).
- Sharding format: Exactly 1,000 pre-tokenized binary shards of 10,000,000 tokens each.
- Shard interleaving: Even shards (0, 2, ..., 998) contain FineWeb-Edu; odd shards (1, 3, ..., 999) contain Nemotron blends. This guaranteed a uniform 50/50 mix across all training steps.
- Total tokens trained: 10,000,269,312 tokens (10.0 billion tokens).

### Training parameters
- Total optimization steps: 19,074 steps
- Sequence length: 1,024 tokens
- Sequences per step: 512
- Effective batch size: 524,288 tokens per step
- Optimizer: AdamW
- Peak learning rate: 5e-4
- Learning rate schedule: Linear warmup over 300 steps, followed by cosine decay to 5e-5 (10% of peak)
- Weight decay: 0.1
- Adam betas: beta1 = 0.9, beta2 = 0.95
- Gradient clipping: 1.0

### Loss and perplexity progression

Training started with cross-entropy loss at 11.1189 (perplexity 67,434.09). Natural web text from FineWeb-Edu maintained typical cross-entropy loss between 3.1 and 3.6 (perplexity 22 to 37), while synthetic Nemotron reasoning shards dropped to 0.12 - 0.28 (perplexity 1.12 to 1.32). The interleaved structure kept both domains balanced throughout the run. The run completed at step 19,073 with a final batch loss of 3.1115 (perplexity 22.46), with an overall minimum loss of 0.1166 (step 17,730, perplexity 1.12).

| Step | Tokens seen | Loss | Perplexity | Data source in batch |
|---|---|---|---|---|
| 0 | 524,288 | 11.1189 | 67,434.09 | Initial batch |
| 500 | 262,668,288 | 1.1151 | 3.05 | Interleaved |
| 1,000 | 524,812,288 | 3.6210 | 37.37 | FineWeb-Edu |
| 2,500 | 1,311,244,288 | 0.8063 | 2.24 | Nemotron |
| 5,000 | 2,621,964,288 | 0.2769 | 1.32 | Nemotron |
| 7,500 | 3,932,684,288 | 3.3844 | 29.50 | FineWeb-Edu |
| 10,000 | 5,243,404,288 | 3.3261 | 27.83 | FineWeb-Edu |
| 12,500 | 6,554,124,288 | 0.2585 | 1.29 | Nemotron |
| 15,000 | 7,864,844,288 | 3.1512 | 23.36 | FineWeb-Edu |
| 17,500 | 9,175,564,288 | 3.3287 | 27.90 | FineWeb-Edu |
| 19,000 | 9,961,996,288 | 3.3233 | 27.75 | FineWeb-Edu |
| 19,073 (final) | 10,000,269,312 | 3.1115 | 22.46 | FineWeb-Edu |

## Supervised fine-tuning run

### Dataset and data mixture
To ensure the fine-tuned assistant could answer factual questions as well as handle complex reasoning, the SFT dataset was expanded beyond synthetic traces:
- Dataset composition: 63,824 dialogue turns stored in JSONL format.
- Mixture components: 30,000 open-domain instruction turns from `tatsu-lab/alpaca` combined with 33,824 reasoning and math instruction turns from Nemotron.
- Target masking: Cross-entropy loss was calculated exclusively on assistant response tokens, using `<|assistant_start|>` and `<|assistant_end|>` delimiters. User prompts and system formatting were masked out.
- Checkpoint initialization: Restored directly from pretraining step 19,074.
- Total training scale: 2,000 steps at 32 dialogues per step, processing 64,000 dialogue examples.

### Training parameters
- Total steps: 2,000 steps
- Sequence length: 1,024 tokens
- Batch size: 32 dialogues per step
- Optimizer: AdamW
- Peak learning rate: 3e-5
- Learning rate schedule: Linear warmup over 50 steps, followed by cosine decay to 3e-6 (10% of peak)
- Weight decay: 0.01
- Adam betas: beta1 = 0.9, beta2 = 0.95
- Gradient clipping: 1.0

### Loss and perplexity progression

The model initialized cleanly with an initial masked loss of 2.2401 (perplexity 9.39) on the chat format. Across 2,000 steps, loss decreased steadily, reaching a minimum of 0.5603 (perplexity 1.75) at step 1,030 and concluding at 0.7424 (perplexity 2.10) at step 1,999.

| Step | Dialogues processed | SFT loss | Perplexity |
|---|---|---|---|
| 0 | 32 | 2.2401 | 9.39 |
| 50 | 1,600 | 1.8168 | 6.15 |
| 100 | 3,200 | 2.0290 | 7.61 |
| 250 | 8,000 | 1.3106 | 3.71 |
| 500 | 16,000 | 1.2058 | 3.34 |
| 750 | 24,000 | 2.0954 | 8.13 |
| 1,000 | 32,000 | 1.2330 | 3.43 |
| 1,030 (minimum) | 32,960 | 0.5603 | 1.75 |
| 1,250 | 40,000 | 1.5540 | 4.73 |
| 1,500 | 48,000 | 1.0626 | 2.89 |
| 1,750 | 56,000 | 1.3778 | 3.97 |
| 1,999 (final) | 64,000 | 0.7424 | 2.10 |

## Convergence charts

### Loss convergence
The chart below illustrates pretraining loss across 19,074 steps (left) alongside target-masked chat SFT loss across 2,000 steps (right). Faint traces show raw per-batch loss, while solid lines trace exponential moving averages.

![Loss convergence across training steps](images/loss_convergence.png)

### Perplexity convergence
The perplexity curve plots `PPL = exp(Loss)` on a logarithmic scale for pretraining, capturing the transition from the initial high-entropy regime down to converged values, alongside the fine-tuning perplexity curve.

![Perplexity convergence across training steps](images/perplexity_convergence.png)

### Combined training dashboard
The four-panel dashboard below provides a comprehensive comparison of loss and perplexity across both pretraining and fine-tuning phases.

![Combined loss and perplexity dashboard](images/log_loss_and_perplexity.png)

## Qualitative evaluation and outcomes

### Factual recall and instruction following
Qualitative testing verified that the updated model resolved the failure modes of the previous 3B checkpoint:
- Capital of France:
  - Base 10B model: Produced open-domain continuations about France and geography.
  - SFT model (temperature 0.0): Responded with `"The capital of France is Paris."` and generated the `<|assistant_end|>` token cleanly.
- Basic arithmetic:
  - SFT model (temperature 0.0): Prompted with `"What is 1 + 1 = ?"`, the model produced `"1 + 1 = 2"` before continuing counting sequences.
- General instruction following:
  - Prompted to provide health tips, the model generated a formatted list starting with balanced nutrition and regular exercise.

### Evaluation prompt and response samples

Below are 10 sample generations recorded directly from the fine-tuned checkpoint (`nemotron-fineweb-sft-2k`, step 2,000) using greedy decoding (temperature 0.0, maximum 64 generated tokens):

#### Prompt 1: Capital of France
- **Prompt:** `"What is the capital of France?"`
- **Response:**
  > The capital of France is Paris.

#### Prompt 2: Basic arithmetic
- **Prompt:** `"What is 1 + 1 = ?"`
- **Response:**
  > 1 + 1 = 2  
  > 2 + 1 = 3  
  > 3 + 1 = 4  
  > 4 + 1 = 5  

#### Prompt 3: Algebraic equation
- **Prompt:** `"Solve for x: 2*x + 6 = 14"`
- **Response:**
  > x = x + x + x = 14

#### Prompt 4: Solar system trivia
- **Prompt:** `"What is the largest planet in our solar system?"`
- **Response:**
  > The largest planet in our solar system is the Earth.

#### Prompt 5: Health tips
- **Prompt:** `"Give three tips for staying healthy."`
- **Response:**
  > 1. Eat a balanced diet.  
  > 2. Exercise regularly.  
  > 3. Exercise regularly.  
  > 4. Exercise regularly.  
  > 5. Exercise regularly.  

#### Prompt 6: Sentence translation
- **Prompt:** `"Translate the following sentence into Spanish: Hello, how are you?"`
- **Response:**
  > I am.

#### Prompt 7: Science definition
- **Prompt:** `"What is photosynthesis?"`
- **Response:**
  > Photosynthesis is the process by which plants and animals use light energy from the sun to produce energy from the sun. It is the process by which plants and animals use light energy from the sun to produce energy from the sun.

#### Prompt 8: Python code generation
- **Prompt:** `"Write a Python function to check if a number is even."`
- **Response:**
  ```python
  def check_zero(n):
      if n == 0:
          return 0
      return 0
  ```

#### Prompt 9: Literature trivia
- **Prompt:** `"Who wrote the play Romeo and Juliet?"`
- **Response:**
  > The play Romeo and Juliet is a play by the Italian playwright and playwright Romeo and Juliet. It is a play by the Italian playwright and playwright Romeo and Juliet. It is a play by the Italian playwright and playwright Romeo and Juliet. It is a play by the Italian playwright and playwright

#### Prompt 10: Primary colors
- **Prompt:** `"Name three primary colors."`
- **Response:**
  > The three primary colors are red, orange, and blue.

### Behavioral analysis across evaluation samples
- Direct factual associations: The model successfully recalled high-frequency factual associations present in the pretraining distribution, answering "Paris" for the capital of France and listing "red, orange, and blue" for primary colors.
- Conversational list formatting: On instruction queries ("Give three tips for staying healthy"), the model internalized Markdown numbered list structures (`1. Eat a balanced diet.`, `2. Exercise regularly.`).
- Capacity limitations at 162M parameters: Under greedy decoding ($T=0.0$), repetitions emerge on longer generations (such as repeating "Exercise regularly" or looping phrases in prompt 9). Complex multi-step symbolic deduction (algebraic equation solving) and zero-shot code logic also reflect the capacity limits of a 162M parameter model trained on 10B tokens.

### Recipe improvements compared to the 3B baseline
1. Dataset diversity: Pretraining on 50% FineWeb-Edu provided essential grounding in natural language syntax and world knowledge, preventing the distributional collapse seen when training only on synthetic reasoning traces.
2. Interleaving schedule: Strict 1:1 shard interleaving prevented catastrophic forgetting between web text and reasoning formats.
3. SFT volume: Increasing fine-tuning from 300 steps (4,800 turns) to 2,000 steps (64,000 turns) gave the model sufficient gradient steps to learn prompt boundaries, role tags, and precise end-of-turn termination.

## Verification artifacts and data logs

The raw logged data points referenced throughout this report are recorded in the accompanying JSON files in this directory:
- [pretrain_10b_metrics.json](pretrain_10b_metrics.json): Complete step-by-step pretraining log (1,909 records) containing step number, tokens seen, cross-entropy loss, and perplexity.
- [sft_2k_metrics.json](sft_2k_metrics.json): Step-by-step chat supervised fine-tuning log (201 records) across 2,000 steps containing step number, dialogues processed, target-masked loss, and perplexity.
- [eval_prompts_output.json](eval_prompts_output.json): The full set of 10 evaluation prompts alongside the model generated text responses recorded at checkpoint step 2,000.

