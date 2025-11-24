Glossary
========

**Accuracy**
    Percentage of exact matches between model outputs and expected answers.

**Activation**
    Output of a neural network layer; intermediate representation passed to next layer.

**Adaptation**
    Process of modifying a pre-trained model for a specific task. LoRA is an adaptation technique.

**AdamW**
    Adaptive Moment Estimation optimizer with decoupled weight decay. Standard optimizer used in Agent-Tunix.

**Attention Mask**
    Binary mask indicating which tokens should be attended to (1) and which are padding (0).

**Baseline**
    Reference value used in reward computation to normalize/center rewards.

**Batch Size**
    Number of samples processed together in one training step. Larger batches = more stable gradients.

**Beam Search**
    Decoding strategy that tracks multiple hypothesis sequences and keeps the best ones.

**Benchmark**
    Set of standard problems used to evaluate model performance.

**Beta (β)**
    In GRPO, weight controlling KL divergence penalty. Higher β keeps model closer to reference.

**Bias (in neural networks)**
    Learnable parameters added to layer outputs; enables modeling non-linear relationships.

**Bias (statistical)**
    Systematic errors in model predictions; different from variance.

**Checkpoint**
    Saved model weights at a training step, allowing resumption and model selection.

**Clipping (Gradient)**
    Limiting gradient magnitudes to prevent exploding gradients during backpropagation.

**Cluster Config**
    Configuration specifying how multiple GPUs/nodes are arranged for training.

**Computation Graph**
    DAG (directed acyclic graph) representing mathematical operations and their dependencies.

**Conditioning**
    Process of providing context to influence model output; e.g., prompt conditioning.

**Configuration (Hydra)**
    YAML-based specification of training parameters, model settings, and experiment details.

**Convergence**
    Training state where loss stops decreasing, model has reached a local optimum.

**Cross-Entropy Loss**
    Standard loss function for classification/language modeling tasks.

**CUDA**
    Compute Unified Device Architecture; NVIDIA's parallel computing platform for GPUs.

**Curriculum Learning**
    Training strategy starting with easy examples, progressing to harder ones.

**Data Parallelism**
    Distributing different data batches across multiple devices while replicating the model.

**Decoding**
    Process of generating text from model logits (scores) using sampling or greedy selection.

**Divergence**
    When training loss increases over time; indicates learning rate too high or data issue.

**Dropout**
    Regularization technique randomly disabling neurons during training to prevent overfitting.

**FSDP**
    Fully Sharded Data Parallel; JAX distributed training strategy sharding model and data.

**Embedding**
    Vector representation of discrete tokens or concepts learned during training.

**Entropy**
    Measure of randomness/uncertainty in a probability distribution.

**Epsilon (ε)**
    In PPO/GRPO, clipping range for policy updates; controls maximum gradient step.

**Epoch**
    One complete pass through the entire training dataset.

**Evaluation**
    Process of assessing model performance on held-out test data using metrics.

**Example (training)**
    Single data point consisting of input prompt and target output.

**Fine-tuning**
    Training a pre-trained model on task-specific data; adapts general knowledge to specific task.

**Flax**
    Neural network library for JAX providing layer abstractions and utilities.

**Forward Pass**
    Computing network output given inputs; first stage of training step.

**Frozen Weights**
    Model parameters that are not updated during training; held constant as reference.

**Generation (text)**
    Process of producing new text sequences conditioned on prompt input.

**Gradient**
    Direction and magnitude of loss change with respect to parameters; used to update weights.

**Gradient Accumulation**
    Computing gradients over multiple mini-batches before updating weights; simulates larger batch.

**Gradient Descent**
    Optimization algorithm updating parameters by moving in negative gradient direction.

**Greedy Decoding**
    Selecting highest-probability token at each step; deterministic, fast generation.

**Group Relative Policy Optimization (GRPO)**
    Reinforcement learning algorithm generating K responses per prompt and computing group-relative rewards.

**Hallucination**
    Model generating plausible-sounding but false information not supported by training data.

**Hyperparameter**
    Configuration setting controlling training dynamics (learning rate, batch size, etc.); not learned.

**Hydra**
    Configuration management framework enabling YAML-based parametrization and composition.

**Input IDs**
    Numeric token indices representing text input to neural network.

**Inference**
    Using trained model to generate predictions on new data.

**Interpolation (Hydra)**
    Referencing other config values using ${path.to.value} syntax.

**JAX**
    Array computation library from Google enabling GPU-accelerated numerical computing.

**KL Divergence**
    Measure of distance between two probability distributions; used to constrain policy deviation.

**Layer**
    Distinct processing unit in neural network; applies transformation to input.

**Learning Rate**
    Hyperparameter controlling step size in gradient descent; critical for training stability.

**Learning Rate Scheduler**
    Strategy for adjusting learning rate during training (warmup, cosine decay, etc.).

**Log (training)**
    Record of metrics (loss, accuracy, etc.) computed during training for monitoring progress.

**Logits**
    Raw, unnormalized output scores from neural network before softmax/sampling.

**LoRA**
    Low-Rank Adaptation; parameter-efficient fine-tuning adding small trainable matrices to frozen model.

**LoRA Rank**
    Dimension of low-rank matrices in LoRA; higher rank = more capacity but more parameters.

**Loss Function**
    Mathematical function quantifying difference between model predictions and targets; guided by gradients.

**Mask (in attention)**
    Binary indicator controlling which tokens interact; prevents attending to future tokens (causal mask).

**Memory (GPU)**
    High-speed storage on GPU holding model weights, activations, and gradients; limited resource.

**Mesh Shape**
    Configuration specifying how devices arranged for distributed training (FSDP × TP dimensions).

**Metric**
    Quantitative measure of model performance (accuracy, loss, F1, etc.).

**Mini-batch**
    Small subset of data processed together; typical size 1-256 examples.

**Mixed Precision**
    Training using both float32 (high precision) and float16 (lower precision) for speed/memory trade-off.

**Model**
    Neural network architecture with learnable parameters (weights, biases, embeddings, etc.).

**Model Family**
    Category of architectures (Gemma3, LLaMA, etc.); defines structure and behavior.

**Momentum**
    Accumulation of previous gradients; helps optimization converge faster and escape local minima.

**Multi-run**
    Running same experiment multiple times with different hyperparameter values (parameter sweep).

**NaN (Not a Number)**
    Invalid floating-point value indicating computation failure; training becomes undefined.

**Normalization**
    Rescaling values to standard range (usually 0-1 or mean 0, std 1) for stable training.

**Nucleus Sampling (Top-p)**
    Decoding strategy selecting from highest-probability tokens summing to threshold p.

**Optimizer**
    Algorithm updating model weights based on gradients (Adam, SGD, AdamW, etc.).

**Overrides (configuration)**
    Command-line changes to config parameters without modifying YAML files.

**Parameter Sharing**
    Reusing same weights across multiple positions/layers to reduce memory and improve efficiency.

**Perplexity**
    Inverse probability of ground truth sequence; lower is better for language models.

**Policy**
    Model trained using reinforcement learning to maximize expected reward.

**PPO (Proximal Policy Optimization)**
    Reinforcement learning algorithm with clipped objective preventing large policy updates.

**Prompt**
    Input text conditioning model output; text input to language model.

**Prompt Engineering**
    Designing effective prompts to elicit desired model behavior.

**Pruning**
    Removing small-weight connections from neural network to reduce size/computation.

**Quantization**
    Reducing precision of weights/activations (float32 → int8) to save memory.

**Rank (LoRA)**
    See LoRA Rank.

**Recall**
    Fraction of positive examples correctly identified; useful for imbalanced problems.

**Reference Model**
    Original frozen model used as baseline; policy model trained relative to reference.

**Regularization**
    Technique preventing overfitting by penalizing complex models (dropout, L2, etc.).

**Reinforcement Learning (RL)**
    Learning paradigm where agent optimizes behavior to maximize cumulative reward signal.

**Reward Function**
    Function evaluating model responses and returning numerical score guiding training.

**Reward Shaping**
    Adding intermediate signals to guide learning beyond primary reward.

**Sampling (decoding)**
    Stochastic generation selecting tokens from probability distribution.

**Scheduler (learning rate)**
    Strategy for adjusting learning rate during training for better convergence.

**Seed (random)**
    Initial value for random number generator; same seed = reproducible randomness.

**Softmax**
    Normalization function converting logits to probability distribution.

**Stable Training**
    Training where loss smoothly decreases without spikes, divergence, or NaN errors.

**Step (training)**
    Single gradient update; one mini-batch processed and weights updated.

**Temperature (sampling)**
    Parameter controlling randomness of decoding (0 = deterministic, ∞ = uniform random).

**Tensor Parallel**
    Distributing model tensors across multiple devices; suits very large models.

**Tokenization**
    Process of converting text into token indices; inverse is detokenization.

**Token**
    Discrete unit of text (word, subword, character); basic unit of language models.

**Top-k Sampling**
    Decoding selecting from k highest-probability tokens.

**Training**
    Process of updating model parameters to minimize loss on training data.

**Validation**
    Evaluating model on held-out data to monitor generalization during training.

**Warmup (learning rate)**
    Initial training phase with gradually increasing learning rate; improves stability.

**Warmup Ratio**
    Fraction of training devoted to warmup phase (typical 0.05-0.1).

**Weights**
    Learnable parameters of neural network; updated during training via gradients.

**Weight Decay**
    Regularization penalizing large weights; encourages sparse solutions.

**Weights & Biases (W&B)**
    Platform for tracking, visualizing, and comparing machine learning experiments.

**Zero-shot**
    Model performing task without seeing examples; relies on pre-training knowledge.

Next Steps
----------

- :doc:`../guide/training` - Training guide
- :doc:`faq` - Frequently asked questions
- :doc:`../getting_started/configuration` - Configuration reference
