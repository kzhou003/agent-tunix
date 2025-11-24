# Agent-Tunix Documentation

Complete Sphinx documentation for the Agent-Tunix GRPO training framework. Comprehensive guides covering installation, configuration, training, evaluation, and advanced topics.

## Documentation Structure

### Getting Started
- **Installation** (`docs/getting_started/installation.rst`) - System requirements, CUDA setup, installation methods, troubleshooting
- **Quick Start** (`docs/getting_started/quick_start.rst`) - First training run, basic configuration, evaluation examples
- **Configuration** (`docs/getting_started/configuration.rst`) - YAML-based configuration system, Hydra framework, override patterns

### User Guides
- **Training** (`docs/guide/training.rst`) - GRPO training process, monitoring, memory optimization, distributed training, checkpointing
- **Evaluation** (`docs/guide/evaluation.rst`) - Model evaluation, metrics, inference strategies, checkpoint selection
- **Hyperparameter Tuning** (`docs/guide/hyperparameter_tuning.rst`) - Tuning strategies, parameter sweeps, monitoring, best practices
- **Experiments** (`docs/guide/experiments.rst`) - Creating and using experiment presets for reproducibility

### Configuration Reference
- **Overview** (`docs/config/overview.rst`) - Configuration system architecture, composition, Hydra concepts
- **Model Configuration** (`docs/config/model.rst`) - Available models (270M, 1B, 4B), LoRA settings, mesh shapes, memory requirements
- **Optimizer Configuration** (`docs/config/optimizer.rst`) - AdamW optimizer, learning rate, warmup, weight decay, tuning strategies
- **Training Configuration** (`docs/config/training.rst`) - Batch sizes, checkpointing, evaluation intervals, seed management

### API Reference
- **Training API** (`docs/api/train.rst`) - `train()` function, configuration classes, GRPO algorithm details, example usage
- **Evaluation API** (`docs/api/evaluate.rst`) - `evaluate()` function, inference configs, metrics, checkpoint selection
- **Models API** (`docs/api/models.rst`) - Model architectures, LoRA configuration, distributed training setup
- **Data API** (`docs/api/data.rst`) - Data loading, preprocessing, tokenization, custom datasets
- **Rewards API** (`docs/api/rewards.rst`) - Reward functions, built-in rewards, custom reward design

### Advanced Topics
- **Distributed Training** (`docs/advanced/distributed_training.rst`) - Multi-GPU setup, FSDP, tensor parallelism, multi-node training
- **Custom Rewards** (`docs/advanced/custom_rewards.rst`) - Designing custom reward functions, reward shaping, curriculum learning, debugging
- **Troubleshooting** (`docs/advanced/troubleshooting.rst`) - Common issues and solutions, debugging tips, error diagnosis

### References
- **FAQ** (`docs/references/faq.rst`) - Common questions covering setup, configuration, training, evaluation, hyperparameter tuning
- **Glossary** (`docs/references/glossary.rst`) - 100+ terms including GRPO, LoRA, FSDP, tensor parallelism, optimization concepts

## Building the Documentation

### Install Dependencies

Using `uv`:

```bash
uv pip install -e ".[docs]"
```

Or using `pip`:

```bash
pip install -e ".[docs]"
```

The documentation dependencies are defined in `pyproject.toml` under `[project.optional-dependencies.docs]`.

### Build HTML Documentation

```bash
cd docs
make html
```

Output will be in `docs/_build/html/`.

### View Documentation

```bash
# Open in browser
open docs/_build/html/index.html

# Or serve locally
cd docs
make serve
# Then visit http://localhost:8000
```

### Other Build Targets

```bash
make clean      # Remove build artifacts
make pdf        # Build PDF (requires LaTeX)
make epub       # Build EPUB ebook
make man        # Build man pages
make text       # Build plain text
```

See `docs/BUILD.md` for complete build instructions.

## Documentation Features

### Comprehensive Coverage

- 22 documentation pages covering all aspects
- ~15,000+ lines of documentation
- 100+ code examples and command references
- Complete API documentation with practical examples

### Well-Organized

- Clear hierarchical structure (Getting Started → Guides → Advanced)
- Consistent formatting with reStructuredText
- Cross-references between related topics
- Quick navigation with toctree

### Practical Examples

- Real command-line examples for all features
- Configuration examples for different scenarios
- Troubleshooting with solutions
- Step-by-step workflows

### Complete References

- Detailed parameter documentation
- Configuration option explanations
- API function references
- Glossary with 100+ ML/training terms

## Key Topics Covered

### Configuration System
- Hydra configuration framework
- YAML-based configuration structure
- Configuration composition and overrides
- Nested parameter overrides with dot notation
- Experiment presets for reproducibility

### Training
- GRPO algorithm explanation
- Training process overview
- LoRA (Low-Rank Adaptation) details
- Gradient clipping and optimization
- Checkpoint management
- Monitoring with Weights & Biases and TensorBoard

### Evaluation
- Evaluation metrics (accuracy, partial accuracy, format accuracy)
- Multiple inference strategies (greedy, standard, liberal)
- Temperature and sampling parameters
- Checkpoint evaluation and comparison
- Multiple-pass evaluation for uncertainty

### Hyperparameter Tuning
- Learning rate search strategies
- Batch size tuning
- Model capacity adjustment
- LoRA rank tuning
- Parameter sweep patterns
- Grid search and random search

### Distributed Training
- Data Parallelism (FSDP)
- Tensor Parallelism (TP)
- Hybrid parallelism strategies
- Multi-node setup
- Communication optimization
- Performance monitoring

### Advanced Customization
- Custom reward function design
- Reward shaping techniques
- Curriculum learning
- Custom data loaders
- Multi-aspect evaluation
- Reward normalization

## Documentation Files

### Documentation Pages (22 files)
- `docs/index.rst` - Main index
- `docs/getting_started/` - 3 files
- `docs/guide/` - 4 files
- `docs/config/` - 4 files
- `docs/api/` - 5 files
- `docs/advanced/` - 3 files
- `docs/references/` - 2 files

### Configuration Files
- `docs/conf.py` - Sphinx configuration with autodoc, napoleon, RTD theme
- `docs/Makefile` - Build automation (HTML, PDF, EPUB, etc.)
- `docs/BUILD.md` - Build instructions
- `requirements-docs.txt` - Python dependencies

### Supporting Directories
- `docs/_static/` - Static assets (CSS, JavaScript, images)
- `docs/_templates/` - Custom Sphinx templates
- `docs/_build/` - Generated documentation (created by build)

## Integration Points

### Sphinx Extensions Used
- **autodoc** - Auto-extract from Python docstrings
- **napoleon** - Google-style docstring support
- **intersphinx** - Cross-reference external docs (Python, NumPy, JAX)
- **viewcode** - Link to source code from API docs

### Theme
- **sphinx_rtd_theme** - Read the Docs theme for professional appearance

### Hosting Options
- Local HTML (`make html` → `_build/html/`)
- Read the Docs (automatic builds on GitHub push)
- Static hosting (GitHub Pages, etc.)

## Usage Workflows

### For New Users
1. Read Quick Start → Get first training running
2. Read Configuration → Understand parameter system
3. Read Training Guide → Learn training process
4. Experiment with examples

### For Configuration
1. Read Configuration Overview → Understand system
2. Read specific config guides (Model, Optimizer, Training)
3. Review examples for different GPU/scale scenarios
4. Use `--cfg job` to preview before training

### For Troubleshooting
1. Search FAQ for common questions
2. Check Troubleshooting guide for specific issues
3. Review error messages with solution suggestions
4. Check Glossary for terminology

### For Advanced Work
1. Read Distributed Training for multi-GPU setup
2. Read Custom Rewards for reward function design
3. Review API reference for implementation details
4. Check examples and patterns section

## Quick References in Documentation

### Configuration Examples
- Memory-constrained setups (11GB GPU)
- Balanced setups (48GB GPU)
- High-performance setups (80GB GPU)
- Multi-GPU distributed training

### Training Examples
- Quick test (10 steps)
- Single GPU training
- Ablation studies
- Production runs

### Evaluation Examples
- Latest checkpoint evaluation
- Specific checkpoint evaluation
- Multiple inference strategies
- Batch evaluation workflows

### Tuning Examples
- Learning rate search
- Batch size tuning
- Model size comparison
- Parameter sweeps

## Maintenance

### Updating Documentation
1. Edit `.rst` files in appropriate directories
2. Run `make clean && make html` to rebuild
3. Review in `_build/html/` before committing
4. Commit changes to documentation files

### Adding New Pages
1. Create new `.rst` file in appropriate directory
2. Add entry to parent `index.rst` or relevant `toctree`
3. Build with `make html`
4. Verify in browser

### Updating Configuration Examples
- Keep examples synchronized with actual configuration files
- Test command examples before documenting
- Update when configuration changes

## Documentation Statistics

- **Total Documentation Pages**: 22
- **Total Lines**: 15,000+
- **Configuration Examples**: 50+
- **Command Examples**: 100+
- **API Reference Functions**: 10+
- **Glossary Terms**: 100+
- **Cross-References**: 200+

## Next Steps

1. **Install dependencies**: `pip install -r requirements-docs.txt`
2. **Build HTML**: `cd docs && make html`
3. **View documentation**: `open docs/_build/html/index.html`
4. **Deploy**: Push to Read the Docs or static hosting

## Support

For documentation issues:
1. Check FAQ section in documentation
2. Search Troubleshooting guide
3. Review Glossary for terminology
4. Check GitHub issues for known problems

## License

Documentation is included with Agent-Tunix project license.
