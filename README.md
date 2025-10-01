# Roxene

A genetic algorithm project for evolving neural networks to play tic-tac-toe.

## Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer

## Installation

### Install uv

If you don't have uv installed, install it using pip:

```bash
pip install uv
```

Or using the official installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install Dependencies

Install the project and all its dependencies:

```bash
uv sync --all-extras
```

This will:
- Create a virtual environment in `.venv`
- Install all runtime dependencies
- Install all development dependencies (pytest, notebook, matplotlib, etc.)

If you only want runtime dependencies (without dev tools):

```bash
uv sync
```

## Running Tests

Run the test suite:

```bash
uv run python -m pytest
```

Run tests with verbose output:

```bash
uv run python -m pytest -v
```

Run a specific test file:

```bash
uv run python -m pytest tests/tic_tac_toe/Trial_test.py
```

## Running the Application

Run the tic-tac-toe training:

```bash
uv run python -m roxene.tic_tac_toe <pool_size> <num_trials>
```

For example:

```bash
uv run python -m roxene.tic_tac_toe 100 1000
```

## Development

### Adding Dependencies

To add a runtime dependency:

```bash
uv add <package-name>
```

To add a development dependency:

```bash
uv add --dev <package-name>
```

### Activating the Virtual Environment

While `uv run` can execute commands without activation, you can also activate the virtual environment:

```bash
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

### Jupyter Notebooks

Start Jupyter notebook server:

```bash
uv run jupyter notebook
```

The notebooks are located in the `notebooks/` directory.

## Project Structure

- `src/roxene/` - Main package source code
  - `tic_tac_toe/` - Tic-tac-toe game implementation
  - `cells/` - Neural network cell implementations
  - `genes/` - Genetic algorithm components
  - `mutagens/` - Mutation operators
- `tests/` - Test suite
- `notebooks/` - Jupyter notebooks for experimentation

## Key Features

- Genetic algorithm-based learning
- Neural network organisms that evolve to play tic-tac-toe
- SQLAlchemy-based persistence for tracking evolutionary progress
- Support for custom mutagens and genes
- Comprehensive test suite

## License

See LICENSE file for details.
