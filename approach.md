# Approach: Simultaneous Organism Cell Updates & MatMul Optimization

## Problem Analysis & Design
1. **Jacobi-style simultaneous dynamics**:
   - In `Organism.update()`, transmitter outputs for all cells are sampled upfront before updating neuron inputs so that all neurons update strictly based on state at step $t$ simultaneously.
   - For all neurons in an organism, weight matrices are combined into block diagonal tensors:
     - `w1 = torch.block_diag(*w1_list)` (mapping concatenated input and feedback to hidden)
     - `w2 = torch.block_diag(*w2_list)` (mapping hidden to concatenated feedback and output)
   - On each tick:
     1. All hiddens are computed from inputs and feedback in one batched matmul: `hidden = torch.tanh(torch.matmul(in_fb, w1))`.
     2. All feedbacks and outputs are updated from the new hiddens in a second batched matmul: `res = torch.tanh(torch.matmul(hidden, w2))`.
   - No hidden state carries over across update ticks or trials—hidden layers are strictly intermediate activations calculated from `[input, feedback]` within the tick.
   - Persistent state in memory and database remains strictly `input`, `feedback`, and `output`.

2. **Clean `Player.sync()` without intermediate ORM tracking**:
   - In `Player.sync()`, `self.organism.update()` is called directly without duplicating organism evaluation code or having mock checks in `players.py`.
   - Between updates, only `OUTPUT_READY` is checked.

## Tensor Reuse and Matrix Caching
1. **Caching compiled matrices using organism ID (`self.id`)**:
   - The compiled weight matrices `w1`, `w2`, and the buffer index slices are built once and stored on the organism instance (`self._compiled_execution`).
   - Because an organism's structure, cell connectivity, and `id` are immutable across its active lifetime, `self._compiled_execution` is bound directly to the organism instance (keyed by `self.id`).
   - If `addNeuron` is called, `self._compiled_execution` is cleared to re-build on the next update.
2. **Re-using the state buffer (`in_fb`)**:
   - A pre-allocated contiguous `in_fb` tensor buffer `[1, total_in_fb]` is kept in place across successive update calls.
   - In each update, input ports write directly into `in_fb`'s input slices.
   - Feedback slices from `res` are copied directly back into `in_fb`'s feedback positions for the next step, avoiding tensor re-allocation and concatenation on every tick.

## Plan to Eliminate TrackedTensor (Approach A) & Cell.update (Approach B)

### Step 1: Eliminate `TrackedTensor`
- Removed `TrackedTensor` from `src/roxene/persistence.py`.
- In `src/roxene/cells/neuron.py`, mapped tensor columns directly to `WrappedTensor` instead of `TrackedTensor.as_mutable(WrappedTensor)`.
- Updated `WrappedTensor.process_bind_param` and `process_result_value` to accept and return standard `torch.Tensor` instances.
- Removed monkeypatching / pausing flags (`_tracking_enabled`) in `src/roxene/persistence.py` and `src/roxene/tic_tac_toe/players.py`.

### Step 2: Eliminate Individual `Cell.update()` / `Neuron.update()`
- Removed `Cell.update()` and `Neuron.update()`.
- Centralized all cell and layer updates exclusively in `Organism.update()`, enforcing Jacobi-style simultaneous dynamics across all cells with two clean batched matmuls per tick.
- Updated `tests/Neuron_test.py`:
  - Wrapped standalone neuron update tests in single-neuron `Organism` instances.
  - Updated tests checking math, input connections, and persistence accordingly.
- Updated `tests/Organism_test.py`:
  - Replaced legacy sequential order test with `test_simultaneous_update`.

### Step 3: Verification & Validation
- Ran full test suite using `uv run pytest`. All 130 tests pass.
- Verified using JetBrains code inspection / linter across modified files to ensure clean types and no regressions.





