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

## Folded Matrix Routing and Matrix Operations
1. **Pre-folded Input Routing Matrix (`W_eff1`)**:
   - Port routing represents a static permutation matrix $P_{\text{route}}$ from cell outputs and recurrent feedbacks into neuron inputs: $S_{\text{in\_fb}} = C \cdot P_{\text{route}}$.
   - Since $H = \tanh(S_{\text{in\_fb}} \cdot W_1) = \tanh((C \cdot P_{\text{route}}) \cdot W_1) = \tanh(C \cdot (P_{\text{route}} \cdot W_1))$, the routing operation is pre-multiplied at compile time:
     $$W_{\text{eff1}} = P_{\text{route}} \cdot W_1$$
   - Hidden states are computed directly in a single matrix multiplication from state vector $C$:
     $$\text{hidden} = \tanh(C \cdot W_{\text{eff1}})$$
2. **Zero-Loop In-Place Recurrent Update**:
   - The recurrent block in $C$ (neuron feedbacks and outputs) directly matches the output layout of $W_2$.
   - Output and feedback activations are written in-place directly into $C$'s contiguous recurrent slice:
     $$\tanh(\text{hidden} \cdot W_2, \text{out}=C_{[:, \text{recurrent}]})$$
   - Individual neuron attributes (`n.feedback` and `n.output`) share direct views into $C$'s slices, automatically updating with zero loops and zero tensor copies.
   - Materializing `n.input` values via `P_input` is optional via `Organism.update(sync_neuron_inputs=False)`, and only synchronized when needed via `organism.sync_neuron_inputs()`. In `Player.sync()`, `sync_neuron_inputs=False` is passed across all internal updates and synced once in the `finally` block.

## Tensor Reuse and Matrix Caching
1. **Caching compiled matrices using organism ID (`self.id`)**:
   - The compiled weight matrices `W_eff1`, `w2`, `P_input`, and tensor state `C` are built once and stored on the organism instance (`self._compiled_execution`).
   - Because an organism's structure, cell connectivity, and `id` are immutable across its active lifetime, `self._compiled_execution` is bound directly to the organism instance (keyed by `self.id`).
   - If `addNeuron` is called, `self._compiled_execution` is cleared to re-build on the next update.
2. **Re-using the state buffer (`C`)**:
   - A pre-allocated contiguous state tensor buffer `C` is kept in place across successive update calls.
   - The hot path in `Player.sync()` runs in **just two pure matrix multiplications per tick**:
     ```python
     hidden = torch.tanh(torch.matmul(C, W_eff1))
     torch.tanh(torch.matmul(hidden, w2), out=C[:, num_inputs:num_inputs + rec_len])
     ```

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





