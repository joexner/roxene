# Approach: Simultaneous Organism Cell Updates & MatMul Optimization

## Problem Analysis & Design
1. **Jacobi-style simultaneous dynamics**:
   - In `Organism.update()`, transmitter outputs for all cells are sampled upfront before updating neuron inputs so that all neurons update strictly based on state at step $t$ simultaneously.
   - For all neurons in an organism, weight matrices across input-to-hidden, feedback-to-hidden, hidden-to-feedback, and hidden-to-output are combined into block diagonal tensors:
     - `w1 = torch.block_diag(*w1_list)` (mapping concatenated input and feedback to hidden)
     - `w2 = torch.block_diag(*w2_list)` (mapping hidden to concatenated feedback and output)
     - `w_big = torch.block_diag(w1, w2)` (one giant unified block diagonal matrix)
   - The organism's neuron layers state is concatenated into `state = torch.cat([in_fb, hidden], dim=1)` and updated with **one giant matmul**:
     `res = torch.tanh(torch.matmul(state, w_big))`.
2. **Clean `Player.sync()` without intermediate ORM tracking**:
   - In `Player.sync()`, `self.organism.update()` is called directly without duplicating organism evaluation code or having mock checks in `players.py`.
   - Between updates, only `OUTPUT_READY` is checked.

## Plan to Eliminate TrackedTensor (Approach A) & Cell.update (Approach B)

### Step 1: Eliminate `TrackedTensor`
- Remove `TrackedTensor` from `src/roxene/persistence.py`.
- In `src/roxene/cells/neuron.py`, map tensor columns directly to `WrappedTensor` instead of `TrackedTensor.as_mutable(WrappedTensor)`.
- Update `WrappedTensor.process_bind_param` and `process_result_value` to accept and return standard `torch.Tensor` instances.
- Remove monkeypatching / pausing flags (`_tracking_enabled`) in `src/roxene/persistence.py` and `src/roxene/tic_tac_toe/players.py`.
- When persisting updated neuron states back to SQLAlchemy sessions (e.g. at trial end), flag modified attributes via `sqlalchemy.orm.attributes.flag_modified` or reassign updated tensors.

### Step 2: Eliminate Individual `Cell.update()` / `Neuron.update()`
- Remove `Cell.update()` and `Neuron.update()`.
- Centralize all cell and layer updates exclusively in `Organism.update()`, enforcing Jacobi-style simultaneous dynamics across all cells with one giant matmul.
- Update `tests/Neuron_test.py`:
  - Wrap standalone neuron update tests in single-neuron `Organism` instances (e.g. `org = Organism(); org.addNeuron(neuron); org.update()`).
  - Update tests checking math, input connections, and persistence accordingly.
- Update `tests/Organism_test.py`:
  - Adjust or remove `test_neuron_update_order` to reflect simultaneous Jacobi evaluation rather than legacy sequential ordering.

### Step 3: Verification & Validation
- Run full test suite using `uv run pytest`.
- Run JetBrains code inspection / linter across modified files to ensure clean types and no regressions.



