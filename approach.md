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
   - In `Neuron.update()`, the individual neuron layers are similarly combined and updated via a single unified matmul.
2. **Clean `Player.sync()` without intermediate ORM tracking**:
   - In `Player.sync()`, `TrackedTensor._tracking_enabled` is set to `False` while looping through updates.
   - `self.organism.update()` is called directly without duplicating organism evaluation code or having mock checks in `players.py`.
   - Between updates, only `OUTPUT_READY` is checked.
   - When sync completes or times out, `TrackedTensor._tracking_enabled = True` is restored in a `finally` block and the neuron tensors trigger change notifications.

## Validation
- Ran full test suite via `uv run pytest`. All 130 tests pass.
- Verified using JetBrains inspection linter on affected files.


