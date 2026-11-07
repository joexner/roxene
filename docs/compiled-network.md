# The CompiledNetwork architecture

How organism state, live computation, and persistence are structured — and why
the old class of device/dtype persistence bugs can no longer occur.

## The problem this replaces

Before this change, `Neuron.input/feedback/output` were **chameleoning
attributes**: the same attribute held different kinds of object depending on
what had happened to the organism so far.

- freshly built or reloaded: a standalone `torch.Tensor` on `DEVICE`
  (`TORCH_PRECISION`, float16);
- after the first `update()`: a **view** into the compiled `C`/`all_inputs`
  torch buffers;
- after the numpy fast path engaged: a **view** into float32 numpy copies of
  those buffers.

Persistence (`WrappedTensor`) pickled *whichever alias was active at flush
time*. Because the aliases had different devices and dtypes, the database
became a grab-bag: some neurons stored GPU tensors, some CPU float16 tensors,
some float32 numpy arrays — all under one column type. Loading only moved
tensors *to* device, never *back* to `TORCH_PRECISION`, so a single
numpy-backend trial could permanently contaminate an organism, and a worker
could die mid-trial leaving poisoned state behind. The reaper then closed the
trial, the organism stayed broken, and its next worker crashed the same way.

## The new architecture

The core change is a hard separation between **canonical state** (what is
persisted) and **live state** (what is being computed).

```
            ┌──────────────────────────────────────────────┐
            │                Organism                      │
            │  cells: [InputCell..., Neuron...]            │
            │                                              │
            │  Neuron.input/feedback/output/weights        │  ← canonical state:
            │      plain np.float16 ndarrays, never        │    plain, small,
            │      rebound, never torch                    │    dialect-agnostic
            │                                              │
            │  _compiled_network: CompiledNetwork | None   │
            │      owns C, all_inputs, W_eff1, w2, P_input │  ← live state: one
            │      (torch on DEVICE, or fp16 numpy)        │    owner, one backend,
            └──────────────┬───────────────────────────────┘    chosen at compile
                           │ commit_state()                     time, never rebinds
                           ▼
                  write live state back into the plain arrays
                           │
                           ▼
                  SQLAlchemy flush (FloatArrayType: ndarray <-> JSON)
```

**Canonical state.** `Neuron` columns are plain ndarrays at the working precision
(`src/roxene/cells/neuron.py`). They are set in `__init__` (a copy, so gene
arrays and neuron state never alias each other), loaded from the database,
and written back by exactly one operation: `commit_state()`. Nothing in the
runtime ever rebinds them to buffer views.

**Live state.** `CompiledNetwork` (`src/roxene/compiled_network.py`) is the
torch backend and the default; `NumpyCompiledNetwork` subclasses it as the
numpy fast path for small organisms. `Organism.compile()` picks the
backend once per organism, at construction (torch by default, numpy when
enabled via `ROXENE_NUMPY_BACKEND=on`/`auto` and the organism is under
`ROXENE_NUMPY_BACKEND_MAX_FLOPS`), and never switches or rebinds mid-flight.

## Compilation

`CompiledNetwork` pre-multiplies the per-neuron structure into
matrices so each `update()` is a handful of matmuls:

- `C` — one contiguous row, laid out as
  `[input cell values | per-neuron (feedback…, output) | unconnected input ports]`.
  Gathered once from the plain arrays (and `InputCell.value`).
- `P_route` — routes `C` to each neuron's `(input ports, feedback)` vector.
- `w1` / `w2` — block-diagonal weight stacks
  (`[input_hidden; feedback_hidden]` and `[hidden_feedback | hidden_output]`).
- `W_eff1 = P_route @ w1` — the effective input→hidden matrix.
- `P_input` — routes `C` to the flat input-port vector.
- `all_inputs` — the routed input values (the source of every neuron's
  `input` at commit time).
- `_state_slices` — where each neuron's `feedback`/`output`/`input` live in
  `C`/`all_inputs`, used by `commit_state()`.

The math is a verbatim port of the old inline `update()`; only the ownership
changed.

## The trial hot loop

Per move, the players call (unchanged API, `src/roxene/tic_tac_toe/players.py`):

```python
for _ in range(updates):
    organism.update(sync_neuron_inputs=False)   # C[rec] <- tanh(tanh(C W1) W2)
organism.set_input(...)                          # writes InputCell.value AND C
organism.get_output(...)                         # reads C (live)
organism.sync_neuron_inputs()                    # all_inputs <- C @ P_input
```

`update()` with `sync_neuron_inputs=False` skips the (larger) `C @ P_input`
matmul; inputs are routed once per move, not once per update. On the torch
backend every op is an in-place device op — no host round-trips per update.
On the numpy backend it is plain CPU matmul, as before.

`set_input`/`get_output` on `Organism` route through the network when one is
live (`Organism.network()` compiles lazily on first use), and fall back to
the plain state otherwise. So reading an organism's output always sees live
values during a trial, without touching the neurons.

## Committing: the single sync point

`CompiledNetwork.commit_state()` copies the live state back into the plain
arrays — `n.feedback`/`n.output` from `C`, `n.input` from `all_inputs` —
using `.copy()` on the numpy backend and
`.detach().cpu().numpy().astype(float16)` on the torch backend. After a
commit the neurons are authoritative and the network may be dropped or
rebuilt from the cells: the network is a pure performance transform.

Commits happen at exactly three places:

1. **`Environment.complete_trial()`** — before `session.merge(trial)`.
   `merge` copies the worker's *detached* organism (the only object that
   holds the live network) into a fresh in-session instance, so the plain
   arrays must be current before the copy. This is the durable end-of-trial
   write.
2. **`before_flush` hook** (`src/roxene/organism.py`) — a Session-level hook
   commits any organism in `session.new`/`session.dirty` before its state is
   serialized. This is a structural guarantee, not a convention: no flush
   can ever persist stale plain arrays while a live network exists on the
   same object.
3. **`Organism.addNeuron()`** — invalidates the network (structural change);
   the next `update()` recompiles from the plain arrays. (Gene execution that
   reorders cells only ever runs on fresh, never-compiled organisms.)

Between commits the plain arrays may be stale — that is intentional: it is
what keeps the hot loop free of device syncs. `Neuron.get_output()` reads the
plain arrays (authoritative after a commit); `Organism.get_output()` reads
the live network. Tests that assert on post-update state call
`organism.commit_state()` first, which documents the contract.

## Persistence: torch is gone from the layer

`FloatArrayType` (`src/roxene/persistence.py`) is a `TypeDecorator` over
`JSON`: in memory an `np.float16` ndarray, on the wire a JSON array (of
arrays, for the 2-D weight matrices). One conversion, no device, no dtype
negotiation, and it works on both PostgreSQL (production) and SQLite
(tests). `WrappedTensor` and its pickle path are deleted.

Consequences:

- One in-memory representation of neuron state (an ndarray at the working
  precision) and one wire representation (JSON numbers): no alias, no view, no
  device, and nothing whose identity or dtype depends on execution history.
- State survives restarts, image changes, and Python/torch upgrades without
  unpickle concerns; the `neuron` table has no `PickleType` columns left.

## Precision

A single precision — `NP_PRECISION` / `TORCH_PRECISION` (currently fp16) —
covers gene payloads, neuron state, the compiled buffers, and the JSON wire
format. The one unbounded RNG source, `wiggle`, is clamped to the precision's
finite range before it becomes a gene; `random_slice` is bounded to [-1, 1].

## Migration

Existing rows hold pickled torch/numpy blobs in the old columns; the new code
expects JSON. The organism pool is ephemeral research data (it is
continuously bred and culled), so the supported migration is a reinitialize:

```bash
kubectl -n roxene scale statefulset worker worker-cpu --replicas=0
kubectl -n roxene delete job init --ignore-not-found
kubectl -n roxene delete pvc data-postgres-0
kubectl -n roxene delete statefulset postgres
kubectl -n roxene apply -k k8s/overlays/enzo.local   # fresh PVC, DB, init job
```

A one-shot backfill (unpickle old rows → convert to the JSON form) is possible
but not done here; the pool regenerates within hours of breeding.

## Performance notes

- **Torch backend (large organisms):** in-place `torch.matmul`/`tanh` on
  `DEVICE`, with no per-update host syncs (state syncs once per trial, at
  commit).
- **Numpy backend (small organisms, production default via
  `ROXENE_NUMPY_BACKEND=on`):** the same buffers in numpy, matching the torch
  backend's dtype at compile and commit.
- Compile happens once per trial per organism instance; caching across trials
  is a possible follow-up.

## Tests

- `Neuron_test` — linear math check at the working precision; port routing
  after update; persistence round-trip of plain arrays.
- `Organism_test` — simultaneous (Jacobi) update semantics via
  `Organism.get_output`; input/output persistence.
- `NumpyBackend_test` — backend selection (torch by default, the numpy
  subclass only when small and enabled), live C is a numpy buffer,
  set-input/sync on numpy state, DB round-trip through the `before_flush`
  hook, and a full trial under the numpy backend.
- `Trial_test` / `Player_test` / `Environment_test` / `Population_test` —
  the worker lifecycle (start trial → run → complete → merge) end to end,
  which is the path that used to leave poisoned state behind.
