"""The compiled execution form of an organism's neural network.

Cells carry plain state (see Neuron). On first update they are compiled into
pre-multiplied routing/weight matrices and one contiguous state vector C, so
each update is a few matmuls. The network owns the live buffers; commit_state()
writes them back to the cells, so it is a pure performance transform.

CompiledNetwork is the torch backend (the default). NumpyCompiledNetwork
subclasses it as the fast path for small organisms, where numpy wins.
The organism picks which to build; the public API is the same either way.
"""

import numpy as np
import torch

from .cells.input_cell import InputCell
from .cells.neuron import Neuron
from .constants import DEVICE, NP_PRECISION, TORCH_PRECISION


class CompiledNetwork:
    """Torch backend (default). NumpyCompiledNetwork subclasses this for small organisms."""

    def __init__(self, cells):
        input_cells = [c for c in cells if isinstance(c, InputCell)]
        neurons = [c for c in cells if isinstance(c, Neuron)]
        if not neurons:
            raise ValueError("Cannot compile a network with no neurons")

        num_inputs = len(input_cells)
        rec_len = sum(n.feedback.shape[0] + 1 for n in neurons)

        unconnected_ports = []
        for n in neurons:
            for p in range(n.input.shape[0]):
                if p not in n.bound_ports:
                    unconnected_ports.append((n, p))
        total_unconnected = len(unconnected_ports)
        dim_C = num_inputs + rec_len + total_unconnected

        # C layout: [input cell values | per-neuron (feedback..., output) | unconnected input ports]
        cell_to_idx = {ic: i for i, ic in enumerate(input_cells)}
        rec_pos = num_inputs
        for n in neurons:
            rec_pos += n.feedback.shape[0]
            cell_to_idx[n] = rec_pos
            rec_pos += 1

        total_in_fb = sum(n.input.shape[0] + n.feedback.shape[0] for n in neurons)
        total_ports = sum(n.input.shape[0] for n in neurons)

        self.P_route = self._build_p_route(
            neurons, cell_to_idx, num_inputs, rec_len, total_in_fb, unconnected_ports)

        w1 = self._block_diag([self._concat(n.input_hidden, n.feedback_hidden, axis=0) for n in neurons])
        w2 = self._block_diag([self._concat(n.hidden_feedback, n.hidden_output, axis=1) for n in neurons])
        W_eff1 = self.P_route @ w1

        self.P_input = self._build_p_input(
            neurons, cell_to_idx, num_inputs, rec_len, total_ports, unconnected_ports)

        # Where each neuron's state lives in C / all_inputs, so commit_state() can write it back.
        self._state_slices = []
        rec_pos, port_pos = num_inputs, 0
        for n in neurons:
            fb_len = n.feedback.shape[0]
            in_len = n.input.shape[0]
            self._state_slices.append((n, rec_pos, fb_len, rec_pos + fb_len, port_pos, in_len))
            rec_pos += fb_len + 1
            port_pos += in_len
        self.neuron_output_pos = {s[0]: s[3] for s in self._state_slices}
        self.input_cells_map = {ic: i for i, ic in enumerate(input_cells)}
        self.num_inputs = num_inputs
        self.rec_len = rec_len

        self.C = self._build_c(input_cells, neurons, unconnected_ports, num_inputs, rec_len, dim_C)
        self.all_inputs = self._zeros((1, total_ports))
        self.W_eff1 = W_eff1
        self.w2 = w2

    # -- backend ops (torch) ---------------------------------------------------

    def _zeros(self, shape):
        return torch.zeros(shape, dtype=TORCH_PRECISION, device=DEVICE)

    def _as_work_array(self, a):
        return torch.tensor(np.asarray(a, dtype=NP_PRECISION), dtype=TORCH_PRECISION, device=DEVICE)

    def _from_values(self, values):
        return torch.tensor(values, dtype=TORCH_PRECISION, device=DEVICE)

    def _concat(self, a, b, axis):
        return torch.cat([self._as_work_array(a), self._as_work_array(b)], dim=axis)

    # -- construction helpers --------------------------------------------------

    def _block_diag(self, blocks):
        out = self._zeros((sum(b.shape[0] for b in blocks), sum(b.shape[1] for b in blocks)))
        r = c = 0
        for b in blocks:
            out[r:r + b.shape[0], c:c + b.shape[1]] = b
            r += b.shape[0]
            c += b.shape[1]
        return out

    def _build_p_route(self, neurons, cell_to_idx, num_inputs, rec_len, total_in_fb, unconnected_ports):
        dim_C = num_inputs + rec_len + len(unconnected_ports)
        P_route = self._zeros((dim_C, total_in_fb))
        in_fb_pos = 0
        rec_fb_pos = num_inputs
        unconn_pos = num_inputs + rec_len
        for n in neurons:
            for p in range(n.input.shape[0]):
                if p in n.bound_ports:
                    P_route[cell_to_idx[n.bound_ports[p]], in_fb_pos + p] = 1.0
                else:
                    P_route[unconn_pos, in_fb_pos + p] = 1.0
                    unconn_pos += 1
            in_fb_pos += n.input.shape[0]
            for fb_i in range(n.feedback.shape[0]):
                P_route[rec_fb_pos + fb_i, in_fb_pos + fb_i] = 1.0
            rec_fb_pos += n.feedback.shape[0] + 1
            in_fb_pos += n.feedback.shape[0]
        return P_route

    def _build_p_input(self, neurons, cell_to_idx, num_inputs, rec_len, total_ports, unconnected_ports):
        dim_C = num_inputs + rec_len + len(unconnected_ports)
        P_input = self._zeros((dim_C, total_ports))
        port_pos = 0
        unconn_pos = num_inputs + rec_len
        for n in neurons:
            in_len = n.input.shape[0]
            for p in range(in_len):
                if p in n.bound_ports:
                    P_input[cell_to_idx[n.bound_ports[p]], port_pos + p] = 1.0
                else:
                    P_input[unconn_pos, port_pos + p] = 1.0
                    unconn_pos += 1
            port_pos += in_len
        return P_input

    def _build_c(self, input_cells, neurons, unconnected_ports, num_inputs, rec_len, dim_C):
        c_vals = [0.0] * dim_C
        for i, ic in enumerate(input_cells):
            val = ic.get_output()
            if val is not None:
                c_vals[i] = float(val)
        rec_pos = num_inputs
        for n in neurons:
            fb_len = n.feedback.shape[0]
            c_vals[rec_pos:rec_pos + fb_len] = n.feedback.ravel().tolist()
            rec_pos += fb_len
            c_vals[rec_pos:rec_pos + 1] = n.output.ravel().tolist()
            rec_pos += 1
        unconn_idx = num_inputs + rec_len
        for n, p in unconnected_ports:
            c_vals[unconn_idx] = float(n.input[p])
            unconn_idx += 1
        return self._from_values([c_vals])

    # -- live operations (torch) -----------------------------------------------

    def update(self, sync_neuron_inputs: bool = True):
        """One Jacobi-style step: C[rec region] <- tanh(tanh(C W1) W2)."""
        C, W_eff1, w2 = self.C, self.W_eff1, self.w2
        rec_start, rec_end = self.num_inputs, self.num_inputs + self.rec_len
        hidden = torch.tanh(torch.matmul(C, W_eff1))
        if sync_neuron_inputs:
            torch.matmul(C, self.P_input, out=self.all_inputs)
        torch.tanh(torch.matmul(hidden, w2), out=C[:, rec_start:rec_end])

    def sync_neuron_inputs(self):
        """all_inputs <- C @ P_input (route every input port from the live state)."""
        torch.matmul(self.C, self.P_input, out=self.all_inputs)

    def set_input(self, input_cell: InputCell, value):
        idx = self.input_cells_map.get(input_cell)
        if idx is not None:
            self.C[0, idx] = value

    def get_output(self, neuron: Neuron):
        """Live output state for a neuron, or None if it isn't part of this network."""
        pos = self.neuron_output_pos.get(neuron)
        if pos is None:
            return None
        return NP_PRECISION(float(self.C[0, pos].item()))

    def commit_state(self):
        """Write the live state back to the plain neuron arrays.

        After this, the neurons' input/feedback/output arrays are authoritative
        and this network may be discarded or rebuilt from the cells.
        """
        C, all_inputs = self.C, self.all_inputs

        def to_plain(t):
            return t.detach().cpu().numpy().astype(NP_PRECISION)

        for n, fb_start, fb_len, out_pos, in_start, in_len in self._state_slices:
            n.feedback = to_plain(C[0, fb_start:fb_start + fb_len])
            n.output = to_plain(C[0, out_pos:out_pos + 1])
            n.input = to_plain(all_inputs[0, in_start:in_start + in_len])


class NumpyCompiledNetwork(CompiledNetwork):
    """Numpy fast path for small organisms; overrides the backend ops and hot path."""

    def _zeros(self, shape):
        return np.zeros(shape, dtype=NP_PRECISION)

    def _as_work_array(self, a):
        return np.asarray(a, dtype=NP_PRECISION)

    def _from_values(self, values):
        return np.array(values, dtype=NP_PRECISION)

    def _concat(self, a, b, axis):
        return np.concatenate([self._as_work_array(a), self._as_work_array(b)], axis=axis)

    def update(self, sync_neuron_inputs: bool = True):
        C, W_eff1, w2 = self.C, self.W_eff1, self.w2
        rec_start, rec_end = self.num_inputs, self.num_inputs + self.rec_len
        hidden = np.tanh(np.matmul(C, W_eff1))
        if sync_neuron_inputs:
            np.matmul(C, self.P_input, out=self.all_inputs)
        C[:, rec_start:rec_end] = np.tanh(np.matmul(hidden, w2))

    def sync_neuron_inputs(self):
        np.matmul(self.C, self.P_input, out=self.all_inputs)

    def get_output(self, neuron: Neuron):
        pos = self.neuron_output_pos.get(neuron)
        if pos is None:
            return None
        return NP_PRECISION(float(self.C[0, pos]))

    def commit_state(self):
        C, all_inputs = self.C, self.all_inputs
        for n, fb_start, fb_len, out_pos, in_start, in_len in self._state_slices:
            n.feedback = C[0, fb_start:fb_start + fb_len].copy()
            n.output = C[0, out_pos:out_pos + 1].copy()
            n.input = all_inputs[0, in_start:in_start + in_len].copy()
