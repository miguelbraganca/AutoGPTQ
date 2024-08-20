import math
import os
import time
from logging import getLogger
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import transformers

from .quantizer import Quantizer
from .hqq import quantize as hqq_quantize
from .hqq import dequantize as hqq_dequantize
from ..utils.stochastic_comb import stochastically_combine_tensors as stochastic_comb

logger = getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:
    def __init__(self, layer: nn.Linear):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H: torch.Tensor = torch.zeros(
            (self.columns, self.columns), device=self.dev
        )
        self.nsamples: int = 0
        self.quantizer = Quantizer()
        self.avg_input: Optional[torch.Tensor] = None
        self.scale: List[torch.Tensor] = []
        self.zero: List[torch.Tensor] = []
        self.now_idx: int = 1

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        self.out1 = out
        self.inp1 = inp
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        if self.avg_input is None:
            self.avg_input = torch.zeros_like(inp)
        self.avg_input += inp.float() / self.nsamples
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self,
        blocksize: int = 128,
        percdamp: float = 0.01,
        group_size: int = -1,
        actorder: bool = False,
        static_groups: bool = False,
        quantized_inputs: Optional[torch.Tensor] = None,
        L: float = 0.0001,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform faster quantization of the layer weights.

        Args:
            blocksize (int): Size of blocks for processing.
            percdamp (float): Percentage of damping to apply.
            group_size (int): Size of groups for quantization. -1 means no grouping.
            actorder (bool): Whether to use activation order.
            static_groups (bool): Whether to use static groups.
            quantized_inputs (Optional[torch.Tensor]): (optional) quantized inputs, in addition to the standard avg_input
            L (float): Relative size of the gradient contribution to the total loss (should be 1 but very unstable)

        Returns:
            tuple: Containing scale, zero, and group indices.
        """
        # Clone and convert weight to float
        W = self.layer.weight.data.clone().float()

        tick = time.time()

        # Initialize quantizer if not ready
        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        # Prepare HQQ quantization
        hqq_group_size = group_size if group_size != -1 else None
        hqq_quant_config = self._prepare_hqq_config(hqq_group_size)

        # Perform HQQ quantization
        _, meta_hqq, Q_hqq = hqq_quantize(W, **hqq_quant_config)
        W_hqq = hqq_dequantize(_, meta_hqq)
        del _

        # Process Hessian matrix
        H = self._process_hessian(W)

        # Initialize quantization parameters
        g_idx, scale, zero = [], [], []
        now_idx = 1

        # Handle static groups if enabled
        if static_groups:
            groups = self._create_static_groups(W, group_size)

        # Reorder columns if actorder is True
        if actorder:
            W, H, invperm = self._reorder_columns(W, H)

        # Initialize quantization variables
        Losses = torch.zeros_like(W)
        Q = W.clone()

        # Prepare Hessian inverse
        Hinv = self._prepare_hessian_inverse(H, percdamp)

        # Prepare input matrix
        X = self._prepare_input_matrix(quantized_inputs)

        # Process weights in blocks
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            # Extract block data
            W1, Q1, W1_hqq, Hinv1, X1 = self._extract_block_data(
                W, Q, W_hqq, Hinv, X, i1, i2
            )

            # Process each column in the block
            for i in range(count):
                w, w_hqq = W1[:, i], W1_hqq[:, i]
                d = Hinv1[i, i]

                # Update quantizer parameters if necessary
                self._update_quantizer_params(
                    i, i1, group_size, static_groups, W, groups, actorder
                )

                # Quantize the column
                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q

                # Calculate losses and errors
                Losses1, err1 = self._calculate_losses_and_errors(w, q, d)

                # Update weights
                W1, Hinv_grad_d1 = self._update_weights(
                    W1, Q1, X1, Hinv1, err1, i, count, L
                )

            # Update global matrices
            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            # Final weight update for the block
            W = self._final_weight_update(W, Q, X, Hinv, Hinv_grad_d1, i1, i2, L)

            # Debug: Update layer weights
            if os.environ.get("DEBUG"):
                self._update_layer_weights_debug(Q, W, i2)

        torch.cuda.synchronize()

        # Prepare group indices
        g_idx = self._prepare_group_indices(group_size, static_groups, actorder)

        # Reorder Q if actorder was used
        if actorder:
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]

        # Update layer weights
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).type_as(
            self.layer.weight.data
        )

        # Log results
        self._log_results(X, Grad, tick)

        # Finalize scale and zero
        scale, zero = self._finalize_scale_zero(scale, zero)

        return scale, zero, g_idx

    def _prepare_hqq_config(self, hqq_group_size: Optional[int]) -> Dict[str, Any]:
        """Prepare the HQQ quantization configuration."""
        return {
            "weight_quant_params": {
                "nbits": self.quantizer.bits,
                "channel_wise": self.quantizer.perchannel,
                "group_size": hqq_group_size,
                "optimize": True,
                "round_zero": False,
                "axis": 1,
                "view_as_float": False,
            },
            "scale_quant_params": None,
            "zero_quant_params": None,
        }

    def _process_hessian(self, W: torch.Tensor) -> torch.Tensor:
        """Process the Hessian matrix."""
        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        return H

    def _create_static_groups(
        self, W: torch.Tensor, group_size: int
    ) -> List[Quantizer]:
        """Create static groups for quantization."""
        import copy

        groups = []
        for i in range(0, self.columns, group_size):
            quantizer = copy.deepcopy(self.quantizer)
            quantizer.find_params(W[:, i : (i + group_size)], weight=True)
            self.scale.append(quantizer.scale)
            self.zero.append(quantizer.zero)
            groups.append(quantizer)
        return groups

    def _reorder_columns(
        self, W: torch.Tensor, H: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reorder columns based on Hessian diagonal."""
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]
        invperm = torch.argsort(perm)
        return W, H, invperm

    def _prepare_hessian_inverse(
        self, H: torch.Tensor, percdamp: float
    ) -> torch.Tensor:
        """Prepare the inverse of the Hessian matrix."""
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        H = H * torch.diag(H).unsqueeze(1)
        return H

    def _prepare_input_matrix(
        self, quantized_inputs: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Prepare the input matrix."""
        if quantized_inputs:
            return quantized_inputs
        else:
            return self.avg_input.float()

    def _extract_block_data(
        self,
        W: torch.Tensor,
        Q: torch.Tensor,
        W_hqq: torch.Tensor,
        Hinv: torch.Tensor,
        X: torch.Tensor,
        i1: int,
        i2: int,
    ) -> Tuple[torch.Tensor, ...]:
        """Extract data for the current block."""
        W1 = W[:, i1:i2].clone()
        Q1 = W1.clone()
        W1_hqq = W_hqq[:, i1:i2].clone()
        Hinv1 = Hinv[i1:i2, i1:i2]
        X1 = X[i1:i2, :].clone().float()
        return W1, Q1, W1_hqq, Hinv1, X1

    def _update_quantizer_params(
        self,
        i: int,
        i1: int,
        group_size: int,
        static_groups: bool,
        W: torch.Tensor,
        groups: List[Quantizer],
        actorder: bool,
    ) -> None:
        """Update quantizer parameters if necessary."""
        if group_size != -1:
            if not static_groups:
                if (i1 + i) % group_size == 0:
                    self.quantizer.find_params(
                        W[:, (i1 + i) : (i1 + i + group_size)], weight=True
                    )
                if ((i1 + i) // group_size) - self.now_idx == -1:
                    self.scale.append(self.quantizer.scale)
                    self.zero.append(self.quantizer.zero)
                    self.now_idx += 1
            else:
                idx = i1 + i
                if actorder:
                    idx = self.perm[idx]
                self.quantizer = groups[idx // group_size]

    def _calculate_losses_and_errors(
        self, w: torch.Tensor, q: torch.Tensor, d: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate losses and errors for the current column."""
        Losses1 = (w - q) ** 2 / d**2
        err1 = (w - q) / d
        return Losses1, err1

    def _update_weights(
        self,
        W1: torch.Tensor,
        Q1: torch.Tensor,
        X1: torch.Tensor,
        X_quant1: Optional[torch.Tensor],
        Hinv1: torch.Tensor,
        err1: torch.Tensor,
        i: int,
        count: int,
        L: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update weights for the current column."""
        # if i != count - 1:
        #     diff[:, i + 1 :] = 0
        if not X_quant1:
            X_quant1 = X1

        grad1 = 2 * (W1.matmul(X1) - Q1.matmul(X_quant1)).matmul(X1.t())
        Hinv1_Grad = Hinv1.matmul(grad1.t()).t()
        hinv_grad_d1 = Hinv1_Grad[:, i] / Hinv1[i, i]

        original_term = err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
        term1 = Hinv1.matmul(grad1.t()).t()[:, i:]
        term2 = (hinv_grad_d1).unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))

        W1[:, i:] += -original_term - (term1 - term2) * L**2
        return W1, hinv_grad_d1

    def _final_weight_update(
        self,
        W: torch.Tensor,
        Q: torch.Tensor,
        X: torch.Tensor,
        X_quant: Optional[torch.Tensor],
        Hinv: torch.Tensor,
        Hinv_grad_d1: torch.Tensor, # TODO: check if this is correct
        i1: int,
        i2: int,
        L: float,
    ) -> torch.Tensor:
        """Perform final weight update for the block."""

        if not X_quant:
            X_quant = X
        Grad = 2 * (W.matmul(X) - Q.matmul(X_quant)).matmul(X.t())
        # Hinv_Grad = Hinv.matmul(Grad.t()).t()
        term1 = Hinv.matmul(Grad.t()).t()[:, i2:]
        term2 = (Hinv_grad_d1).matmul(Hinv[i1:i2, i2:])

        W[:, i2:] += -Hinv_grad_d1.matmul(Hinv[i1:i2, i2:]) - (term1 - term2) * L**2
        return W

    def _update_layer_weights_debug(
        self, Q: torch.Tensor, W: torch.Tensor, i2: int
    ) -> None:
        """Update layer weights for debugging."""
        self.layer.weight.data[:, :i2] = Q[:, :i2]
        self.layer.weight.data[:, i2:] = W[:, i2:]

    def _prepare_group_indices(
        self, group_size: int, static_groups: bool, actorder: bool
    ) -> torch.Tensor:
        """Prepare group indices."""
        group_size = group_size if group_size != -1 else self.columns
        if static_groups and actorder:
            g_idx = [self.perm[i] // group_size for i in range(self.columns)]
        else:
            g_idx = [i // group_size for i in range(self.columns)]
        return torch.tensor(g_idx, dtype=torch.int32, device=Q.device)

    def _log_results(self, X: torch.Tensor, Grad: torch.Tensor, tick: float) -> None:
        """Log the results of quantization."""
        logger.info(
            f"Layer L2 Loss: {torch.sum((self.layer(self.inp1) - self.out1) ** 2)}"
        )
        logger.info(f"avg magnitude gradient: {torch.mean(torch.abs(Grad))}")
        logger.info(f"duration: {(time.time() - tick)}")

    def _finalize_scale_zero(
        self, scale: List[torch.Tensor], zero: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Finalize scale and zero parameters."""
        if not scale:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)
        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)
        return scale, zero

    @property
    def quantized_weight(self) -> torch.Tensor:
        return self.layer.weight.data

    def calculate_quantized_inputs(self) -> torch.Tensor:
        with torch.no_grad():
            quantized_output = self.layer(self.inp1)
        return quantized_output

    def free(self) -> None:
        if os.environ.get("DEBUG"):
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()


__all__ = ["GPTQ"]
