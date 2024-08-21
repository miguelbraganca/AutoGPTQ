import math
import os
import time
from logging import getLogger

import torch
import torch.nn as nn
import transformers

from .quantizer import Quantizer

logger = getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows, self.columns = W.shape
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.quantizer = Quantizer()

    def add_batch(self, inp, out):
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
        if not hasattr(self, 'avg_input'):
            self.avg_input = torch.zeros_like(inp)
        self.avg_input += inp.float() / self.nsamples
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(self, blocksize=128, percdamp=0.01, group_size=-1, actorder=False, static_groups=False):
        W = self.layer.weight.data.clone().float()
        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        g_idx, scale, zero = [], [], []
        now_idx = 1

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, group_size):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + group_size)], weight=True)
                scale.append(quantizer.scale)
                zero.append(quantizer.zero)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = W.clone()

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        H = H * torch.diag(H).unsqueeze(1)
        Hinv = H
        X = self.avg_input.float()

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = W1.clone()
            Err1 = torch.zeros_like(W1)
            Hinv_grad_d1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            X1 = X[i1:i2, :].clone().float()

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if group_size != -1:
                    if not static_groups:
                        if (i1 + i) % group_size == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + group_size)], weight=True)
                        if ((i1 + i) // group_size) - now_idx == -1:
                            scale.append(self.quantizer.scale)
                            zero.append(self.quantizer.zero)
                            now_idx += 1
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // group_size]

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2
                err1 = (w - q) / d
                
                grad1 = 2 * ((W1 - Q1).matmul(X1)).matmul(X1.t())
                Hinv1_Grad = Hinv1.matmul(grad1.t()).t()
                hinv_grad_d1 = Hinv1_Grad[:, i] / d
                
                original_term = err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                term1 = Hinv1.matmul(grad1.t()).t()[:, i:]
                term2 = (hinv_grad_d1).unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))

                W1[:, i:] += -original_term - (term1 - term2) * 0.001**2
                Err1[:, i] = err1
                Hinv_grad_d1[:, i] = hinv_grad_d1

            Q[:, i1:i2] = Q1            
            Losses[:, i1:i2] = Losses1 / 2
            Grad = 2 * ((W - Q).matmul(X)).matmul(X.t())
            Hinv_Grad = Hinv.matmul(Grad.t()).t()
            term1 = Hinv.matmul(Grad.t()).t()[:, i2:]
            term2 = (Hinv_grad_d1).matmul(Hinv[i1:i2, i2:])

            W[:, i2:] += -Err1.matmul(Hinv[i1:i2, i2:]) - (term1 - term2) * 0.001**2

            if os.environ.get("DEBUG"):
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]

        torch.cuda.synchronize()

        group_size = group_size if group_size != -1 else self.columns
        if static_groups and actorder:
            g_idx = [perm[i] // group_size for i in range(self.columns)]
        else:
            g_idx = [i // group_size for i in range(self.columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)
        if actorder:
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).type_as(self.layer.weight.data)

        logger.info(f"Layer L2 Loss: {torch.sum((self.layer(self.inp1) - self.out1) ** 2)}")
        logger.info(f"avg magnitude gradient: {torch.mean(torch.abs(Grad))}")
        logger.info(f"duration: {(time.time() - tick)}")

        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)
        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)

        return scale, zero, g_idx

    def free(self):
        if os.environ.get("DEBUG"):
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()

__all__ = ["GPTQ"]