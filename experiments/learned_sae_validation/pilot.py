"""Small, fully enumerated learned-SAE scope experiment (CPU).

The representation model and SAE are trained on 192/256 Boolean states.
The test set is never used for training or hyperparameter selection.
All 256 states define the finite intervention population for exact auditing.
No scope information enters either training objective.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)


def boolean_grid(n: int) -> np.ndarray:
    return ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(np.int64)


def xor_map(n: int, r: int) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        matrix[i, [(i + k) % n for k in range(r)]] = 1
    grid = boolean_grid(n)
    transformed = (grid @ matrix.T) % 2
    index = transformed @ (1 << np.arange(n))
    if len(np.unique(index)) != len(index):
        raise ValueError(f"Mixing map is not invertible for n={n}, r={r}.")
    return matrix, index


def fourier(values: np.ndarray) -> np.ndarray:
    """Uniform Boolean Fourier coefficients; row indices encode input subsets."""
    out = np.asarray(values, dtype=np.float64).copy()
    count = len(out)
    if count < 1 or count & (count - 1):
        raise ValueError("Population size must be a positive power of two.")
    size = 1
    while size < count:
        view = out.reshape(-1, 2 * size, out.shape[1])
        a = view[:, :size].copy()
        b = view[:, size:].copy()
        view[:, :size] = a + b
        view[:, size:] = a - b
        size *= 2
    return out / count


def exact_scope_errors(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coeff = fourier(z)
    energy = coeff ** 2
    captured = energy.copy()
    n = int(np.log2(len(z)))
    for bit in range(n):
        for mask in range(1 << n):
            if mask & (1 << bit):
                captured[mask] += captured[mask ^ (1 << bit)]
    errors = np.maximum(energy.sum(axis=0)[None, :] - captured, 0.0)
    errors[-1] = 0.0
    return errors, coeff


def optimize_scopes(errors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact minimum with <=B input-feature incidences, including backtracking."""
    count, features = errors.shape
    n = int(np.log2(count))
    card = np.array([i.bit_count() for i in range(count)])
    best = np.empty((features, n + 1))
    best_mask = np.empty((features, n + 1), dtype=np.int64)
    for k in range(n + 1):
        masks = np.flatnonzero(card <= k)
        arg = np.argmin(errors[masks], axis=0)
        best_mask[:, k] = masks[arg]
        best[:, k] = errors[masks[arg], np.arange(features)]
    max_b = features * n
    dp = np.zeros(max_b + 1)
    choices = np.zeros((features, max_b + 1), dtype=np.int64)
    for j in range(features):
        candidates = np.full((n + 1, max_b + 1), np.inf)
        for k in range(n + 1):
            candidates[k, k:] = dp[:max_b + 1 - k] + best[j, k]
        choices[j] = np.argmin(candidates, axis=0)
        dp = np.min(candidates, axis=0)
    return dp, choices, best_mask


def recover_assignment(budget: int, choices: np.ndarray, best_mask: np.ndarray) -> np.ndarray:
    scopes = np.zeros(len(choices), dtype=np.int64)
    for j in range(len(choices) - 1, -1, -1):
        k = choices[j, budget]
        scopes[j] = best_mask[j, k]
        budget -= k
    return scopes


class DenseAE(nn.Module):
    def __init__(self, n: int, d: int):
        super().__init__()
        self.encoder = nn.Linear(n, d)
        self.decoder = nn.Linear(d, n)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.hidden(x))


class SparseAE(nn.Module):
    def __init__(self, d: int, m: int, mean: torch.Tensor):
        super().__init__()
        self.enc = nn.Linear(d, m)
        self.raw_decoder = nn.Parameter(torch.randn(d, m) * 0.2)
        self.bias = nn.Parameter(mean.detach().clone())
        with torch.no_grad():
            self.enc.weight.copy_(self.dictionary().T)
            self.enc.bias.fill_(-0.05)

    def dictionary(self) -> torch.Tensor:
        return self.raw_decoder / torch.linalg.vector_norm(self.raw_decoder, dim=0, keepdim=True).clamp_min(1e-8)

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.enc(h - self.bias))

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(h)
        return z @ self.dictionary().T + self.bias, z


def train(seed: int, directory: Path, n: int = 8, d: int = 24, m: int = 48,
          model_steps: int = 1500, sae_steps: int = 2500, sparsity: float = 0.01) -> dict:
    started = time.perf_counter()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    grid = boolean_grid(n)
    inputs = torch.tensor(2 * grid - 1, dtype=torch.float32)
    permutation = rng.permutation(len(grid))
    train_idx, test_idx = permutation[:192], permutation[192:]
    model = DenseAE(n, d)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(model_steps):
        opt.zero_grad()
        loss = ((model(inputs[train_idx]) - inputs[train_idx]) ** 2).mean()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        h_raw = model.hidden(inputs)
        mean = h_raw[train_idx].mean(dim=0)
        scale = torch.sqrt(((h_raw[train_idx] - mean) ** 2).sum(dim=1).mean())
        h = (h_raw - mean) / scale
        task_train_mse = float(((model(inputs[train_idx]) - inputs[train_idx]) ** 2).mean())
        task_test_mse = float(((model(inputs[test_idx]) - inputs[test_idx]) ** 2).mean())
    sae = SparseAE(d, m, h[train_idx].mean(dim=0))
    opt = torch.optim.Adam(sae.parameters(), lr=0.01)
    history = []
    for step in range(sae_steps):
        opt.zero_grad()
        hhat, z = sae(h[train_idx])
        rec = ((hhat - h[train_idx]) ** 2).sum(dim=1).mean()
        penalty = z.sum(dim=1).mean()
        loss = rec + sparsity * penalty
        loss.backward()
        opt.step()
        if step % 250 == 0 or step == sae_steps - 1:
            history.append([step, float(rec.detach()), float(penalty.detach()), float((z > 0).float().sum(dim=1).mean())])
    sae.eval()
    with torch.no_grad():
        hhat, z = sae(h)
        h_np, z_np, hhat_np = [a.numpy().astype(np.float64) for a in (h, z, hhat)]
        D = sae.dictionary().numpy().astype(np.float64)
        bias = sae.bias.numpy().astype(np.float64)
    variance = ((h_np - h_np.mean(axis=0)) ** 2).sum(axis=1).mean()
    R = ((hhat_np - h_np) ** 2).sum(axis=1).mean()
    test_var = ((h_np[test_idx] - h_np[test_idx].mean(axis=0)) ** 2).sum(axis=1).mean()
    test_R = ((hhat_np[test_idx] - h_np[test_idx]) ** 2).sum(axis=1).mean()
    W = z_np.var(axis=0).sum()
    result = {
        'seed': seed, 'n': n, 'd': d, 'm': m, 'grid_size': len(grid),
        'train_states': len(train_idx), 'heldout_states': len(test_idx),
        'model_steps': model_steps, 'sae_steps': sae_steps, 'lambda': sparsity,
        'task_train_mse': task_train_mse, 'task_heldout_mse': task_test_mse,
        'R_over_Hvar': R / variance, 'heldout_R_over_Hvar': test_R / test_var,
        'R': R, 'Hvar': variance, 'K': float((z_np > 0).sum(axis=1).mean()), 'W': W,
        'dead_features': int((z_np.max(axis=0) == 0).sum()),
        'D_operator_squared': float(np.linalg.norm(D, 2) ** 2), 'maps': {}, 'history': history,
    }
    saved = dict(h=h_np, z=z_np, hhat=hhat_np, D=D, bias=bias,
                 train_idx=train_idx, test_idx=test_idx)
    for order in [1, 3, 5]:
        M, ids = xor_map(n, order)
        h_map, z_map = h_np[ids], z_np[ids]
        errors, z_coeff = exact_scope_errors(z_map)
        curve, choices, best_mask = optimize_scopes(errors)
        h_coeff = fourier(h_map)
        card = np.array([mask.bit_count() for mask in range(len(grid))])
        spectra = np.array([np.linalg.eigvalsh(h_coeff[card >= r].T @ h_coeff[card >= r])[::-1].clip(min=0)
                            for r in range(1, n + 1)])
        map_result = {'B': {}, 'spectral_lower_B': {},
                      'activation_order_energy': [float((z_coeff[card == r]**2).sum() / W) for r in range(1, n+1)],
                      'representation_order_energy': [float((h_coeff[card == r]**2).sum() / variance) for r in range(1, n+1)]}
        for tolerance in [.01, .05, .1, .2]:
            target = tolerance * W
            budget = int(np.flatnonzero(curve <= target + 1e-12)[0])
            threshold = (np.sqrt(R) + np.linalg.norm(D, 2) * np.sqrt(target)) ** 2
            tail = np.maximum(spectra.sum(axis=1)[:, None] - np.concatenate([np.zeros((n, 1)), np.cumsum(spectra, axis=1)], axis=1), 0)
            lower_B = int(sum(np.flatnonzero(row <= threshold + 1e-12)[0] for row in tail))
            map_result['B'][str(tolerance)] = budget
            map_result['spectral_lower_B'][str(tolerance)] = lower_B
        result['maps'][str(order)] = map_result
        saved.update({f'M_{order}': M, f'ids_{order}': ids, f'curve_{order}': curve,
                      f'errors_{order}': errors, f'choices_{order}': choices,
                      f'best_mask_{order}': best_mask, f'spectra_{order}': spectra})
    result['elapsed_seconds'] = time.perf_counter() - started
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / f'seed_{seed}.npz', **saved)
    torch.save({'dense_model': model.state_dict(), 'sae': sae.state_dict(), 'mean': mean, 'scale': scale},
               directory / f'seed_{seed}_weights.pt')
    (directory / f'seed_{seed}.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--out', type=Path, default=Path(__file__).parent / 'results')
    args = parser.parse_args()
    for seed in args.seeds:
        result = train(seed, args.out)
        short = {k: result[k] for k in ['seed','task_heldout_mse','R_over_Hvar','heldout_R_over_Hvar','K','dead_features','D_operator_squared','elapsed_seconds']}
        short['budgets'] = {k: v['B'] for k,v in result['maps'].items()}
        short['spectral_lower_B'] = {k: v['spectral_lower_B'] for k,v in result['maps'].items()}
        print(json.dumps(short, indent=2), flush=True)
