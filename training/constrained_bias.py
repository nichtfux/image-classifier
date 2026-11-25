"""Fit sparse output offsets on calibration data with recall constraints.

False-positive constraints use a conservative union bound across custom outputs.
Recall constraints preserve real/custom separation, followed by exact calibration
recall checks and pairwise constraints. Independent guards remain decisive.
"""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix


def fit_offsets(baseline, candidate, keys, top3):
    negatives = []
    budgets = []
    names = []
    for key in keys:
        with np.load(baseline / f'{key}.npz') as b, np.load(candidate / f'{key}.npz') as c:
            use = (c['role'] == 0) & (c['y'] < 270)
            if not use.any():
                continue
            bt = top3(b['z'][use])[:, 0]
            z = c['z'][use]
            negatives.append(z[:, 6:] - z[:, :1])
            budgets.append((int((bt >= 270).sum()), int((bt == 275).sum())))
            names.append(key)
    with np.load(baseline / 'black.npz') as b, np.load(candidate / 'black.npz') as c:
        use = (c['role'] == 0) & (c['y'] >= 270)
        y, z = c['y'][use], c['z'][use]
        initial = top3(z)
        installed = top3(b['z'][use])
    initial_hit = [initial[:, 0] == y, (initial == y[:, None]).any(1)]
    installed_hit = [installed[:, 0] == y, (installed == y[:, None]).any(1)]
    loss_budget = [int(hit.sum() - np.ceil((reference.mean() - .002) * len(y)))
                   for hit, reference in zip(initial_hit, installed_hit)]
    if min(loss_budget) < 0:
        return None, {'feasible': False, 'reason': 'Initial custom recall below calibration safety guard'}
    true_score = z[np.arange(len(z)), y - 270 + 6]
    # First enforce real/custom separation. Cross-custom ordering is checked
    # exactly after solving and repaired with pairwise linear constraints.
    gaps = [true_score - z[:, 0], true_score - z[:, 2]]
    choices = []
    columns = []
    costs = []
    for index in range(29):
        options = [0.]
        for margin in negatives:
            positive = np.sort(margin[:, index][margin[:, index] >= 0])[::-1]
            for rank in [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]:
                if rank < len(positive):
                    options.append(float(positive[rank]) + .001)
        for gap, hit in zip(gaps, initial_hit):
            values = np.sort(gap[(y == 270 + index) & hit])
            for rank in [0, 1, 2, 3, 5, 8, 13, 21, 34]:
                if rank < len(values):
                    options.append(max(0., float(values[rank]) - .001))
        for offset in np.unique(options):
            counts = []
            for margin in negatives:
                counts.extend([int((margin[:, index] > offset).sum()),
                               int((margin[:, index] > offset).sum()) if index == 5 else 0])
            losses = [int(((y == 270 + index) & hit & (gap <= offset)).sum())
                      for gap, hit in zip(gaps, initial_hit)]
            one = np.zeros(29)
            one[index] = 1
            columns.append(np.r_[counts, losses, one])
            costs.append(100 * losses[0] + 30 * losses[1] + .01 * offset)
            choices.append((index, float(offset)))
    matrix = np.stack(columns, 1)
    caps = np.r_[np.asarray(budgets).flatten(), loss_budget, np.ones(29)]
    lower = np.r_[np.full(2 * len(budgets) + 2, -np.inf), np.ones(29)]
    pair_caps = {}
    for iteration in range(4):
        result = milp(np.asarray(costs), integrality=np.ones(len(costs)),
                      bounds=Bounds(0., 1.), constraints=LinearConstraint(csc_matrix(matrix), lower, caps),
                      options={'time_limit': 15., 'mip_rel_gap': .005})
        info = {'solver_status': int(result.status), 'solver_message': result.message,
                'calibration_cohorts': names, 'custom_recall_loss_budgets': loss_budget,
                'choices': len(choices), 'iterations': iteration + 1, 'feasible': False}
        if result.x is None:
            return None, info
        selected = np.rint(result.x)
        if np.any(matrix @ selected > caps + 1e-5) or np.any(matrix @ selected < lower - 1e-5):
            return None, info
        offsets = np.zeros(29, np.float32)
        for active, (index, offset) in zip(selected, choices):
            if active:
                offsets[index] = offset
        predicted = top3(z, offsets)
        hits = [predicted[:, 0] == y, (predicted == y[:, None]).any(1)]
        passes = [hit.mean() >= reference.mean() - .002 for hit, reference in zip(hits, installed_hit)]
        if all(passes):
            info['feasible'] = True
            info['actual_custom_top1'] = float(hits[0].mean())
            info['actual_custom_top3'] = float(hits[1].mean())
            return offsets, info
        additions = {}
        for hit, original_hit, width in zip(hits, initial_hit, [1, 3]):
            for row in np.flatnonzero(original_hit & ~hit):
                rivals = [int(label) for label in predicted[row, :width] if label >= 270]
                if not rivals:
                    continue
                rival = rivals[-1] - 270
                truth = int(y[row] - 270)
                bound = float(true_score[row] - z[row, 6 + rival]) - .001
                pair = (truth, rival)
                additions[pair] = min(additions.get(pair, np.inf), bound)
        changed = False
        for pair, bound in additions.items():
            if bound >= pair_caps.get(pair, np.inf):
                continue
            pair_caps[pair] = bound
            truth, rival = pair
            coefficients = np.asarray([offset * ((index == truth) - (index == rival))
                                       for index, offset in choices])
            matrix = np.vstack([matrix, coefficients])
            lower = np.r_[lower, -np.inf]
            caps = np.r_[caps, bound]
            changed = True
        if not changed:
            return None, info
    return None, info
