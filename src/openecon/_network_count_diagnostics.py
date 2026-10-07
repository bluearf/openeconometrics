"""Bounded restart diagnostics shared by the native count block models."""


def restart_seed(seed, start):
    """Keep a restart independent of earlier starts' stopping decisions."""
    return (seed + start) % 2**63


def run_diagnostics(seed, start, groups, max_iter, history, membership_changes):
    if groups == 1:
        reason = "unique_partition"
    elif not max_iter:
        reason = "fixed_partition"
    elif membership_changes and not membership_changes[-1]:
        reason = "no_admissible_move"
    else:
        reason = "max_iter"
    return dict(seed=restart_seed(seed, start), stopping_reason=reason,
                membership_changes=membership_changes,
                likelihood_gains=[after - before for before, after in zip(history, history[1:])])
