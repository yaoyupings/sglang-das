"""WeLM exact per-rank prefix routing for prefix_affinity DP dispatch.

Enabled with WELM_DP_PAGE_ROUTE=1 (--load-balance-method prefix_affinity).

Each rank keeps the chained hashes of every WELM_DP_PAGE_GRAN-token block it
was sent, so a request's cached prefix on each rank is known to block
precision. The rank with the longest cached prefix wins unless its uncached
prefill backlog exceeds the cheapest rank's by WELM_DP_PAGE_SLACK.

Knobs:
  WELM_DP_PAGE_GRAN           block size in tokens (64, = page size)
  WELM_DP_PAGE_SLACK          prefill backlog slack in tokens (4096)
  WELM_DP_PAGE_MAX_KEYS       clear all block sets above this many keys (4M)
  WELM_DP_PAGE_OWN_BACKLOG    <tok/s>: estimate each rank's prefill backlog from
                              the uncached tokens this router sent it, drained
                              at that rate, instead of the scheduler snapshot
                              (which counts queued requests whose prefix is
                              still being prefilled by an earlier request as
                              uncached). 0 = use the snapshot.
  WELM_DP_PAGE_TOKEN_SLACK    total-token imbalance that overrides prefix
                              ownership (0 = SGLANG_DP_AFFINITY_SLACK_TOKENS)
  WELM_DP_PAGE_BAL_HITS=1     on token imbalance, pick the rank with the most
                              cached prefix among those within the token slack
                              of the lightest rank (instead of the lightest)
  WELM_DP_PAGE_TRACE=<path>   per-request record: time rank branch len hit
                              pref_hit max_hit, and FLUSH on flush_cache
"""

import logging
import os
import time

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def enabled() -> bool:
    return os.environ.get("WELM_DP_PAGE_ROUTE", "0") == "1"


class WelmPageRouter:
    def __init__(self, dp_size: int):
        self.dp_size = dp_size
        self.gran = int(os.environ.get("WELM_DP_PAGE_GRAN", "64"))
        self.slack = int(os.environ.get("WELM_DP_PAGE_SLACK", "4096"))
        self.max_keys = int(os.environ.get("WELM_DP_PAGE_MAX_KEYS", "4000000"))
        self.drain = float(os.environ.get("WELM_DP_PAGE_OWN_BACKLOG", "0"))
        self.token_slack = int(os.environ.get("WELM_DP_PAGE_TOKEN_SLACK", "0"))
        self.bal_hits = os.environ.get("WELM_DP_PAGE_BAL_HITS", "0") == "1"
        trace_path = os.environ.get("WELM_DP_PAGE_TRACE")
        self.trace = open(trace_path, "a", buffering=1) if trace_path else None
        self.sets = [set() for _ in range(dp_size)]
        self.backlog = [0.0] * dp_size
        self.backlog_t = time.monotonic()
        # requests, routed hit tokens, routed tokens, per-branch counts, per-branch lost hits
        self.stats = [0, 0, 0, [0, 0, 0], [0, 0, 0]]
        logger.info(
            "WELM_DP_PAGE_ROUTE: gran=%d slack=%d own_backlog_drain=%g bal_hits=%d",
            self.gran, self.slack, self.drain, self.bal_hits,
        )

    def reset(self):
        """flush_cache emptied every rank's radix cache."""
        for s in self.sets:
            s.clear()
        if self.trace is not None:
            self.trace.write(f"{time.monotonic():.3f} FLUSH\n")

    def select(self, ids, budget) -> int:
        """Return the target rank and update the load budget."""
        gran, ranks = self.gran, budget.dp_size
        if ranks != self.dp_size:  # elastic scale-up/down: start over with the new rank set
            self.dp_size = ranks
            self.sets = [set() for _ in range(ranks)]
            self.backlog = [0.0] * ranks
        length = len(ids)
        chain = []
        h = 0
        for start in range(0, length - length % gran, gran):
            h = hash((h, tuple(ids[start:start + gran])))
            chain.append(h)
        hits = [0] * ranks
        for rank in range(ranks):
            s = self.sets[rank]
            k = 0
            for key in chain:
                if key not in s:
                    break
                k += 1
            hits[rank] = k * gran
        if self.drain > 0:
            now = time.monotonic()
            dec = self.drain * (now - self.backlog_t)
            self.backlog_t = now
            for rank in range(ranks):
                self.backlog[rank] = max(0.0, self.backlog[rank] - dec)
            backlog = self.backlog
        else:
            backlog = budget.waiting_uncached
        costs = [backlog[rank] + length - hits[rank] for rank in range(ranks)]
        preferred = min(range(ranks), key=lambda r: (-hits[r], costs[r], budget.total_tokens[r]))
        best = min(range(ranks), key=lambda r: (costs[r], budget.total_tokens[r], budget.total_requests[r]))
        least = min(range(ranks), key=lambda r: (budget.total_tokens[r], budget.total_requests[r]))
        token_slack = self.token_slack or envs.SGLANG_DP_AFFINITY_SLACK_TOKENS.get()
        if costs[preferred] - costs[best] > self.slack:
            target, branch = best, 1
        elif token_slack > 0 and budget.total_tokens[preferred] - budget.total_tokens[least] > token_slack:
            branch = 2
            if self.bal_hits:
                floor = budget.total_tokens[least] + token_slack
                target = min((r for r in range(ranks) if budget.total_tokens[r] <= floor),
                             key=lambda r: (-hits[r], costs[r], budget.total_tokens[r]))
            else:
                target = least
        else:
            target, branch = preferred, 0
        budget.waiting_uncached[target] = costs[target]
        if self.drain > 0:
            backlog[target] += length - hits[target]
        budget.total_requests[target] += 1
        budget.total_tokens[target] += length
        if sum(len(s) for s in self.sets) > self.max_keys:
            for s in self.sets:
                s.clear()
        self.sets[target].update(chain)

        stats = self.stats
        stats[0] += 1
        stats[1] += hits[target]
        stats[2] += length
        stats[3][branch] += 1
        stats[4][branch] += hits[preferred] - hits[target]
        if stats[0] % 500 == 0:
            logger.info("WeLM page routing: requests=%d est_hit=%.3f backlog=%s "
                        "branches(pref/cost/tokens)=%s lost_hit_tokens=%s",
                        stats[0], stats[1] / max(1, stats[2]), [int(b) for b in backlog],
                        stats[3], stats[4])
        if self.trace is not None:
            self.trace.write(f"{time.monotonic():.3f} {target} {branch} {length} "
                             f"{hits[target]} {hits[preferred]} {max(hits)}\n")
        return target
