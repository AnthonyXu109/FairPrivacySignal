# Fairness Diagnostics

FairPrivacySignal reports fairness diagnostics separately from ranking utility. The
benchmark does not treat any single metric as proof that a ranking or allocation
policy is fair.

## Ranking Diagnostics

### Low-Signal NDCG@3

Low-signal NDCG@3 measures ranking quality for households with reduced behavioral
signal availability. The benchmark also reports the gap between not-low-signal and
low-signal NDCG@3.

### Score-Matched Subgroup Calibration

Aggregate ranking metrics can hide differences among similarly scored candidates.
The score-matched calibration diagnostic:

1. partitions test-set events into shared predicted-score bins
2. measures observed relevance separately for low-signal and not-low-signal events
3. compares observed relevance rates only when both groups have enough events in a bin
4. reports a weighted mean absolute matched relevance gap across eligible bins

The diagnostic also reports expected calibration error (ECE) for each subgroup:
the weighted average absolute difference between mean predicted relevance and
observed relevance within score bins.

The implementation is intentionally lightweight. It is inspired by public
ranking-calibration research but is not a full matched-pair estimator.

## Allocation Diagnostics

When service capacity is limited, the benchmark reports:

- allocated relevance rate
- low-signal and not-low-signal selection rates
- selection-rate gap
- allocated low-signal share
- relevance cost relative to utility-only allocation

These metrics expose tradeoffs rather than declaring one universally optimal policy.

## Limitations

All fairness results are synthetic diagnostics. They do not establish legal,
operational, or domain-specific fairness, and they should not be used as eligibility
criteria for real services.
