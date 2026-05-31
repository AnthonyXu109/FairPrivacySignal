# Policy-Rule Configuration

FairPrivacySignal keeps its behavioral-signal suppression rules in
[`config/policy_rules.json`](../config/policy_rules.json). The file is intentionally
small and uses standard JSON so that benchmark reviewers can inspect the scenarios
without tracing Python conditionals.

## Behavioral-Signal Scenarios

Each scenario declares three boolean flags:

| Flag | Meaning |
|---|---|
| `require_consent` | Keep behavioral history only when the synthetic household consent field is true. |
| `exclude_sensitive_cohort` | Remove behavioral history for the synthetic sensitive-cohort proxy. |
| `remove_all_behavioral_history` | Remove behavioral history for every household. |

The current scenarios are:

| Scenario | Consent required | Sensitive cohort excluded | Remove all history |
|---|---:|---:|---:|
| `full_signal` | no | no | no |
| `consent_restricted` | yes | no | no |
| `policy_restricted` | yes | yes | no |
| `severe_signal_loss` | no | no | yes |

## Privacy-Exposure Weights

The same file declares weights for the benchmark's interpretable privacy-exposure
proxy. The weights must be non-negative and sum to `1.0`. This score is a comparison
diagnostic, not a formal privacy guarantee.

## Validation

`fairprivacysignal.policy_rules` validates required scenarios, rule flags, exposure
features, and weight normalization when the module loads. The regression suite also
checks expected scenario masks and rejects weight drift.
