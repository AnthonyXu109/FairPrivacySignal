# Contributing to FairPrivacySignal

FairPrivacySignal is a public synthetic benchmark for studying privacy, utility,
and fairness tradeoffs under signal loss. Contributions are welcome when they
make the benchmark easier to reproduce, audit, or extend.

## Before opening a pull request

Create a focused branch and keep each change reviewable. From the repository
root, run:

```bash
bash scripts/verify_benchmark.sh
```

This command runs the regression tests, checks Python compilation, rebuilds the
benchmark outputs, and executes the validation gate. If a change intentionally
updates a tracked figure or generated report, include that output in the same
pull request.

## Useful contribution types

- Reproducibility fixes for ordinary laptop environments
- Additional validation checks or sensitivity analyses
- Clearer documentation of assumptions, metrics, and limitations
- New synthetic scenarios that isolate a meaningful methodological question
- Figure improvements that make tradeoffs easier to inspect

## Data and reporting boundaries

Use synthetic data or clearly documented public aggregate references. Do not
submit personal data, confidential data, credentials, or organization-specific
implementation details.

When reporting a reproducibility problem, include the environment, command,
expected behavior, observed behavior, and the smallest useful excerpt of any
error output. Remove personal paths, secrets, and unrelated system details.

## Methodology changes

For a new experiment, describe:

1. The methodological question
2. The synthetic scenario or parameter sweep
3. The evidence the experiment should produce
4. The reproducibility plan
5. The limitations that should remain visible

Avoid claims that exceed the evidence. In particular, a stress diagnostic is
not a formal privacy guarantee, and a fairness diagnostic is not proof that a
fairness problem has been solved.
