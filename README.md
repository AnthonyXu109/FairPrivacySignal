# FairPrivacySignal

FairPrivacySignal is a public, non-confidential synthetic-data benchmark for studying privacy-preserving and fairness-aware AI ranking and matching systems under signal loss.

Many AI systems rely on user-level behavioral signals to decide which services, resources, or recommendations should be shown to people or communities. Privacy requirements, consent restrictions, data minimization rules, and reduced access to tracking signals can remove or weaken those signals. While these protections are important, signal loss can also reduce model utility and may disproportionately affect small, low-signal, or underserved participants.

This project demonstrates a simple, reproducible public-service outreach scenario: matching communities or households to relevant public services such as preventive health outreach, food assistance, housing support, job training, and education resources. The project uses synthetic data, optionally calibrated with public aggregate datasets, to show how privacy-preserving transformations and fairness-aware evaluation can reduce reliance on raw personal data while preserving useful ranking performance.

The same technical pattern can apply to public agencies, healthcare outreach, nonprofit service delivery, education programs, local marketplaces, and small-business discovery systems.

## What this project demonstrates

- Synthetic public-service ranking and matching data generation
- Privacy-driven signal loss simulation
- Consent-aware and policy-aware feature suppression
- Cohort aggregation and k-thresholding
- Differential-privacy-style noise for aggregate features
- Contextual and geography-level signals
- Utility metrics such as AUC and NDCG@K
- Fairness metrics for low-signal or underserved participants
- Privacy exposure scoring
- Reproducible notebooks and visualizations

## Non-confidentiality statement

This project uses only synthetic data and public aggregate references. It does not use, disclose, depend on, or derive from any Meta confidential information, internal datasets, internal architecture, internal metrics, product-specific implementation details, or proprietary code.

This repository is an educational and research-oriented synthetic benchmark. It is not affiliated with, endorsed by, or derived from Meta Platforms, Inc.

## Intended use

FairPrivacySignal is intended to illustrate engineering patterns for evaluating privacy, utility, and fairness tradeoffs in AI ranking and matching systems. It is not a production privacy system and does not provide formal privacy guarantees unless explicitly state.
