# Policy and National-Relevance Context

This note records the public, verifiable context that motivates FairPrivacySignal. It
documents why privacy-driven signal loss is a broad U.S. problem across sectors, using
government and public sources. It does **not** claim that this repository is deployed in
any of the systems below, and it does not rely on any confidential or employer-specific
information. FairPrivacySignal is a public, synthetic, methodological contribution; the
sources here establish the importance of the *problem* the method studies.

## 1. Privacy-driven signal loss is an economy-wide U.S. phenomenon

- **State privacy law is broad and expanding.** Roughly twenty U.S. states have
  comprehensive consumer-privacy laws in effect as of the start of 2026 (with more
  enacted and phasing in), each imposing data-minimization, retention, consent, and
  opt-out requirements that reduce the individual-level behavioral signal available to
  data-driven systems. See the IAPP U.S. State Privacy Legislation Tracker.
- **Platform privacy changes have removed measurable economic value.** After Apple's
  App Tracking Transparency (ATT) moved mobile tracking to opt-in, opt-in rates fell to
  roughly 15-25%, and major platforms lost on the order of \$10 billion in advertising
  revenue in the second half of 2021 alone. An economic analysis hosted by the U.S.
  Federal Trade Commission estimated a U.S. impact on the order of \$15 billion.
  Marketers commonly report seeing only ~40-60% of real conversions after signal loss.
  These figures quantify the same technical problem the benchmark studies: useful signal
  disappears, and systems must remain effective without it.

## 2. The problem is recognized in U.S. federal AI risk guidance

- The **NIST AI Risk Management Framework (AI RMF)** treats privacy and data
  minimization as core components of trustworthy AI, and is referenced across U.S.
  federal agencies, including the FTC, CFPB, FDA, SEC, and EEOC. The U.S. Treasury's
  Financial Services AI RMF (released February 2026) translates these principles into
  hundreds of control objectives for financial institutions.
- The practical implication is that maintaining model utility, evaluation reliability,
  and stability *under* privacy constraints is now an expected engineering competency,
  not an optional one. That is precisely the capability FairPrivacySignal studies and
  measures.

## 3. The problem bears on U.S. AI competitiveness

- Current U.S. federal AI policy direction (2025) emphasizes maintaining American
  leadership in AI. Keeping large-scale AI systems effective and reliable as usable
  signal shrinks, without re-introducing restricted raw personal data, is part of
  sustaining that competitiveness. Methods that recover useful learning information
  under privacy constraints reduce wasted experimentation and protect model quality,
  which supports national AI capability.

## 4. The pattern is cross-sector, not advertising-specific

The benchmark deliberately uses neutral entities and synthetic data so the method can be
examined independently of any one application. The same system shape -- ranking or
matching candidates from a mix of permitted context and restricted historical signal --
appears in healthcare outreach, education, public and nonprofit services, financial
access, and marketplaces (see the "Where this pattern appears" table in the README).
This cross-sector reach is what makes the underlying method relevant beyond any single
company or industry.

## 5. How FairPrivacySignal relates to this context

FairPrivacySignal is a public, openly licensed, DOI-archived reference implementation
and benchmark. It makes a privacy-aware signal-recovery method and its evaluation
**inspectable and reusable by anyone**, rather than internal to one company. Its
contribution is methodological: a reusable evaluation protocol, a recovery method, and a
set of controls for testing *why* recovery occurs. The synthetic design is intentional --
it enables fully public, non-confidential, reproducible study of a real and documented
problem. What the synthetic benchmark does and does not claim is stated in
[\`limitations.md\`](limitations.md).

## References

- IAPP -- U.S. State Privacy Legislation Tracker: https://iapp.org/resources/article/us-state-privacy-legislation-tracker
- MultiState -- comprehensive state privacy laws taking effect in 2026: https://www.multistate.us/insider/2026/2/4/all-of-the-comprehensive-privacy-laws-that-take-effect-in-2026
- FTC-hosted economic analysis of opt-in vs. opt-out (ATT impact): https://www.ftc.gov/system/files/ftc_gov/pdf/3-Skiera-Economic-Impact-of-Opt-in-versus-Opt-out-Requirements-for-Personal-Data-Usage.pdf
- NIST AI Risk Management Framework: https://www.nist.gov/itl/ai-risk-management-framework
- 2025 U.S. federal AI policy direction (American leadership in AI): https://www.whitehouse.gov/presidential-actions/2025/01/removing-barriers-to-american-leadership-in-artificial-intelligence/

*All sources are public. This document contains no confidential or employer-specific
information. Figures are summarized from the cited public sources and should be quoted
with their attribution.*
