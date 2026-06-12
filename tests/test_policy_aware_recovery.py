import numpy as np

from fairprivacysignal.data_generator import generate_all
from fairprivacysignal.policy_aware_recovery import (
    DEFAULT_RECOVERY_SPEC,
    RECONSTRUCTED_SIGNAL,
    SignalRecoverySpec,
    attach_reconstructed_signal,
    cross_fit_signal_reconstruction,
)
from fairprivacysignal.privacy_recovery import split_household_events
from fairprivacysignal.signal_loss import apply_signal_loss


def test_signal_reconstruction_does_not_read_hidden_test_signal() -> None:
    _, _, _, events = generate_all(
        n_communities=12,
        n_households=300,
        seed=17,
    )
    train, test = split_household_events(events)

    _, reference_proxy, diagnostics = cross_fit_signal_reconstruction(
        train,
        test,
        seed=17,
        n_splits=3,
    )
    altered_test = test.copy()
    altered_test["historical_service_engagement_count"] += 1000
    _, altered_proxy, _ = cross_fit_signal_reconstruction(
        train,
        altered_test,
        seed=17,
        n_splits=3,
    )

    assert np.allclose(
        reference_proxy[RECONSTRUCTED_SIGNAL],
        altered_proxy[RECONSTRUCTED_SIGNAL],
    )
    assert diagnostics["reconstruction_folds"] == 3
    assert np.isfinite(diagnostics["reconstruction_oof_mae"])


def test_reconstruction_replaces_only_unavailable_serving_signal() -> None:
    _, _, _, events = generate_all(
        n_communities=12,
        n_households=300,
        seed=23,
    )
    train, test = split_household_events(events)
    _, test_proxy, _ = cross_fit_signal_reconstruction(
        train,
        test,
        seed=23,
        n_splits=3,
    )
    restricted = apply_signal_loss(test, "policy_restricted")
    transformed = attach_reconstructed_signal(restricted, test_proxy)
    available = transformed["behavioral_available"].astype(bool)

    assert np.allclose(
        transformed.loc[available, RECONSTRUCTED_SIGNAL],
        transformed.loc[
            available,
            "available_historical_service_engagement_count",
        ],
    )
    assert (
        transformed.loc[~available, "signal_reconstruction_applied"] == 1.0
    ).all()
    assert np.isfinite(
        transformed.loc[~available, RECONSTRUCTED_SIGNAL]
    ).all()


def test_recovery_spec_supports_domain_specific_column_names() -> None:
    _, _, _, events = generate_all(
        n_communities=10,
        n_households=200,
        seed=31,
    )
    train, test = split_household_events(events)
    renamed_columns = {
        "event_id": "decision_id",
        "household_id": "entity_id",
        "historical_service_engagement_count": "restricted_signal",
        "available_historical_service_engagement_count": "permitted_signal",
        "behavioral_available": "signal_available",
    }
    spec = SignalRecoverySpec(
        event_id_column="decision_id",
        group_column="entity_id",
        raw_signal_column="restricted_signal",
        available_signal_column="permitted_signal",
        availability_column="signal_available",
        reconstructed_signal_column="recovered_signal",
        context_numeric_features=DEFAULT_RECOVERY_SPEC.context_numeric_features,
        categorical_features=DEFAULT_RECOVERY_SPEC.categorical_features,
        signal_lower_bound=0.0,
        signal_upper_bound=50.0,
    )
    renamed_train = train.rename(columns=renamed_columns)
    renamed_test = test.rename(columns=renamed_columns)
    _, proxy, _ = cross_fit_signal_reconstruction(
        renamed_train,
        renamed_test,
        seed=31,
        n_splits=2,
        spec=spec,
    )
    masked = apply_signal_loss(test, "severe_signal_loss").rename(
        columns=renamed_columns
    )
    transformed = attach_reconstructed_signal(masked, proxy, spec=spec)

    assert "recovered_signal" in transformed
    assert np.isfinite(transformed["recovered_signal"]).all()
