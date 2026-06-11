from __future__ import annotations

from mxalloy.runtime import (
    ActivationOption,
    ComponentSpec,
    DeviceProfile,
    WorkloadSpec,
    estimate_peak_gb,
    plan_execution,
)


def _device(working_set_gb: float) -> DeviceProfile:
    return DeviceProfile(
        machine="arm64",
        processor="arm",
        is_apple_silicon=True,
        total_memory_gb=working_set_gb + 2.0,
        working_set_gb=working_set_gb,
        os_reserve_gb=1.5,
        safety_margin_gb=0.5,
    )


def _klein_spec() -> WorkloadSpec:
    return WorkloadSpec(
        name="klein-test",
        components=(
            ComponentSpec(
                name="all",
                precision_memory_gb={"bf16": 17.9, "int8": 8.56, "int4": 4.54},
            ),
        ),
        activation_options=(
            ActivationOption("resident", activation_peak_gb=10.1, vae_tile_latent=128),
            ActivationOption("survival", activation_peak_gb=6.8, vae_tile_latent=64),
        ),
        default_steps=4,
    )


def test_planner_keeps_klein_on_int4_resident_for_18gb_class() -> None:
    strategy = plan_execution(_device(16.0), _klein_spec())
    assert strategy.fits
    assert strategy.precision == "int4"
    assert strategy.memory_mode == "resident"
    assert strategy.vae_tile_latent == 128


def test_planner_chooses_bf16_when_it_fits() -> None:
    strategy = plan_execution(_device(32.0), _klein_spec())
    assert strategy.fits
    assert strategy.precision == "bf16"
    assert strategy.memory_mode == "resident"


def test_planner_can_shrink_memory_mode_for_forced_precision() -> None:
    strategy = plan_execution(_device(16.0), _klein_spec(), requested_precision="int8")
    assert strategy.fits
    assert strategy.precision == "int8"
    assert strategy.memory_mode == "survival"


def test_planner_reports_when_no_candidate_fits() -> None:
    strategy = plan_execution(_device(8.0), _klein_spec())
    assert not strategy.fits
    assert strategy.precision == "int4"
    assert strategy.memory_mode == "survival"
    assert "estimated_peak_exceeds_working_set" in strategy.warnings


def test_component_param_estimate_is_available_without_measured_memory() -> None:
    spec = WorkloadSpec(
        name="param-estimate",
        components=(ComponentSpec(name="tiny", params=1_000_000_000),),
        activation_options=(ActivationOption("resident", activation_peak_gb=0.0),),
        default_steps=1,
    )
    assert estimate_peak_gb(spec, "bf16", spec.activation_options[0]) > 0
