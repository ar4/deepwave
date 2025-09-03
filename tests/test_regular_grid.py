import unittest.mock
from unittest.mock import patch

import pytest
import torch

from deepwave.regular_grid import set_pml_profiles


def test_set_pml_profiles_basic_functionality():
    pml_width = [10, 10, 10, 10]
    accuracy = 4
    fd_pad = [0, 0, 0, 0]
    dt = 0.001
    grid_spacing = [10.0, 10.0]
    max_vel = 1500.0
    dtype = torch.float32
    device = torch.device("cpu")
    pml_freq = 25.0
    ny = 100
    nx = 100

    with patch("deepwave.common.setup_pml") as mock_setup_pml:
        with patch("deepwave.common.diff") as mock_diff:
            # Configure mock_setup_pml to return dummy tensors
            mock_setup_pml.side_effect = [
                (
                    torch.ones(ny, dtype=dtype, device=device),
                    torch.ones(ny, dtype=dtype, device=device),
                ),  # For ay, by
                (
                    torch.ones(nx, dtype=dtype, device=device),
                    torch.ones(nx, dtype=dtype, device=device),
                ),  # For ax, bx
            ]

            # Configure mock_diff to return dummy tensors
            mock_diff.side_effect = [
                torch.ones(ny, dtype=dtype, device=device),  # For dbydy
                torch.ones(nx, dtype=dtype, device=device),  # For dbxdx
            ]

            result = set_pml_profiles(
                pml_width,
                accuracy,
                fd_pad,
                dt,
                grid_spacing,
                max_vel,
                dtype,
                device,
                pml_freq,
                ny,
                nx,
            )

            # Assert that setup_pml was called twice with correct arguments
            mock_setup_pml.assert_any_call(
                pml_width[:2],  # [10, 10]
                [
                    fd_pad[0] + pml_width[0],
                    ny - 1 - fd_pad[1] - pml_width[1],
                ],  # [10, 89]
                pytest.approx(
                    max(
                        pml_width[0] * grid_spacing[0],
                        pml_width[1] * grid_spacing[0],
                        pml_width[2] * grid_spacing[1],
                        pml_width[3] * grid_spacing[1],
                    ),
                ),
                dt,
                ny,
                max_vel,
                dtype,
                device,
                pml_freq,
            )
            mock_setup_pml.assert_any_call(
                pml_width[2:],  # [10, 10]
                [
                    fd_pad[2] + pml_width[2],
                    nx - 1 - fd_pad[3] - pml_width[3],
                ],  # [10, 89]
                pytest.approx(
                    max(
                        pml_width[0] * grid_spacing[0],
                        pml_width[1] * grid_spacing[0],
                        pml_width[2] * grid_spacing[1],
                        pml_width[3] * grid_spacing[1],
                    ),
                ),
                dt,
                nx,
                max_vel,
                dtype,
                device,
                pml_freq,
            )

            # Assert that diff was called twice with correct arguments
            mock_diff.assert_any_call(unittest.mock.ANY, accuracy, grid_spacing[0])
            mock_diff.assert_any_call(unittest.mock.ANY, accuracy, grid_spacing[1])

            # Assert the return type and shape
            assert isinstance(result, list)
            assert len(result) == 6
            assert all(isinstance(t, torch.Tensor) for t in result)

            # Check shapes of the returned tensors after unsqueezing
            assert result[0].shape == (1, ny, 1)  # ay
            assert result[1].shape == (1, 1, nx)  # ax
            assert result[2].shape == (1, ny, 1)  # by
            assert result[3].shape == (1, 1, nx)  # bx
            assert result[4].shape == (1, ny, 1)  # dbydy
            assert result[5].shape == (1, 1, nx)  # dbxdx


def test_set_pml_profiles_different_pml_width():
    pml_width = [20, 5, 15, 0]  # Different widths, one side zero
    accuracy = 2
    fd_pad = [1, 1, 1, 1]
    dt = 0.0005
    grid_spacing = [5.0, 5.0]
    max_vel = 2000.0
    dtype = torch.float64
    device = torch.device("cpu")  # Changed from cuda to cpu
    pml_freq = 10.0
    ny = 200
    nx = 150

    with patch("deepwave.common.setup_pml") as mock_setup_pml:
        with patch("deepwave.common.diff") as mock_diff:
            # Configure mock_setup_pml to return dummy tensors
            mock_setup_pml.side_effect = [
                (
                    torch.ones(ny, dtype=dtype, device=device),
                    torch.ones(ny, dtype=dtype, device=device),
                ),
                (
                    torch.ones(nx, dtype=dtype, device=device),
                    torch.ones(nx, dtype=dtype, device=device),
                ),
            ]

            # Configure mock_diff to return dummy tensors
            mock_diff.side_effect = [
                torch.ones(ny, dtype=dtype, device=device),
                torch.ones(nx, dtype=dtype, device=device),
            ]

            result = set_pml_profiles(
                pml_width,
                accuracy,
                fd_pad,
                dt,
                grid_spacing,
                max_vel,
                dtype,
                device,
                pml_freq,
                ny,
                nx,
            )

            # Assert setup_pml calls with updated pml_start and max_pml
            mock_setup_pml.assert_any_call(
                pml_width[:2],  # [20, 5]
                [
                    fd_pad[0] + pml_width[0],
                    ny - 1 - fd_pad[1] - pml_width[1],
                ],  # [21, 200 - 1 - 1 - 5 = 193]
                pytest.approx(
                    max(
                        pml_width[0] * grid_spacing[0],
                        pml_width[1] * grid_spacing[0],
                        pml_width[2] * grid_spacing[1],
                        pml_width[3] * grid_spacing[1],
                    ),
                ),
                dt,
                ny,
                max_vel,
                dtype,
                device,
                pml_freq,
            )
            mock_setup_pml.assert_any_call(
                pml_width[2:],  # [15, 0]
                [
                    fd_pad[2] + pml_width[2],
                    nx - 1 - fd_pad[3] - pml_width[3],
                ],  # [16, 150 - 1 - 1 - 0 = 148]
                pytest.approx(
                    max(
                        pml_width[0] * grid_spacing[0],
                        pml_width[1] * grid_spacing[0],
                        pml_width[2] * grid_spacing[1],
                        pml_width[3] * grid_spacing[1],
                    ),
                ),
                dt,
                nx,
                max_vel,
                dtype,
                device,
                pml_freq,
            )

            # Assert the return type and shape
            assert isinstance(result, list)
            assert len(result) == 6
            assert all(isinstance(t, torch.Tensor) for t in result)

            assert result[0].shape == (1, ny, 1)
            assert result[1].shape == (1, 1, nx)


def test_set_pml_profiles_edge_cases():
    # Test with zero pml_width everywhere
    pml_width = [0, 0, 0, 0]
    accuracy = 4
    fd_pad = [0, 0, 0, 0]
    dt = 0.001
    grid_spacing = [10.0, 10.0]
    max_vel = 1500.0
    dtype = torch.float32
    device = torch.device("cpu")
    pml_freq = 25.0
    ny = 10
    nx = 10

    with patch("deepwave.common.setup_pml") as mock_setup_pml:
        with patch("deepwave.common.diff") as mock_diff:
            mock_setup_pml.side_effect = [
                (
                    torch.zeros(ny, dtype=dtype, device=device),
                    torch.zeros(ny, dtype=dtype, device=device),
                ),
                (
                    torch.zeros(nx, dtype=dtype, device=device),
                    torch.zeros(nx, dtype=dtype, device=device),
                ),
            ]
            mock_diff.side_effect = [
                torch.zeros(ny, dtype=dtype, device=device),
                torch.zeros(nx, dtype=dtype, device=device),
            ]

            result = set_pml_profiles(
                pml_width,
                accuracy,
                fd_pad,
                dt,
                grid_spacing,
                max_vel,
                dtype,
                device,
                pml_freq,
                ny,
                nx,
            )

            # setup_pml should still be called, but with pml_width of 0
            mock_setup_pml.assert_any_call(
                [0, 0],
                [0, ny - 1],
                pytest.approx(0.0),
                dt,
                ny,
                max_vel,
                dtype,
                device,
                pml_freq,
            )
            mock_setup_pml.assert_any_call(
                [0, 0],
                [0, nx - 1],
                pytest.approx(0.0),
                dt,
                nx,
                max_vel,
                dtype,
                device,
                pml_freq,
            )

            assert all(torch.all(t == 0) for t in result)  # All profiles should be zero

        # Test with minimal grid size where PML might overlap
        pml_width = [1, 1, 1, 1]
        fd_pad = [0, 0, 0, 0]
        ny = 2  # Minimal size
        nx = 2

        with patch("deepwave.common.setup_pml") as mock_setup_pml:
            with patch("deepwave.common.diff") as mock_diff:
                mock_setup_pml.side_effect = [
                    (
                        torch.ones(ny, dtype=dtype, device=device),
                        torch.ones(ny, dtype=dtype, device=device),
                    ),
                    (
                        torch.ones(nx, dtype=dtype, device=device),
                        torch.ones(nx, dtype=dtype, device=device),
                    ),
                ]
                mock_diff.side_effect = [
                    torch.ones(ny, dtype=dtype, device=device),
                    torch.ones(nx, dtype=dtype, device=device),
                ]

                result = set_pml_profiles(
                    pml_width,
                    accuracy,
                    fd_pad,
                    dt,
                    grid_spacing,
                    max_vel,
                    dtype,
                    device,
                    pml_freq,
                    ny,
                    nx,
                )

                # Check pml_start values for minimal grid
                mock_setup_pml.assert_any_call(
                    pml_width[:2],
                    [
                        fd_pad[0] + pml_width[0],
                        ny - 1 - fd_pad[1] - pml_width[1],
                    ],  # [1, 2 - 1 - 0 - 1 = 0]
                    pytest.approx(
                        max(
                            pml_width[0] * grid_spacing[0],
                            pml_width[1] * grid_spacing[0],
                            pml_width[2] * grid_spacing[1],
                            pml_width[3] * grid_spacing[1],
                        ),
                    ),
                    dt,
                    ny,
                    max_vel,
                    dtype,
                    device,
                    pml_freq,
                )
                mock_setup_pml.assert_any_call(
                    pml_width[2:],
                    [
                        fd_pad[2] + pml_width[2],
                        nx - 1 - fd_pad[3] - pml_width[3],
                    ],  # [1, 2 - 1 - 0 - 1 = 0]
                    pytest.approx(
                        max(
                            pml_width[0] * grid_spacing[0],
                            pml_width[1] * grid_spacing[0],
                            pml_width[2] * grid_spacing[1],
                            pml_width[3] * grid_spacing[1],
                        ),
                    ),
                    dt,
                    nx,
                    max_vel,
                    dtype,
                    device,
                    pml_freq,
                )

                assert result[0].shape == (1, ny, 1)
                assert result[1].shape == (1, 1, nx)
