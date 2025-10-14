"""Tests for deepwave.staggered_grid."""

from unittest.mock import patch

import pytest
import torch

from deepwave.staggered_grid import set_pml_profiles


def test_set_pml_profiles_basic_functionality() -> None:
    """Test set_pml_profiles with basic functionality."""
    pml_width = [10, 10, 10, 10]
    accuracy = 4
    fd_pad = [2, 1, 2, 1]
    dt = 0.001
    grid_spacing = [10.0, 10.0]
    max_vel = 1500.0
    dtype = torch.float32
    device = torch.device("cpu")
    pml_freq = 25.0
    ny = 100
    nx = 100

    with patch("deepwave.common.setup_pml") as mock_setup_pml:
        # Configure mock_setup_pml to return dummy tensors
        # It will be called 4 times
        mock_setup_pml.side_effect = [
            (
                torch.ones(ny, dtype=dtype, device=device),
                torch.ones(ny, dtype=dtype, device=device),
            ),  # ay, by
            (
                torch.ones(ny, dtype=dtype, device=device),
                torch.ones(ny, dtype=dtype, device=device),
            ),  # ayh, byh
            (
                torch.ones(nx, dtype=dtype, device=device),
                torch.ones(nx, dtype=dtype, device=device),
            ),  # ax, bx
            (
                torch.ones(nx, dtype=dtype, device=device),
                torch.ones(nx, dtype=dtype, device=device),
            ),  # axh, bxh
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
            [ny, nx],
        )

        # Calculate pml_start values as they are in the function
        pml_start_y = [fd_pad[0] + pml_width[0], ny - 1 - fd_pad[1] - pml_width[1]]
        pml_start_x = [fd_pad[2] + pml_width[2], nx - 1 - fd_pad[3] - pml_width[3]]
        max_pml_val = max(
            pml_width[0] * grid_spacing[0],
            pml_width[1] * grid_spacing[0],
            pml_width[2] * grid_spacing[1],
            pml_width[3] * grid_spacing[1],
        )

        # Assert that setup_pml was called 4 times with correct arguments
        # Call 1: ay, by
        mock_setup_pml.assert_any_call(
            pml_width[:2],
            pml_start_y,
            pytest.approx(max_pml_val),
            dt,
            ny,
            max_vel,
            dtype,
            device,
            pml_freq,
            start=0.0,
        )
        # Call 2: ax, bx
        mock_setup_pml.assert_any_call(
            pml_width[2:],
            pml_start_x,
            pytest.approx(max_pml_val),
            dt,
            nx,
            max_vel,
            dtype,
            device,
            pml_freq,
            start=0.0,
        )
        # Call 3: ayh, byh
        mock_setup_pml.assert_any_call(
            pml_width[:2],
            pml_start_y,
            pytest.approx(max_pml_val),
            dt,
            ny,
            max_vel,
            dtype,
            device,
            pml_freq,
            start=0.5,
        )
        # Call 4: axh, bxh
        mock_setup_pml.assert_any_call(
            pml_width[2:],
            pml_start_x,
            pytest.approx(max_pml_val),
            dt,
            nx,
            max_vel,
            dtype,
            device,
            pml_freq,
            start=0.5,
        )

        # Assert the return type and shape
        assert isinstance(result, list)
        assert len(result) == 8  # 8 tensors returned
        assert all(isinstance(t, torch.Tensor) for t in result)

        # Check shapes of the returned tensors
        assert result[0].shape == (1, ny, 1)  # ay
        assert result[1].shape == (1, ny, 1)  # by
        assert result[2].shape == (1, ny, 1)  # ayh
        assert result[3].shape == (1, ny, 1)  # byh
        assert result[4].shape == (
            1,
            1,
            nx,
        )  # ax
        assert result[5].shape == (
            1,
            1,
            nx,
        )  # axh
        assert result[6].shape == (
            1,
            1,
            nx,
        )  # bx
        assert result[7].shape == (
            1,
            1,
            nx,
        )  # bxh


def test_set_pml_profiles_different_pml_width() -> None:
    """Test set_pml_profiles with different PML widths."""
    pml_width = [20, 5, 15, 0]  # Different widths, one side zero
    accuracy = 2
    fd_pad = [1, 0, 1, 0]
    dt = 0.0005
    grid_spacing = [5.0, 5.0]
    max_vel = 2000.0
    dtype = torch.float64
    device = torch.device("cpu")
    pml_freq = 10.0
    ny = 200
    nx = 150

    with patch("deepwave.common.setup_pml") as mock_setup_pml:
        # Configure mock_setup_pml to return dummy tensors
        mock_setup_pml.side_effect = [
            (
                torch.ones(ny, dtype=dtype, device=device),
                torch.ones(ny, dtype=dtype, device=device),
            ),
            (
                torch.ones(ny, dtype=dtype, device=device),
                torch.ones(ny, dtype=dtype, device=device),
            ),
            (
                torch.ones(nx, dtype=dtype, device=device),
                torch.ones(nx, dtype=dtype, device=device),
            ),
            (
                torch.ones(nx, dtype=dtype, device=device),
                torch.ones(nx, dtype=dtype, device=device),
            ),
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
            [ny, nx],
        )

        # Calculate pml_start values as they are in the function
        pml_start_y = [
            fd_pad[0] + pml_width[0],
            ny - 1 - fd_pad[1] - pml_width[1],
        ]
        pml_start_x = [
            fd_pad[2] + pml_width[2],
            nx - 1 - fd_pad[3] - pml_width[3],
        ]
        max_pml_val = max(
            pml_width[0] * grid_spacing[0],
            pml_width[1] * grid_spacing[0],
            pml_width[2] * grid_spacing[1],
            pml_width[3] * grid_spacing[1],
        )

        # Assert setup_pml calls with updated pml_start and max_pml
        mock_setup_pml.assert_any_call(
            pml_width[:2],
            pml_start_y,
            pytest.approx(max_pml_val),
            dt,
            ny,
            max_vel,
            dtype,
            device,
            pml_freq,
            start=0.0,
        )
        mock_setup_pml.assert_any_call(
            pml_width[2:],
            pml_start_x,
            pytest.approx(max_pml_val),
            dt,
            nx,
            max_vel,
            dtype,
            device,
            pml_freq,
            start=0.0,
        )
        mock_setup_pml.assert_any_call(
            pml_width[:2],
            pml_start_y,
            pytest.approx(max_pml_val),
            dt,
            ny,
            max_vel,
            dtype,
            device,
            pml_freq,
            start=0.5,
        )
        mock_setup_pml.assert_any_call(
            pml_width[2:],
            pml_start_x,
            pytest.approx(max_pml_val),
            dt,
            nx,
            max_vel,
            dtype,
            device,
            pml_freq,
            start=0.5,
        )

        # Assert the return type and shape
        assert isinstance(result, list)
        assert len(result) == 8
        assert all(isinstance(t, torch.Tensor) for t in result)

        for i in range(4):
            assert result[i].shape == (1, ny, 1)
        for i in range(4, 8):
            assert result[i].shape == (
                1,
                1,
                nx,
            )


def test_set_pml_profiles_edge_cases() -> None:
    """Test set_pml_profiles with edge cases (zero PML width)."""
    # Test with zero pml_width everywhere
    pml_width = [0, 0, 0, 0]
    accuracy = 4
    fd_pad = [2, 1, 2, 1]
    dt = 0.001
    grid_spacing = [10.0, 10.0]
    max_vel = 1500.0
    dtype = torch.float32
    device = torch.device("cpu")
    pml_freq = 25.0
    ny = 10
    nx = 10

    with patch("deepwave.common.setup_pml") as mock_setup_pml:
        mock_setup_pml.side_effect = [
            (
                torch.zeros(ny, dtype=dtype, device=device),
                torch.zeros(ny, dtype=dtype, device=device),
            ),
            (
                torch.zeros(ny, dtype=dtype, device=device),
                torch.zeros(ny, dtype=dtype, device=device),
            ),
            (
                torch.zeros(nx, dtype=dtype, device=device),
                torch.zeros(nx, dtype=dtype, device=device),
            ),
            (
                torch.zeros(nx, dtype=dtype, device=device),
                torch.zeros(nx, dtype=dtype, device=device),
            ),
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
            [ny, nx],
        )

        # Calculate pml_start values as they are in the function
        pml_start_y = [
            fd_pad[0] + pml_width[0],
            ny - 1 - fd_pad[1] - pml_width[1],
        ]
        pml_start_x = [
            fd_pad[2] + pml_width[2],
            nx - 1 - fd_pad[3] - pml_width[3],
        ]
        max_pml_val = max(
            pml_width[0] * grid_spacing[0],
            pml_width[1] * grid_spacing[0],
            pml_width[2] * grid_spacing[1],
            pml_width[3] * grid_spacing[1],
        )

        mock_setup_pml.assert_any_call(
            [0, 0],
            pml_start_y,
            pytest.approx(max_pml_val),
            dt,
            ny,
            max_vel,
            dtype,
            device,
            pml_freq,
            start=0.0,
        )
        mock_setup_pml.assert_any_call(
            [0, 0],
            pml_start_x,
            pytest.approx(max_pml_val),
            dt,
            nx,
            max_vel,
            dtype,
            device,
            pml_freq,
            start=0.0,
        )
        mock_setup_pml.assert_any_call(
            [0, 0],
            pml_start_y,
            pytest.approx(max_pml_val),
            dt,
            ny,
            max_vel,
            dtype,
            device,
            pml_freq,
            start=0.5,
        )
        mock_setup_pml.assert_any_call(
            [0, 0],
            pml_start_x,
            pytest.approx(max_pml_val),
            dt,
            nx,
            max_vel,
            dtype,
            device,
            pml_freq,
            start=0.5,
        )

        assert all(torch.all(t == 0) for t in result)  # All profiles should be zero

    # Test with minimal grid size where PML might overlap
    pml_width = [1, 1, 1, 1]
    fd_pad = [0, 0, 0, 0]
    ny = 2  # Minimal size
    nx = 2

    with patch("deepwave.common.setup_pml") as mock_setup_pml:
        mock_setup_pml.side_effect = [
            (
                torch.ones(ny, dtype=dtype, device=device),
                torch.ones(ny, dtype=dtype, device=device),
            ),
            (
                torch.ones(ny, dtype=dtype, device=device),
                torch.ones(ny, dtype=dtype, device=device),
            ),
            (
                torch.ones(nx, dtype=dtype, device=device),
                torch.ones(nx, dtype=dtype, device=device),
            ),
            (
                torch.ones(nx, dtype=dtype, device=device),
                torch.ones(nx, dtype=dtype, device=device),
            ),
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
            [ny, nx],
        )

        # Calculate pml_start values as they are in the function
        pml_start_y = [
            fd_pad[0] + pml_width[0],
            ny - 1 - fd_pad[1] - pml_width[1],
        ]
        pml_start_x = [
            fd_pad[2] + pml_width[2],
            nx - 1 - fd_pad[3] - pml_width[3],
        ]
        max_pml_val = max(
            pml_width[0] * grid_spacing[0],
            pml_width[1] * grid_spacing[0],
            pml_width[2] * grid_spacing[1],
            pml_width[3] * grid_spacing[1],
        )

        mock_setup_pml.assert_any_call(
            pml_width[:2],
            pml_start_y,
            pytest.approx(max_pml_val),
            dt,
            ny,
            max_vel,
            dtype,
            device,
            pml_freq,
            start=0.0,
        )
        mock_setup_pml.assert_any_call(
            pml_width[2:],
            pml_start_x,
            pytest.approx(max_pml_val),
            dt,
            nx,
            max_vel,
            dtype,
            device,
            pml_freq,
            start=0.0,
        )
        mock_setup_pml.assert_any_call(
            pml_width[:2],
            pml_start_y,
            pytest.approx(max_pml_val),
            dt,
            ny,
            max_vel,
            dtype,
            device,
            pml_freq,
            start=0.5,
        )
        mock_setup_pml.assert_any_call(
            pml_width[2:],
            pml_start_x,
            pytest.approx(max_pml_val),
            dt,
            nx,
            max_vel,
            dtype,
            device,
            pml_freq,
            start=0.5,
        )

        assert len(result) == 8
        for i in range(4):
            assert result[i].shape == (1, ny, 1)
        for i in range(4, 8):
            assert result[i].shape == (
                1,
                1,
                nx,
            )
