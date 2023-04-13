import torch
import deepwave
from deepwave.location_interpolation import Hicks
from test_scalar import direct_2d_approx


def test_monopole(c=1500, freq=25, dx=(5, 5), dt=0.001,
                  nx=(50, 50), nt=200, device=None,
                  dtype=torch.double):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    peak_time = 1.5/freq
    model = torch.ones(*nx, dtype=dtype, device=device) * c
    dx = torch.tensor(dx)
    source_amplitudes = (deepwave.wavelets.ricker(freq, nt, dt, peak_time,
                                                  dtype=dtype)
                         .reshape(1, 1, -1).to(device).repeat(20, 1, 1))
    source_locations = torch.zeros(20, 1, 2, device=device)
    source_locations[..., 0] = (torch.linspace(20.0, 20.9, 10, device=device)
                                .reshape(10, 1).repeat(2, 1))
    source_locations[:10, :, 1] = 25.0
    source_locations[10:, :, 1] = 25.25
    receiver_locations = torch.zeros(20, 20, 2, device=device)
    receiver_locations[..., 0] = (torch.linspace(30.0, 30.9, 10, device=device)
                                  .reshape(1, -1).repeat(20, 2))
    receiver_locations[:10, :, 1] = 25.0
    receiver_locations[10:, :, 1] = 25.25
    hicks_source = Hicks(source_locations, dtype=dtype)
    hicks_source_locations = hicks_source.hicks_locations
    hicks_amplitudes = hicks_source.source(source_amplitudes)
    hicks_receiver = Hicks(receiver_locations, dtype=dtype)
    hicks_receiver_locations = hicks_receiver.hicks_locations
    o = deepwave.scalar(model, dx, dt,
                        source_amplitudes=hicks_amplitudes,
                        source_locations=hicks_source_locations,
                        receiver_locations=hicks_receiver_locations)
    receiver_amplitudes = hicks_receiver.receiver(o[-1])
    for i, source_location in enumerate(source_locations):
        for j, receiver_location in enumerate(receiver_locations[i]):
            e = direct_2d_approx(receiver_location.float().cpu(),
                                 source_location[0].float().cpu(),
                                 dx, dt, c,
                                 -source_amplitudes[i].flatten().cpu())
            assert torch.allclose(receiver_amplitudes[i, j].cpu(), e,
                                  atol=0.025)


def test_shot_idxs(n_shots=10, n_per_shot=3, nt=5, device=None,
                   dtype=torch.double):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1)
    locations = torch.rand(n_shots, n_per_shot, 2, dtype=dtype,
                           device=device)*10
    amplitudes = torch.randn(n_shots, n_per_shot, nt, dtype=dtype,
                             device=device)
    hicks = Hicks(locations, dtype=dtype)
    hicks_amplitudes_fwd = hicks.source(amplitudes)
    shot_idxs = torch.tensor([n_shots//2])
    hicks_amplitudes_half = hicks.source(amplitudes[shot_idxs], shot_idxs)
    assert torch.allclose(hicks_amplitudes_fwd[shot_idxs],
                          hicks_amplitudes_half)
    shot_idxs = torch.arange(n_shots).flip(0)
    hicks_amplitudes_bwd = hicks.source(amplitudes.flip(0), shot_idxs)
    assert torch.allclose(hicks_amplitudes_fwd.flip(0),
                          hicks_amplitudes_bwd)
