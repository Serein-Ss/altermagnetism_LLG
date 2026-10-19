"""Explicit legacy time scaling and exploratory, time-homogeneous window rollout."""
import torch
from scripts.model.gate_f import sample
from tests.experiments.conditional_source import conditional_source


def crop_training_windows(paths,frames,generator):
    """Uniform start per selected parent path; never cross path/split boundaries."""
    if not 2<=frames<=paths.shape[1]:raise ValueError('invalid training window')
    if frames==paths.shape[1]:return paths
    starts=torch.randint(paths.shape[1]-frames+1,(len(paths),),device=paths.device,generator=generator)
    return paths[torch.arange(len(paths),device=paths.device)[:,None],
                 starts[:,None]+torch.arange(frames,device=paths.device)[None]]


def normalized_window(initial, frames=201, physical_window=16000., training_window=16000.):
    if frames < 2 or physical_window <= 0 or training_window <= 0:
        raise ValueError('positive windows and at least two frames required')
    return torch.linspace(0., physical_window / training_window, frames,
                          device=initial.device, dtype=initial.dtype)[None].expand(len(initial), -1)


@torch.no_grad()
def sample_chunks(net, initial, condition, time, graph, axis, generator, *, chunks, steps=32,source_scale=None):
    """Constant-condition autonomous dynamics only; NOT a validated long-time model.

    Reuse the trained relative-time window, draw a fresh source per chunk and
    anchor each chunk to the previous endpoint. Distributional drift remains to
    be tested; continuity alone does not establish the correct transition kernel.
    """
    if chunks < 1:
        raise ValueError('positive chunk count required')
    if time.shape[0] != len(initial) or time.shape[1] < 2 or not torch.all(time[:, 0] == 0):
        raise ValueError('relative window starting at zero required')
    paths = []
    current = initial
    for index in range(chunks):
        source=None if source_scale is None else conditional_source(current,time.shape[1],generator,source_scale)
        path = sample(net, current, condition, time, graph, axis, generator, steps=steps,source=source)
        if not torch.equal(path[:, 0], current):
            raise RuntimeError('chunk initial condition violated')
        paths.append(path if index == 0 else path[:, 1:])
        current = path[:, -1]
    return torch.cat(paths, dim=1)
