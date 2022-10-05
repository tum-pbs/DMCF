import numpy as np
from tqdm import trange


def sample_sphere(r, res, sres, dim=2):
    rg = np.linspace(0.5, res - 0.5, int((res - 2) * sres))
    data = np.stack(np.meshgrid(rg,
                                rg if dim > 1 else [0.0],
                                rg if dim > 2 else [0.0],
                                indexing='ij'),
                    axis=-1)
    center = [
        res / 2, res / 2 if dim > 1 else 0.0, res / 2 if dim > 2 else 0.0
    ]
    data = data[np.where(np.linalg.norm(data - center, axis=-1) < r)]
    return data.reshape(-1, 3)


def step(pos, vel, grav, dt, mode=0):
    if mode == 0:
        vel1 = vel + dt * np.array([0.0, grav, 0.0])
        pos1 = pos + dt * vel1
    else:
        vel1 = vel + dt * np.array([0.0, grav, 0.0])
        pos1 = pos + dt * vel + (vel + vel1) / 2

    return pos1, vel1


def gen_dict(pos, vel, idx, res, grav):
    d = []
    for t in range(len(pos)):
        frame = {
            'frame_id': t,
            'scene_id': 'sim_%04d' % idx,
            'grav': np.array([0.0, grav, 0.0])
        }
        frame['pos'] = pos[t]
        frame['vel'] = vel[t]
        frame['box'] = np.ones((1, 3)) * res * 2
        frame['box_normals'] = np.zeros((1, 3))

        frame['pos'] /= res
        frame['vel'] /= res
        frame['box'] /= res
        frame['grav'] /= res

        d.append(frame)

    return d


def gen_data(data_cnt=1,
             timesteps=100,
             res=100,
             dim=2,
             radius=20,
             dt=0.01,
             gravity=-10.0,
             mode=0):

    data = []

    gravity *= res

    for d in trange(data_cnt):
        points = sample_sphere(radius, res, 0.5, dim)
        vel = np.zeros_like(points)
        pos = [points]
        vel = [vel]
        for t in range(timesteps):
            p, v = step(pos[t], vel[t], gravity, dt, mode)
            pos.append(p)
            vel.append(v)

        data.append(gen_dict(pos, vel, d, res, gravity))

    return data
