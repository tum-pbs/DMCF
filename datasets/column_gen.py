import numpy as np
from tqdm import trange


class SPH1D:
    def __init__(self,
                 radius=0.25,
                 mass=1.0,
                 dens=None,
                 stiffness=10.0,
                 visc=1e-4,
                 gravity=-10.0):

        self.h = 4 * radius
        self.mass = mass
        self.rest_dens = self.mass / (radius * 2.0) if dens is None else dens
        self.stiffness = stiffness
        self.visc = visc
        self.gravity = gravity

        self.setup(1)

    def setup(self, cnt, bcnt=2, rnd=0.0, offset=0.0):
        self.bcnt = bcnt

        self.particles = np.zeros((cnt + bcnt, 3), dtype="float32")
        self.particles[:, 0] = np.arange(self.particles.shape[0],
                                         dtype="float32") * self.h * 0.5
        if rnd > 0:
            self.particles[bcnt:,
                           0] += np.random.normal(scale=rnd, size=cnt) * self.h
        if offset > 0:
            self.particles[bcnt:, 0] += offset
        self.particles[:, 2] = self.mass

    def cnt_nn(self):
        def nn(q):
            return np.where(q <= 1, 1, 0)

        dist_mat = np.abs(
            np.expand_dims(self.particles[:, 0], axis=1) -
            np.expand_dims(self.particles[:, 0], axis=0)) / self.h
        return np.sum(nn(dist_mat), axis=1)

    def compute_val(self, dens=[1.0], val=[1.0], particles=None, h=None):
        if particles is None:
            particles = self.particles
        if h is None:
            h = self.h

        def cubic_spline(q):
            return 4 / (3 * h) * np.where(
                q <= 1,
                np.where(q <= 0.5, 6 * (q**3 - q**2) + 1, 2 * (1 - q)**3), 0)

        dist_mat = np.abs(
            np.expand_dims(particles[:, 0], axis=1) -
            np.expand_dims(particles[:, 0], axis=0))
        weighted_vals = particles[:, 2] / dens * val * cubic_spline(dist_mat)
        return np.sum(weighted_vals, axis=1)

    def compute_grad(self, dens=[1.0], val=[1.0], particles=None, h=None):
        if particles is None:
            particles = self.particles
        if h is None:
            h = self.h

        def cubic_spline_grad(q):
            return 4 / (3 * h) * np.where(
                np.abs(q) <= 1,
                np.where(
                    np.abs(q) <= 0.5, 18 * np.sign(q) * q**2 - 12 * q,
                    -6 * np.sign(q) * (1 - np.abs(q))**2), 0)

        dist_mat = (np.expand_dims(particles[:, 0], axis=1) -
                    np.expand_dims(particles[:, 0], axis=0))
        weighted_vals = particles[:, 2] * (
            np.expand_dims(val / dens**2, axis=1) + np.expand_dims(
                val / dens**2, axis=0)) * cubic_spline_grad(dist_mat)
        return dens * np.sum(weighted_vals, axis=1)

    def compute_grad2(self, val=[1.0], particles=None, h=None):
        if particles is None:
            particles = self.particles
        if h is None:
            h = self.h

        def cubic_spline_grad(q):
            #return np.where(np.abs(q) <= 1, np.sign(q), 0.0)
            return 4 / (3 * h) * np.where(
                np.abs(q) <= 1,
                np.where(
                    np.abs(q) <= 0.5, 18 * np.sign(q) * q**2 - 12 * q,
                    -6 * np.sign(q) * (1 - np.abs(q))**2), 0)

        dist_mat = (np.expand_dims(particles[:, 0], axis=1) -
                    np.expand_dims(particles[:, 0], axis=0))
        weighted_vals = particles[:, 2] * (
            np.expand_dims(val, axis=1) +
            np.expand_dims(val, axis=0)) * cubic_spline_grad(dist_mat)
        return np.sum(weighted_vals, axis=1)

    def compute_laplace(self, dens=[1.0], val=[1.0], particles=None, h=None):
        if particles is None:
            particles = self.particles
        if h is None:
            h = self.h

        def cubic_spline_grad(q):
            return 4 / (3 * h) * np.where(
                np.abs(q) <= 1,
                np.where(
                    np.abs(q) <= 0.5, 18 * np.sign(q) * q**2 - 12 * q,
                    -6 * np.sign(q) * (1 - np.abs(q))**2), 0)

        dist_mat = np.expand_dims(particles[:, 0], axis=1) - np.expand_dims(
            particles[:, 0], axis=0)
        weighted_vals = particles[:, 2] / dens * (
            np.expand_dims(val, axis=1) - np.expand_dims(
                val, axis=0)) * dist_mat * cubic_spline_grad(dist_mat)
        weighted_vals /= dist_mat**2 + 0.01 * h**2

        return 2 * np.sum(weighted_vals, axis=1)

    def compute_div(self, val=[1.0], particles=None, h=None):
        if particles is None:
            particles = self.particles
        if h is None:
            h = self.h

        def cubic_spline_grad(q):
            return 4 / (3 * h) * np.where(
                np.abs(q) <= 1,
                np.where(
                    np.abs(q) <= 0.5, 18 * np.sign(q) * q**2 - 12 * q,
                    -6 * np.sign(q) * (1 - np.abs(q))**2), 0)

        dist_mat = np.abs(
            np.expand_dims(particles[:, 0], axis=1) -
            np.expand_dims(particles[:, 0], axis=0))
        weighted_vals = (np.expand_dims(val, axis=1) - np.expand_dims(
            val, axis=0)) * cubic_spline_grad(dist_mat)
        return np.sum(weighted_vals, axis=1)

    def compute_dens(self):
        return self.compute_val()

    def compute_pres(self, dens=None):
        dens = self.compute_dens() if dens is None else dens
        pres = np.clip(self.stiffness * ((dens / self.rest_dens)**7 - 1), 0,
                       None)
        pres[:self.bcnt] = pres[self.bcnt]
        return pres

    def compute_visc(self, dens=None):
        dens = self.compute_dens() if dens is None else dens
        return self.visc * self.compute_laplace(dens, self.particles[:, 1])

    def step(self, dt=0.1, mode=2, eps=0.01, max_iter=10000, verbose=False):

        f_visc = self.compute_visc()[self.bcnt:]
        self.particles[self.bcnt:, 1] += dt * (self.gravity + f_visc)
        self.particles[self.bcnt:, 0] += dt * self.particles[self.bcnt:, 1]

        for i in range(max_iter):
            dens = self.compute_val()
            pres = self.compute_pres(dens)

            err = np.max(np.clip(dens - self.rest_dens, 0, None)[self.bcnt:])

            f_pres = -(self.particles[:, 2] /
                       dens)[self.bcnt:] * self.compute_grad(dens,
                                                             pres)[self.bcnt:]

            self.particles[self.bcnt:,
                           1] += dt * f_pres / self.particles[self.bcnt:, 2]
            self.particles[self.bcnt:,
                           0] += dt**2 * f_pres / self.particles[self.bcnt:, 2]

            if err < eps:
                break

        if (verbose):
            print("Iterations: %d/%d, Error: %f (eps: %f)" %
                  (i + 1, max_iter, err, eps))


def gen_dict(data, idx, res, obs_size, grav, width=1, side_walls=False):
    d = []
    for t in range(len(data)):
        frame = {
            'frame_id': t,
            'scene_id': 'sim_%04d' % idx,
            #'num_rigid_bodies': 0,
            #'m': np.fill((cnt, 1), mass),
            #'viscosity': np.fill((cnt, 1), visc),
            #'force': np.zeros((cnt, 3)),
            'grav': np.array([0.0, grav, 0.0])
        }
        frame['pos'] = data[t, :-obs_size, 0]
        frame['vel'] = data[t, :-obs_size, 1]
        frame['box'] = data[t, -obs_size:, 0]

        z = np.zeros_like(frame['pos'])
        frame['pos'] = np.stack([z, frame['pos'], z], axis=-1)
        frame['vel'] = np.stack([z, frame['vel'], z], axis=-1)

        z = np.zeros_like(frame['box'])
        frame['box'] = np.stack([z, frame['box'], z], axis=-1)
        frame['box_normals'] = np.stack([z, z + 1, z], axis=-1)

        if width > 1:
            frame['pos'] = np.expand_dims(frame['pos'], axis=1) + np.reshape(
                np.stack([
                    np.linspace(-(width - 1) * 0.25,
                                (width - 1) * 0.25, width),
                    np.zeros((width, )),
                    np.zeros((width, ))
                ],
                         axis=-1), (1, width, 3))
            frame['pos'] = np.reshape(frame['pos'], (-1, 3))
            frame['box'] = np.expand_dims(frame['box'], axis=1) + np.reshape(
                np.stack([
                    np.linspace(-(width - 1) * 0.25,
                                (width - 1) * 0.25, width),
                    np.zeros((width, )),
                    np.zeros((width, ))
                ],
                         axis=-1), (1, width, 3))
            frame['box'] = np.reshape(frame['box'], (-1, 3))
            frame['vel'] = np.repeat(frame['vel'], width, axis=0)
            frame['box_normals'] = np.repeat(frame['box_normals'],
                                             width,
                                             axis=0)

            if side_walls:
                z = np.zeros(50)
                p = np.arange(50, dtype="float32") * 0.5
                frame['box'] = np.concatenate([
                    frame['box'],
                    np.stack([z - (width + 1) * 0.25, p, z], axis=-1),
                    np.stack([z - (width + 1) * 0.25 - 0.5, p, z], axis=-1),
                    np.stack([z + (width + 1) * 0.25, p, z], axis=-1),
                    np.stack([z + (width + 1) * 0.25 + 0.5, p, z], axis=-1)
                ],
                                              axis=0)
                frame['box_normals'] = np.concatenate([
                    frame['box_normals'],
                    np.stack([z + 1, z, z], axis=-1),
                    np.stack([z + 1, z, z], axis=-1),
                    np.stack([z - 1, z, z], axis=-1),
                    np.stack([z - 1, z, z], axis=-1)
                ],
                                                      axis=0)

        frame['pos'] /= res
        frame['vel'] /= res
        frame['box'] /= res
        frame['grav'] /= res

        d.append(frame)

    return d


def gen_data(data_cnt,
             timesteps,
             res=100,
             min_pts=1,
             max_pts=28,
             pts_cnt=None,
             obs_size=2,
             dt=0.01,
             rnd=0.0,
             radius=0.25,
             mass=1.0,
             stiffness=20.0,
             visc=0.1,
             width=1,
             gravity=-10.0,
             side_walls=False,
             offset=0.0):

    data = []

    gravity *= res

    solver = SPH1D(radius=radius,
                   mass=mass,
                   stiffness=stiffness,
                   visc=visc,
                   gravity=gravity)

    if pts_cnt is None:
        if rnd > 0:
            pts_cnt = np.random.randint(min_pts, max_pts + 1, size=data_cnt)
        elif data_cnt <= max_pts - min_pts + 1:
            pts_cnt = np.random.choice(np.arange(min_pts, max_pts + 1),
                                       size=data_cnt,
                                       replace=False)
            pts_cnt = np.sort(pts_cnt)
        else:
            raise NotImplementedError()

    for d in trange(data_cnt):
        n = pts_cnt[d]
        solver.setup(n, obs_size, rnd=rnd, offset=offset)
        seq = np.empty((timesteps, n + obs_size, 2), dtype="float32")
        for t in range(timesteps):
            seq[t, :, 0] = solver.particles[::-1, 0]
            seq[t, :, 1] = solver.particles[::-1, 1]
            solver.step(dt=dt, mode=2)

        data.append(gen_dict(seq, d, res, obs_size, gravity, width,
                             side_walls))

    return data
