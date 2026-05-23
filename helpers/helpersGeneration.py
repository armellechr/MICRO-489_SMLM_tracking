import numpy as np
import random
import matplotlib.pyplot as plt
#import stackview
import copy
from skimage.measure import block_reduce



class Trajectory:
    def __init__(self, id, initial_position=None, start_frame=0,
             D1_ini=None, D2_ini=None, theta_ini=None,
             initial_intensity=None, initial_sigma=None,
             particle_type="particle", initial_state=None,
             initial_bound_to=None, metadata=None):
    
        self.id = id
        self.positions = []
        self.intensities = []
        self.sigmas = []
        self.states = []
        self.bound_to = []
        self.particle_type = particle_type
        self.metadata = metadata or {}
        self.color = (random.random(), random.random(), random.random())

        # Diffusion parameters
        self.MSD = []
        self.D_tensor = None
        self.D1 = D1_ini
        self.D2 = D2_ini
        self.theta = theta_ini

        self.start_frame = start_frame
        self.end_frame = start_frame - 1

        if initial_position is not None:
            self.positions.append(tuple(initial_position))
            self.intensities.append(initial_intensity)
            self.sigmas.append(initial_sigma)
            self.states.append(initial_state)
            self.bound_to.append(initial_bound_to)
            self.end_frame = start_frame

    def set_id(self, id):
        self.id = id

    def add_position(
        self,
        position,
        frame=None,
        intensity=None,
        sigma=None,
        state=None,
        bound_to=None,
    ):
        if frame is None:
            frame = self.end_frame + 1 if self.positions else self.start_frame

        if not self.positions:
            self.start_frame = frame
            self.end_frame = frame
            self.positions.append(tuple(position))
            self.intensities.append(intensity)
            self.sigmas.append(sigma)
            self.states.append(state)
            self.bound_to.append(bound_to)
            return

        if frame != self.end_frame + 1:
            raise ValueError(
            f"Trajectory {self.id}: expected frame {self.end_frame + 1}, got {frame}"
        )

        self.positions.append(tuple(position))
        self.intensities.append(intensity)
        self.sigmas.append(sigma)
        self.states.append(state)
        self.bound_to.append(bound_to)
        self.end_frame = frame

    def get_positions(self):
        return self.positions

    def get_intensities(self):
        return self.intensities
    
    def get_sigmas(self):
        return self.sigmas

    def get_states(self):
        return self.states

    def get_bound_to(self):
        return self.bound_to

    def get_position_at_frame(self, frame):
        if frame < self.start_frame or frame > self.end_frame:
            return None
        return self.positions[frame - self.start_frame]

    def get_intensity_at_frame(self, frame):
        if frame < self.start_frame or frame > self.end_frame:
            return None
        return self.intensities[frame - self.start_frame]
    
    def get_sigma_at_frame(self, frame):
        if frame < self.start_frame or frame > self.end_frame:
            return None
        return self.sigmas[frame - self.start_frame]

    def get_state_at_frame(self, frame):
        if frame < self.start_frame or frame > self.end_frame:
            return None
        return self.states[frame - self.start_frame]

    def get_bound_to_at_frame(self, frame):
        if frame < self.start_frame or frame > self.end_frame:
            return None
        return self.bound_to[frame - self.start_frame]

    def frames(self):
        return list(range(self.start_frame, self.end_frame + 1))

    def last_position(self):
        return self.positions[-1] if self.positions else None

    def last_intensity(self):
        return self.intensities[-1] if self.intensities else None

    def last_sigma(self):
        return self.sigmas[-1] if self.sigmas else None

    def last_state(self):
        return self.states[-1] if self.states else None

    def last_bound_to(self):
        return self.bound_to[-1] if self.bound_to else None

    def length(self):
        return len(self.positions)

    def print_trajectory(self):
        print(f"Trajectory {self.id} (frames {self.start_frame} -> {self.end_frame}):")
        for frame, pos, intensity, sigma in zip(self.frames(), self.positions, self.intensities, self.sigmas):
            print(frame, float(pos[0]), float(pos[1]), intensity, sigma)


def gaussian_2d_image_coords(xc, yc, sigma, grid_size, amplitude):
    """
    2D Gaussian defined directly in image coordinates.
    xc, yc are in pixel coordinates of the high-resolution grid.
    """
    y = np.arange(grid_size)
    x = np.arange(grid_size)
    x, y = np.meshgrid(x, y)
    gauss = amplitude * np.exp(-((x - xc)**2 + (y - yc)**2) / (2 * sigma**2))
    return gauss

def generate_diffusion_steps(
    num_steps,
    D1,
    D2=None,
    theta=0.0,
    dt=1.0,
    nposframe=1,
    mode="isotropic",
):
    """
    Generate Brownian increments.

    Parameters
    ----------
    D1 : float
        Main diffusion coefficient.
    D2 : float or None
        Secondary diffusion coefficient. If None in isotropic mode, D2 = D1.
    theta : float
        Direction angle in radians.
    mode : str
        "isotropic" or "anisotropic".
    """

    if mode == "isotropic":
        D2 = D1

    elif mode == "anisotropic":
        if D2 is None:
            raise ValueError("D2 must be provided for anisotropic diffusion.")

    else:
        raise ValueError("mode must be 'isotropic' or 'anisotropic'.")

    sigma1 = np.sqrt(2 * D1 * dt / nposframe)
    sigma2 = np.sqrt(2 * D2 * dt / nposframe)

    # local principal-axis displacements
    local_steps = np.zeros((num_steps, 2))
    local_steps[:, 0] = np.random.randn(num_steps) * sigma1
    local_steps[:, 1] = np.random.randn(num_steps) * sigma2

    # rotation matrix
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    # rotate local principal-axis motion into image coordinates
    return local_steps @ R.T

def simulate_brownian_motion(
    nparticles,
    nframes,
    nposframe,
    D_list,
    dt,
    startAtZero=False,
    boundary_margin=10,
    frame_size=(128, 128),
    intensity_mean=1000,
    intensity_std=50,
    sigma_mean=1.0,
    sigma_std=0.2,
    anisotropy_ratio_range=(0.1, 1.0),
    theta_range=(0, np.pi),
    particle_type="particle",
):
    """
    Simulate Brownian motion trajectories for multiple particles, each with a diffusion coefficient D drawn from a provided distribution.
    Parameters
    ----------
    nparticles : int
        Number of particles to simulate.
    nframes : int
        Number of output frames in the final video.
    nposframe : int
        Number of subpositions (subframes) per output frame. Total subframes = nframes * nposframe.
    D_list : list of tuples
        List of (D, probability) pairs defining the distribution of diffusion coefficients.
    dt : float
        Time step between subframes (in seconds).
    startAtZero : bool
        If True, all trajectories start at (0, 0). If False, they start at random positions within the field of view.
    boundary_margin : float
        If startAtZero is False, this margin ensures that initial positions are not too close to the border of the field of view.
    frame_size : tuple
        Size of the field of view in pixels (width, height). Used to determine random starting positions if startAtZero is False.
    intensity_mean : float
        Mean intensity for the simulated particles (used when adding intensity to Trajectory).
    intensity_std : float
        Standard deviation of intensity for the simulated particles.
    sigma_mean : float
        Mean sigma for the simulated particles (used when adding sigma to Trajectory).
    sigma_std : float
        Standard deviation of sigma for the simulated particles.
    anisotropy_ratio_range:
        Tuple that defines the range of possible ratios D2/D1.
    theta_range:
        Tuple that defines the range of possible angles (in radians) for the principal diffusion axis.
    particle_type:
        Label attached to every generated trajectory.
    """
    num_steps = nframes * nposframe

    D_values = [d for d, _ in D_list]
    probs = np.array([p for _, p in D_list], dtype=float)
    probs /= probs.sum()

    trajectories = []

    for p in range(nparticles):
        D1 = float(np.random.choice(D_values, p=probs))

        ratio = np.random.uniform(anisotropy_ratio_range[0], anisotropy_ratio_range[1])
        D2 = D1 * ratio
        theta = np.random.uniform(theta_range[0], theta_range[1])


        dxy = generate_diffusion_steps(
            num_steps=num_steps,
            D1=D1,
            D2=D2,
            theta=theta,
            dt=dt,
            nposframe=nposframe
        )

        positions = np.cumsum(dxy, axis=0)

        if startAtZero:
            positions[0] = [0.0, 0.0]
        else:
            start_x = np.random.uniform(boundary_margin, frame_size[0] - boundary_margin)
            start_y = np.random.uniform(boundary_margin, frame_size[1] - boundary_margin)
            positions += np.array([start_x, start_y])

        traj = Trajectory(
            id=p,
            start_frame=0,
            D1_ini=D1,
            D2_ini=D2,
            theta_ini=theta,
            particle_type=particle_type,
        )

        for t in range(num_steps):
            base_intensity = max(np.random.normal(intensity_mean, intensity_std), 0.0)
            base_sigma = max(np.random.normal(sigma_mean, sigma_std), 0.1)

            traj.add_position(
                tuple(positions[t]),
                frame=t,
                intensity=base_intensity,
                sigma=base_sigma,
                state="free",
                bound_to=None,
            )

        trajectories.append(traj)

    return trajectories


def _sample_diffusion_value(D_config):
    if isinstance(D_config, (int, float, np.integer, np.floating)):
        return float(D_config)

    D_values = [d for d, _ in D_config]
    probs = np.array([p for _, p in D_config], dtype=float)
    probs /= probs.sum()
    return float(np.random.choice(D_values, p=probs))


def _sample_positive_normal(mean, std, min_value=0.0):
    return max(float(np.random.normal(mean, std)), min_value)


def _random_position(frame_size, boundary_margin):
    width, height = frame_size
    return np.array([
        np.random.uniform(boundary_margin, width - boundary_margin),
        np.random.uniform(boundary_margin, height - boundary_margin),
    ])


def _reflect_position(position, frame_size):
    reflected = np.array(position, dtype=float)

    for axis, limit in enumerate(frame_size):
        if limit <= 1:
            reflected[axis] = 0.0
            continue

        while reflected[axis] < 0 or reflected[axis] >= limit:
            if reflected[axis] < 0:
                reflected[axis] = -reflected[axis]
            if reflected[axis] >= limit:
                reflected[axis] = 2 * (limit - 1) - reflected[axis]

    return reflected


def simulate_ligand_receptor_motion(
    n_ligands,
    n_receptors,
    nframes,
    nposframe,
    dt,
    ligand_D=4.0,
    receptor_D=0.02,
    frame_size=(128, 128),
    boundary_margin=10,
    binding_radius=3.0,
    kon=0.4,
    koff=0.03,
    allow_multiple_ligands_per_receptor=False,
    bound_position_noise=0.15,
    ligand_intensity_mean=550,
    ligand_intensity_std=100,
    receptor_intensity_mean=950,
    receptor_intensity_std=150,
    ligand_sigma_mean=0.75,
    ligand_sigma_std=0.08,
    receptor_sigma_mean=1.35,
    receptor_sigma_std=0.12,
    ligand_start_positions=None,
    receptor_start_positions=None,
    reflect_boundaries=True,
    return_events=False,
):
    """
    Simulate two interacting populations: fast ligands and slow receptors.

    Ligands diffuse freely until they are within ``binding_radius`` of an
    available receptor. Binding and unbinding probabilities are computed from
    ``kon`` and ``koff`` at the subframe time step ``dt / nposframe``.
    """
    num_steps = nframes * nposframe
    sub_dt = dt / nposframe
    p_on = 1.0 - np.exp(-kon * sub_dt)
    p_off = 1.0 - np.exp(-koff * sub_dt)

    receptor_ids = list(range(n_receptors))
    ligand_ids = list(range(n_receptors, n_receptors + n_ligands))

    receptor_positions = np.zeros((n_receptors, num_steps, 2), dtype=float)
    receptor_D_values = []

    for r in range(n_receptors):
        D = _sample_diffusion_value(receptor_D)
        receptor_D_values.append(D)

        if receptor_start_positions is None:
            receptor_positions[r, 0] = _random_position(frame_size, boundary_margin)
        else:
            receptor_positions[r, 0] = receptor_start_positions[r]

        if num_steps > 1:
            steps = generate_diffusion_steps(
                num_steps=num_steps - 1,
                D1=D,
                D2=D,
                theta=0.0,
                dt=dt,
                nposframe=nposframe,
            )

            for t in range(1, num_steps):
                receptor_positions[r, t] = receptor_positions[r, t - 1] + steps[t - 1]
                if reflect_boundaries:
                    receptor_positions[r, t] = _reflect_position(
                        receptor_positions[r, t],
                        frame_size,
                    )

    ligand_positions = np.zeros((n_ligands, num_steps, 2), dtype=float)
    ligand_states = np.full((n_ligands, num_steps), "free", dtype=object)
    ligand_bound_to = np.full((n_ligands, num_steps), None, dtype=object)
    ligand_D_values = []

    ligand_steps = []
    for l in range(n_ligands):
        D = _sample_diffusion_value(ligand_D)
        ligand_D_values.append(D)

        if ligand_start_positions is None:
            ligand_positions[l, 0] = _random_position(frame_size, boundary_margin)
        else:
            ligand_positions[l, 0] = ligand_start_positions[l]

        if num_steps > 1:
            ligand_steps.append(
                generate_diffusion_steps(
                    num_steps=num_steps - 1,
                    D1=D,
                    D2=D,
                    theta=0.0,
                    dt=dt,
                    nposframe=nposframe,
                )
            )
        else:
            ligand_steps.append(np.zeros((0, 2), dtype=float))

    current_bound_to = [None] * n_ligands
    active_event_index = [None] * n_ligands
    binding_events = []

    for t in range(num_steps):
        occupied_receptors = {
            receptor_idx
            for receptor_idx in current_bound_to
            if receptor_idx is not None
        }

        for l in range(n_ligands):
            receptor_idx = current_bound_to[l]

            if receptor_idx is not None and np.random.rand() < p_off:
                binding_events[active_event_index[l]]["end_subframe"] = t - 1
                current_bound_to[l] = None
                active_event_index[l] = None
                if not allow_multiple_ligands_per_receptor:
                    occupied_receptors.discard(receptor_idx)
                receptor_idx = None

            if receptor_idx is not None:
                ligand_positions[l, t] = receptor_positions[receptor_idx, t]
                if bound_position_noise > 0:
                    ligand_positions[l, t] += np.random.normal(
                        0.0,
                        bound_position_noise,
                        size=2,
                    )
                if reflect_boundaries:
                    ligand_positions[l, t] = _reflect_position(
                        ligand_positions[l, t],
                        frame_size,
                    )
                ligand_states[l, t] = "bound"
                ligand_bound_to[l, t] = receptor_ids[receptor_idx]
                continue

            if t > 0:
                ligand_positions[l, t] = ligand_positions[l, t - 1] + ligand_steps[l][t - 1]
                if reflect_boundaries:
                    ligand_positions[l, t] = _reflect_position(
                        ligand_positions[l, t],
                        frame_size,
                    )

            receptor_distances = np.linalg.norm(
                receptor_positions[:, t] - ligand_positions[l, t],
                axis=1,
            )
            candidate_indices = np.where(receptor_distances <= binding_radius)[0]

            if not allow_multiple_ligands_per_receptor:
                candidate_indices = np.array([
                    idx
                    for idx in candidate_indices
                    if idx not in occupied_receptors
                ])

            if candidate_indices.size > 0 and np.random.rand() < p_on:
                nearest_idx = candidate_indices[
                    np.argmin(receptor_distances[candidate_indices])
                ]
                current_bound_to[l] = nearest_idx
                occupied_receptors.add(nearest_idx)

                ligand_positions[l, t] = receptor_positions[nearest_idx, t]
                if bound_position_noise > 0:
                    ligand_positions[l, t] += np.random.normal(
                        0.0,
                        bound_position_noise,
                        size=2,
                    )
                if reflect_boundaries:
                    ligand_positions[l, t] = _reflect_position(
                        ligand_positions[l, t],
                        frame_size,
                    )

                ligand_states[l, t] = "bound"
                ligand_bound_to[l, t] = receptor_ids[nearest_idx]

                binding_events.append({
                    "ligand_id": ligand_ids[l],
                    "receptor_id": receptor_ids[nearest_idx],
                    "start_subframe": t,
                    "end_subframe": None,
                })
                active_event_index[l] = len(binding_events) - 1

    for event in binding_events:
        if event["end_subframe"] is None:
            event["end_subframe"] = num_steps - 1

        event["start_frame"] = event["start_subframe"] // nposframe
        event["end_frame"] = event["end_subframe"] // nposframe

    receptor_states = np.full((n_receptors, num_steps), "free", dtype=object)
    receptor_bound_to = np.full((n_receptors, num_steps), None, dtype=object)

    for r in range(n_receptors):
        receptor_id = receptor_ids[r]
        for t in range(num_steps):
            bound_ligands = [
                ligand_ids[l]
                for l in range(n_ligands)
                if ligand_bound_to[l, t] == receptor_id
            ]
            if len(bound_ligands) == 1:
                receptor_states[r, t] = "bound"
                receptor_bound_to[r, t] = bound_ligands[0]
            elif len(bound_ligands) > 1:
                receptor_states[r, t] = "bound"
                receptor_bound_to[r, t] = tuple(bound_ligands)

    trajectories = []

    for r in range(n_receptors):
        intensity = _sample_positive_normal(
            receptor_intensity_mean,
            receptor_intensity_std,
            min_value=0.0,
        )
        sigma = _sample_positive_normal(
            receptor_sigma_mean,
            receptor_sigma_std,
            min_value=0.1,
        )
        traj = Trajectory(
            id=receptor_ids[r],
            start_frame=0,
            D1_ini=receptor_D_values[r],
            D2_ini=receptor_D_values[r],
            theta_ini=0.0,
            particle_type="receptor",
            metadata={"role": "receptor"},
        )

        for t in range(num_steps):
            traj.add_position(
                tuple(receptor_positions[r, t]),
                frame=t,
                intensity=intensity,
                sigma=sigma,
                state=receptor_states[r, t],
                bound_to=receptor_bound_to[r, t],
            )

        trajectories.append(traj)

    for l in range(n_ligands):
        intensity = _sample_positive_normal(
            ligand_intensity_mean,
            ligand_intensity_std,
            min_value=0.0,
        )
        sigma = _sample_positive_normal(
            ligand_sigma_mean,
            ligand_sigma_std,
            min_value=0.1,
        )
        traj = Trajectory(
            id=ligand_ids[l],
            start_frame=0,
            D1_ini=ligand_D_values[l],
            D2_ini=ligand_D_values[l],
            theta_ini=0.0,
            particle_type="ligand",
            metadata={"role": "ligand"},
        )

        for t in range(num_steps):
            traj.add_position(
                tuple(ligand_positions[l, t]),
                frame=t,
                intensity=intensity,
                sigma=sigma,
                state=ligand_states[l, t],
                bound_to=ligand_bound_to[l, t],
            )

        trajectories.append(traj)

    if return_events:
        return trajectories, binding_events

    return trajectories


def trajectories_to_global_video(trajectories, nframes, nPosPerFrame, image_props=None):
    """
    Render all Trajectory objects into one shared microscopy movie.

    Parameters
    ----------
    trajectories : list[Trajectory]
        List of Trajectory objects. Each trajectory is assumed to store one
        position per subframe.
    nPosPerFrame : int
        Number of subpositions integrated into one output frame.
    image_props : dict
        Rendering parameters.

    Returns
    -------
    out_video : ndarray
        Shape (nFrames, output_size, output_size)
    """
    if image_props is None:
        image_props = {}

    if len(trajectories) == 0:
        raise ValueError("trajectories is empty")

    props = {
        "particle_intensity": [500, 100],
        "particle_sigma": [1.0, 0.2],
        "particle_type_props": {},
        "use_trajectory_sigma": False,
        "NA": 1.46,
        "wavelength": 500e-9,
        "psf_division_factor": 1,
        "resolution": 100e-9,
        "output_size": 128,
        "upsampling_factor": 5,
        "background_intensity": [100, 30],
        "poisson_noise": 100,
        "trajectory_unit": -1,   # -1 means positions are already in pixels
        "invert_y": True,
    }
    props.update(image_props)


    resolution = props["resolution"]
    traj_unit = props["trajectory_unit"]

    output_size = props["output_size"]
    upsampling_factor = props["upsampling_factor"]
    psf_div_factor = props["psf_division_factor"]

    fwhm_psf = props["wavelength"] / 2 * props["NA"] / psf_div_factor
    gaussian_sigma = upsampling_factor / resolution * fwhm_psf / 2.355

    background_mean, background_std = props["background_intensity"]
    poisson_noise = props["poisson_noise"]
    particle_type_props = props["particle_type_props"]

    hr_size = output_size * upsampling_factor
    out_video = np.zeros((nframes, output_size, output_size), dtype=np.float32)

    for f in range(nframes):
        frame_hr = np.zeros((hr_size, hr_size), dtype=np.float32)

        start = f * nPosPerFrame
        end = (f + 1) * nPosPerFrame

        # --- accumulate all particles into the same high-res frame ---
        for traj in trajectories:
            if traj.length() == 0:
                continue

            type_props = particle_type_props.get(
                getattr(traj, "particle_type", "particle"),
                {}
            )
            particle_mean, particle_std = type_props.get(
                "particle_intensity",
                props["particle_intensity"]
            )
            sigma_mean, sigma_std = type_props.get(
                "particle_sigma",
                props["particle_sigma"]
            )


            for subframe in range(start, end):
                pos = traj.get_position_at_frame(subframe)
                if pos is None:
                    continue

                x, y = pos

                if props["invert_y"]:
                    y = output_size - y

                if traj_unit != -1:
                    x = x * traj_unit / (resolution * 1e9)
                    y = y * traj_unit / (resolution * 1e9)

                x *= upsampling_factor
                y *= upsampling_factor

                # skip if outside field of view
                if x < 0 or x >= hr_size or y < 0 or y >= hr_size:
                    continue

                intensity = traj.get_intensity_at_frame(subframe)
                sigma = traj.get_sigma_at_frame(subframe)
                if intensity is None:
                    intensity = max(np.random.normal(particle_mean, particle_std), 0.0)
                if sigma is None:
                    sigma = max(np.random.normal(sigma_mean, sigma_std), 0.1)
                if props["use_trajectory_sigma"]:
                    spot_sigma = max(float(sigma), 1e-6) * upsampling_factor
                else:
                    spot_sigma = gaussian_sigma
        
                spot_intensity = intensity / nPosPerFrame

                frame_spot = gaussian_2d_image_coords(
                    x, y, spot_sigma, hr_size, spot_intensity
                )

                spot_max = np.max(frame_spot)
                if spot_max > 1e-8:
                    frame_hr += (spot_intensity / spot_max) * frame_spot

        frame_lr = block_reduce(
            frame_hr,
            block_size=(upsampling_factor, upsampling_factor),
            func=np.mean
        )

        frame_lr += np.clip(
            np.random.normal(background_mean, background_std, frame_lr.shape),
            0,
            background_mean + 3 * background_std
        )

        if poisson_noise != -1:
            frame_lr = frame_lr * np.random.poisson(poisson_noise, size=frame_lr.shape) / poisson_noise

        out_video[f] = frame_lr.astype(np.float32)

    return out_video

import imageio

def save_video_as_gif(out_video, filename="movie.gif", fps=10):
    # Normalize frames to 0–255 uint8
    vid = out_video.copy()
    vid -= vid.min()
    vid /= vid.max()
    vid = (vid * 255).astype(np.uint8)

    frames = [vid[i] for i in range(vid.shape[0])]
    imageio.mimsave(f'GIFs\\{filename}', frames, fps=fps)




# ----- VISUALIZATION FUNCTIONS -----

def show_img(image):
    plt.imshow(image, cmap='gray', vmin=0, vmax=5000) # setting 0-5000 range
    plt.show()

def check_min_max(img):
    print("min:", np.min(img))
    print("max:", np.max(img))

   
def view_frames(frames):
    stack = np.stack(frames, axis=0).astype(np.float32)
    stack = np.clip(stack, 0, 5000)
    return stackview.slice(stack, continuous_update=True)

def show_trajectory(frames, trajectories, traj_id=0, frame_id=0, save_path=None):
    traj = trajectories[traj_id]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(frames[frame_id], cmap="gray", vmin=0, vmax=5000)

    if traj.start_frame <= frame_id <= traj.end_frame:
        valid_positions = traj.positions[:frame_id - traj.start_frame + 1]
    elif frame_id > traj.end_frame:
        valid_positions = traj.positions
    else:
        valid_positions = []

    if len(valid_positions) > 1:
        positions = np.array(valid_positions)
        y = positions[:, 0]
        x = positions[:, 1]
        ax.plot(x, y, '-', color=traj.color, linewidth=2)
        ax.plot(x[0], y[0], '^', color=traj.color, markersize=5)
        ax.plot(x[-1], y[-1], '+', color=traj.color, markersize=5)

    ax.set_title(
        f"Trajectory {traj_id} on frame {frame_id} "
        f"(active: {traj.start_frame}->{traj.end_frame})"
    )
    ax.axis("off")

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)

    plt.show()

# def show_trajectories(frames, trajectories, frame_id=0, title=None, save_path=None):
#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.imshow(frames[frame_id], cmap="gray", vmin=0, vmax=5000)

#     for traj in trajectories:
#         if frame_id < traj.start_frame:
#             continue

#         if frame_id <= traj.end_frame:
#             valid_positions = traj.positions[:frame_id - traj.start_frame + 1]
#         else:
#             valid_positions = traj.positions

#         if len(valid_positions) > 1:
#             positions = np.array(valid_positions)
#             y = positions[:, 0]
#             x = positions[:, 1]
#             ax.plot(x, y, '-', color=traj.color, linewidth=2)
#             ax.plot(x[0], y[0], '^', color=traj.color, markersize=5)
#             ax.plot(x[-1], y[-1], '+', color=traj.color, markersize=5)

#     ax.set_title(title if title else f"Trajectories on frame {frame_id}")
#     ax.axis("off")

#     if save_path is not None:
#         fig.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)

#     plt.show()

def show_trajectories(frames, trajectories, frame_id=0, title=None, save_path=None, show_id=False):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(frames[frame_id], cmap="gray", vmin=0, vmax=5000)

    for traj in trajectories:
        if frame_id < traj.start_frame:
            continue

        if frame_id <= traj.end_frame:
            valid_positions = traj.positions[:frame_id - traj.start_frame + 1]
        else:
            valid_positions = traj.positions

        if len(valid_positions) > 1:
            positions = np.array(valid_positions)
            y = positions[:, 0]
            x = positions[:, 1]

            # Draw trajectory
            ax.plot(x, y, '-', color=traj.color, linewidth=2)
            ax.plot(x[0], y[0], '^', color=traj.color, markersize=5)
            ax.plot(x[-1], y[-1], '+', color=traj.color, markersize=5)

            # Draw ID label
            if show_id:
                ax.text(
                    x[-1] + 1, y[-1] + 1,          # slight offset
                    str(traj.id),                 # trajectory ID
                    color=traj.color,
                    fontsize=8,
                    weight='bold'
                )

    ax.set_title(title if title else f"Trajectories on frame {frame_id}")
    ax.axis("off")

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)

    plt.show()


def linear_trajectories_visualizer(trajectories_new, trajectories_GT):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))

    for traj in trajectories_GT:
        if traj.length() > 0:
            plt.hlines(
                traj.id,
                traj.start_frame,
                traj.end_frame + 1,
                colors=traj.color,
                linestyles='solid',
                label='GT' if traj.id == 0 else ""
            )

    for traj in trajectories_new:
        if traj.length() > 0 and traj.id is not None and traj.id != -1:
            plt.hlines(
                traj.id + 0.2,
                traj.start_frame,
                traj.end_frame + 1,
                colors=traj.color,
                linestyles='dashed',
                label='New' if traj.id == 0 else ""
            )

    plt.xlabel('Frame Number')
    plt.ylabel('Trajectory ID')

    all_ids = [traj.id for traj in trajectories_GT if traj.length() > 0] + \
              [traj.id for traj in trajectories_new if traj.length() > 0 and traj.id is not None and traj.id != -1]

    if all_ids:
        plt.ylim(-1, max(all_ids) + 1)

    plt.title('Comparison of GT and experimental trajectories')
    plt.legend()
    plt.grid()
    plt.show()

from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def animate_video(video, interval=100, cmap="gray", vmin=0, vmax=1000):
    """
    Animate a video of shape (nframes, H, W).
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    
    if vmin is None:
        vmin = video.min()
    if vmax is None:
        vmax = video.max()

    im = ax.imshow(video[0], cmap=cmap, vmin=vmin, vmax=vmax, animated=True)
    title = ax.set_title("Frame 0")
    ax.axis("off")

    def update(frame):
        im.set_array(video[frame])
        title.set_text(f"Frame {frame}")
        return [im, title]

    ani = FuncAnimation(
        fig,
        update,
        frames=video.shape[0],
        interval=interval,
        blit=True
    )
    plt.close(fig)
    return HTML(ani.to_jshtml())

def print_diffusion_params(trajectories):
    for traj in trajectories:
        print(
            f"id={traj.id}"
            f"D1={traj.D1:.3f}, D2={traj.D2:.3f}, "
            f"theta={np.rad2deg(traj.theta):.1f} deg"
        )

# ------------ PATCHES EXTRACTION AND VISUALIZATION -------------

def extract_patches_per_id(frames, trajectories, traj_id=0, patch_size=11):
    half = patch_size // 2
    patches = []

    # find the trajectory with the given ID
    traj = next(t for t in trajectories if t.id == traj_id)

    # iterate over (frame index, position)
    for frame_idx, pos in zip(traj.frames(), traj.positions):
        y, x = int(pos[0]), int(pos[1])

        # extract patch
        patch = frames[
            frame_idx,
            y-half:y+half+1,
            x-half:x+half+1
        ]

        # skip if patch is incomplete (near borders)
        if patch.shape != (patch_size, patch_size):
            continue

        patches.append(patch)

    return np.array(patches)

def visualize_particle_patches(patches):
    # Remove padded frames (all zeros)
    mask = patches.sum(axis=(1,2)) > 0
    real_patches = patches[mask]

    nframes = real_patches.shape[0]
    fig, axes = plt.subplots(1, nframes, figsize=(2*nframes, 3))
    axes = np.atleast_1d(axes)

    for j in range(nframes):
        axes[j].imshow(real_patches[j], cmap='gray')
        axes[j].set_title(f'Frame {j}')
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def interactive_particle_viewer(patches):
    """
    patches: array of shape (nframes, H, W)
    """

    # stackview.slice returns a matplotlib Figure
    fig = stackview.slice(patches, continuous_update=True)

    return fig
