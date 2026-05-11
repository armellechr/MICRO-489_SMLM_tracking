import numpy as np
import random
import matplotlib.pyplot as plt
import stackview
from skimage.measure import block_reduce


class Trajectory:
    def __init__(self, id, initial_position=None, start_frame=0,
             D_GT=None, D1_GT=None, D2_GT=None, theta_GT=None,
             initial_intensity=None, initial_sigma=None):
    
        self.id = id
        self.positions = []
        self.intensities = []
        self.sigmas = []
        self.color = (random.random(), random.random(), random.random())

        self.msd = []

        self.D1_GT = D1_GT
        self.D2_GT = D2_GT
        self.theta_GT = theta_GT

        self.D1_trajectory = np.nan
        self.D2_trajectory = np.nan
        self.theta_trajectory = np.nan

        self.D1_detection = np.nan
        self.D2_detection = np.nan
        self.theta_detection = np.nan

        self.D1_localization = np.nan
        self.D2_localization = np.nan
        self.theta_localization = np.nan

        self.start_frame = start_frame
        self.end_frame = start_frame - 1

        if initial_position is not None:
            self.positions.append(tuple(initial_position))
            self.intensities.append(initial_intensity)
            self.sigmas.append(initial_sigma)
            self.end_frame = start_frame

    def set_id(self, id):
        self.id = id

    def add_position(self, position, frame=None, intensity=None, sigma=None):
        if frame is None:
            frame = self.end_frame + 1 if self.positions else self.start_frame

        if not self.positions:
            self.start_frame = frame
            self.end_frame = frame
            self.positions.append(tuple(position))
            self.intensities.append(intensity)
            self.sigmas.append(sigma)
            return

        if frame != self.end_frame + 1:
            raise ValueError(
            f"Trajectory {self.id}: expected frame {self.end_frame + 1}, got {frame}"
        )

        self.positions.append(tuple(position))
        self.intensities.append(intensity)
        self.sigmas.append(sigma)
        self.end_frame = frame

    # def add_intensity(self, intensity, frame=None):
    #     """
    #     Optional helper if you want to modify/store intensity separately,
    #     but frame must already exist.
    #     """
    #     if frame is None:
    #         if not self.positions:
    #             raise ValueError("Cannot add intensity to empty trajectory")
    #         frame = self.end_frame

    #     if frame < self.start_frame or frame > self.end_frame:
    #         raise ValueError(f"Frame {frame} outside trajectory span")

    #     idx = frame - self.start_frame
    #     self.intensities[idx] = intensity

    # def add_sigma(self, sigma, frame=None):
    #     """
    #     Optional helper if you want to modify/store sigma separately,
    #     but frame must already exist.
    #     """
    #     if frame is None:
    #         if not self.positions:
    #             raise ValueError("Cannot add sigma to empty trajectory")
    #         frame = self.end_frame

    #     if frame < self.start_frame or frame > self.end_frame:
    #         raise ValueError(f"Frame {frame} outside trajectory span")

    #     idx = frame - self.start_frame
    #     self.sigmas[idx] = sigma

    def get_positions(self):
        return self.positions

    def get_intensities(self):
        return self.intensities
    
    def get_sigmas(self):
        return self.sigmas

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

    def frames(self):
        return list(range(self.start_frame, self.end_frame + 1))

    def last_position(self):
        return self.positions[-1] if self.positions else None

    def last_intensity(self):
        return self.intensities[-1] if self.intensities else None

    def last_sigma(self):
        return self.sigmas[-1] if self.sigmas else None

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
    bg_size=(128, 128),
    intensity_mean=1000,
    intensity_std=50,
    sigma_mean=1.0,
    sigma_std=0.2,
    anisotropy_ratio_range=(0.1, 1.0),
    theta_range=(0, np.pi)
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
    bg_size : tuple
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
            start_x = np.random.uniform(boundary_margin, bg_size[0] - boundary_margin)
            start_y = np.random.uniform(boundary_margin, bg_size[1] - boundary_margin)
            positions += np.array([start_x, start_y])

        traj = Trajectory(
            id=p,
            start_frame=0,
            D1_GT=D1,
            D2_GT=D2,
            theta_GT=theta
        )

        for t in range(num_steps):
            base_intensity = max(np.random.normal(intensity_mean, intensity_std), 0.0)
            base_sigma = max(np.random.normal(sigma_mean, sigma_std), 0.1)

            traj.add_position(
                tuple(positions[t]),
                frame=t,
                intensity=base_intensity,
                sigma=base_sigma
            )

        trajectories.append(traj)

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

    particle_mean, particle_std = props["particle_intensity"]
    background_mean, background_std = props["background_intensity"]
    poisson_noise = props["poisson_noise"]

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
        
                spot_intensity = intensity / nPosPerFrame

                frame_spot = gaussian_2d_image_coords(
                    x, y, gaussian_sigma, hr_size, spot_intensity
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

def show_trajectories(frames, trajectories, frame_id=0, title=None, save_path=None):
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
            ax.plot(x, y, '-', color=traj.color, linewidth=2)
            ax.plot(x[0], y[0], '^', color=traj.color, markersize=5)
            ax.plot(x[-1], y[-1], '+', color=traj.color, markersize=5)

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

def animate_video(video, interval=300, cmap="gray", vmin=0, vmax=1000):
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
            f"D1={traj.D1_GT:.3f}, D2={traj.D2_GT:.3f}, "
            f"theta={np.rad2deg(traj.theta_GT):.1f} deg"
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