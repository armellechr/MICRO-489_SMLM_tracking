from turtle import position

import numpy as np
import random
import matplotlib.pyplot as plt
import stackview
from skimage.measure import block_reduce

# ----- BACKGROUND GENERATION -----

def uniform_bg(size, value=100):
    return np.full((size, size), value)

def background(x,y):
    return 100 * np.ones_like(x) # 100 to mimic microscope background intensity

def background1(x,y):
    sigma = 0.5
    return 100 * np.ones_like(x) +  500 * (1/(2*np.pi*sigma**2)) * np.exp(-((x-0)**2 + (y-0)**2)/(2*sigma**2))

def background2(x,y):
    sigma = 0.5
    return 100 * np.ones_like(x) +  500 * (1/(2*np.pi*sigma**2)) * np.exp(-((x-1)**2 + (y-1)**2)/(2*sigma**2))

def background3(x, y, peak=1000, floor=100, theta_deg=35, offset=0.75, sigma=0.4):
    
    theta = np.deg2rad(theta_deg)

    # Coordinate perpendicular to the band (distance-like in normalized units)
    v = -np.sin(theta) * x + np.cos(theta) * y

    # Bright ridge centered at v = offset
    ridge = np.exp(-0.5 * ((v - offset) / sigma) ** 2)

    # Map ridge (0..1) to intensity (floor..peak)
    return floor + (peak - floor) * ridge

# ----- PARTICLES GENERATION -----

def generate_particles(N):
    # shape: (ID, x_c, y_c, d)
    particles = np.empty((0, 4))
    for i in range(1, N+1):
        x_c = random.uniform(0, 128)
        y_c = random.uniform(0, 128)
        d = random.randint(0, 4)
        tag = i
        particles = np.append(particles, [(tag, x_c, y_c, d)], axis=0)
    return particles


class Particle:
    def __init__(self, ID, row_c, col_c, d, start_frame=0, end_frame=None, blink_prob=0.05, max_blink_len = 2):
        self.ID = ID
        self.center = (row_c, col_c)
        self.d = d
        self.color = (random.random(), random.random(), random.random())
        self.start_frame = start_frame
        self.end_frame = end_frame

        # blinking
        self.visible = True
        self.blink_prob = blink_prob
        self.max_blink_len = max_blink_len
        self.remaining_blink = 0

def generate_particles_OOP(N):
    particles = []
    for i in range(N):
        row_c = random.uniform(0, 128)
        col_c = random.uniform(0, 128)
        d = random.randint(0, 4)
        particles.append(Particle(i, row_c, col_c, d))
    return particles

def generate_particles(N, D=None):
    particles = []
    for i in range(N):
        row_c = random.uniform(0, 128)
        col_c = random.uniform(0, 128)
        d = random.randint(0, 4) if D is None else D
        particles.append(Particle(i, row_c, col_c, d))
    return particles

def generate_particles_setD(N, D):
    particles = []
    for i in range(N):
        row_c = random.uniform(20, 108) # far from edges
        col_c = random.uniform(20, 108) # far from edges
        d = D
        particles.append(Particle(i, row_c, col_c, d))
    return particles

def g(c, x, y, x_c, y_c, sigma):
    return c * np.exp(-((x-x_c)**2 + (y-y_c)**2)/(2*sigma**2))

def update_visibility(p):
    # currently hidden
    if p.remaining_blink > 0:
        p.remaining_blink -= 1
        p.visible = False
        return

    # currently visible, maybe start blinking
    if random.random() < p.blink_prob:
        p.remaining_blink = random.randint(1, p.max_blink_len) # start new blink with random length
        p.visible = False
    else:
        p.visible = True

def place_particles(img, particles, frame, c=1000, sigma=0.5):
    output = img.copy()
    for p in particles:
        if frame < p.start_frame:
            continue
        if p.end_frame is not None and frame > p.end_frame:
            continue
        if not p.visible:
            continue

        x_c, y_c = p.center
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                output[x, y] += g(c, x, y, x_c, y_c, sigma)
    return output

# ----- NOISE ADDITION -----
def add_gaussian_noise(img, mean=100, std=30):
    noise_g = np.random.normal(mean, std, img.shape)
    #print('Max noise value:', np.max(noise_g))
    #print('Min noise value:', np.min(noise_g))
    noisy_img = img + noise_g
    noisy_img = np.clip(noisy_img, 0, 5000) # clip to valid intensity range
    return noisy_img

def add_poisson_noise(img, lam=100):
    noisy_img = img * np.random.poisson(lam, size=img.shape) / lam
    return noisy_img

# ----- PARTICLES MOVEMENT -----

class Trajectory:
    def __init__(self, id, initial_position=None, start_frame=0, initial_intensity=None):
        self.id = id
        self.positions = []
        self.intensities = []   # aligned with positions
        self.color = (random.random(), random.random(), random.random())

        self.msd = []
        self.D_trajectory = 0.0
        self.D_detection = 0.0
        self.D_localization = 0.0

        self.start_frame = start_frame
        self.end_frame = start_frame - 1

        if initial_position is not None:
            self.positions.append(tuple(initial_position))
            self.intensities.append(initial_intensity)
            self.end_frame = start_frame

    def set_id(self, id):
        self.id = id

    def add_position(self, position, frame=None, intensity=None):
        if frame is None:
            frame = self.end_frame + 1 if self.positions else self.start_frame

        if not self.positions:
            self.start_frame = frame
            self.end_frame = frame
            self.positions.append(tuple(position))
            self.intensities.append(intensity)
            return

        if frame != self.end_frame + 1:
            raise ValueError(
            f"Trajectory {self.id}: expected frame {self.end_frame + 1}, got {frame}"
        )

        self.positions.append(tuple(position))
        self.intensities.append(intensity)
        self.end_frame = frame

    def add_intensity(self, intensity, frame=None):
        """
        Optional helper if you want to modify/store intensity separately,
        but frame must already exist.
        """
        if frame is None:
            if not self.positions:
                raise ValueError("Cannot add intensity to empty trajectory")
            frame = self.end_frame

        if frame < self.start_frame or frame > self.end_frame:
            raise ValueError(f"Frame {frame} outside trajectory span")

        idx = frame - self.start_frame
        self.intensities[idx] = intensity

    def get_positions(self):
        return self.positions

    def get_intensities(self):
        return self.intensities

    def get_position_at_frame(self, frame):
        if frame < self.start_frame or frame > self.end_frame:
            return None
        return self.positions[frame - self.start_frame]

    def get_intensity_at_frame(self, frame):
        if frame < self.start_frame or frame > self.end_frame:
            return None
        return self.intensities[frame - self.start_frame]

    def frames(self):
        return list(range(self.start_frame, self.end_frame + 1))

    def last_position(self):
        return self.positions[-1] if self.positions else None

    def last_intensity(self):
        return self.intensities[-1] if self.intensities else None

    def length(self):
        return len(self.positions)

    def print_trajectory(self):
        print(f"Trajectory {self.id} (frames {self.start_frame} -> {self.end_frame}):")
        for frame, pos, intensity in zip(self.frames(), self.positions, self.intensities):
            print(frame, float(pos[0]), float(pos[1]), intensity)

# old    
# def update(particles):
#     particles_updated = np.empty((0, 4))
#     for i in range(len(particles)):
#         tag, x_c, y_c, d = particles[i]
#         theta = random.uniform(0, 2*np.pi) # random direction
#         x_c += d*np.cos(theta) # diffusion coeff * random direction (cos)
#         y_c += d*np.sin(theta) # diffusion coeff * random direction (sin) (SIGN?)
#         particles_updated = np.append(particles_updated, [(tag, x_c, y_c, d)], axis=0)
        
#     return particles_updated

def update(Particles):
    for p in Particles:
        theta = random.uniform(0, 2*np.pi) # random direction
        p.center = (p.center[0] + p.d * np.cos(theta), p.center[1] + p.d * np.sin(theta))
    return Particles

def generate_frames_OOP(F, N, amp = 1000, noise_gaussian=False, noise_poisson=False):
    particles = generate_particles_OOP(N) # a list of Particle objects
    frames = []
    trajectories_GT = [Trajectory(p.ID, start_frame=0) for p in particles] # GT trajectories for each particle
    D_GT = [p.d for p in particles] # GT diffusion coefficients for each particle
    for f in range(F):
        # print every 10 frames
        if f % 10 == 0:
            print(f"Frame {f}")
        img = uniform_bg(128, value=100) # background
        if noise_gaussian:
            img = add_gaussian_noise(img, mean=0, std=30)
        if noise_poisson:
            img = add_poisson_noise(img)
        img_with_particles = place_particles(img, particles, frame=f, c=amp)
        frames.append(img_with_particles)
        particles = update(particles)
        for i, p in enumerate(particles):
            trajectories_GT[i].add_position(p.center, frame=f)
    return frames, trajectories_GT, D_GT

def generate_frames(F, N, D=None, amp = 1000, noise_gaussian=False, noise_poisson=False):
    particles = generate_particles(N, D=D) # a list of Particle objects
    frames = []
    trajectories_GT = [Trajectory(p.ID, start_frame=0) for p in particles] # GT trajectories for each particle
    D_GT = [p.d for p in particles] # GT diffusion coefficients for each particle
    for f in range(F):
        # print every 10 frames
        if f % 10 == 0:
            print(f"Frame {f}")
        img = uniform_bg(128, value=100) # background
        if noise_gaussian:
            img = add_gaussian_noise(img, mean=0, std=30)
        if noise_poisson:
            img = add_poisson_noise(img)
        img_with_particles = place_particles(img, particles, frame=f, c=amp)
        frames.append(img_with_particles)
        particles = update(particles)
        for i, p in enumerate(particles):
            trajectories_GT[i].add_position(p.center, frame=f)
    return frames, trajectories_GT, D_GT

# def generate_frames_setD(F, N, D, amp = 1000, noise_gaussian=False, noise_poisson=False):
#     particles = generate_particles_setD(N, D) # a list of Particle objects
#     frames = []
#     trajectories_GT = [Trajectory(p.ID, start_frame=0) for p in particles] # GT trajectories for each particle
#     D_GT = [p.d for p in particles] # GT diffusion coefficients for each particle
#     for f in range(F):
#         if F > 10 and f % 10 == 0: # print count every 10 frames
#             print(f"Frame {f}")
#         else:
#             print(f"Frame {f}")
#         img = uniform_bg(128, value=100) # background
#         if noise_gaussian:
#             img = add_gaussian_noise(img, mean=100, std=30)
#         if noise_poisson:
#             img = add_poisson_noise(img)
#         img_with_particles = place_particles(img, particles, frame=f, c=amp)
#         frames.append(img_with_particles)
#         particles = update(particles)
#         for i, p in enumerate(particles):
#             trajectories_GT[i].add_position(p.center, frame=f)
#     return frames, trajectories_GT, D_GT

def generate_frames_random(F, N, D = None, amp = 1000, noise_gaussian=False, noise_poisson=False):
    particles = generate_particles(N, D=D) # a list of Particle objects
    frames = []
    trajectories_GT = [Trajectory(p.ID, start_frame=0) for p in particles] # GT trajectories for each particle
    D_GT = [p.d for p in particles] # GT diffusion coefficients for each particle
    for f in range(F):
        # print every 10 frames
        if f % 10 == 0:
            print(f"Frame {f}")
        img = uniform_bg(128, value=100) # background
        if noise_gaussian:
            img = add_gaussian_noise(img, mean=0, std=30)
        if noise_poisson:
            img = add_poisson_noise(img)
        img_with_particles = place_particles(img, particles, frame=f, c=amp)
        frames.append(img_with_particles)
        particles = update(particles)
        for i, p in enumerate(particles):
            trajectories_GT[i].add_position(p.center, frame=f)

    # randomly crop some trajectories in the first 30% and last 30% of frames
    for traj in trajectories_GT:
        L = traj.length()
        if L == 0:
            continue

        start = traj.start_frame
        end = traj.end_frame

        # crop start
        if random.random() < 0.4:
            start_crop = random.randint(1, max(1, int(0.3 * L)))
            start += start_crop
            print(f"Trajectory {traj.id} cropped at start by {start_crop} frames")

        # crop end
        if random.random() < 0.4:
            max_end_crop = max(1, int(0.3 * L))
            end_crop = random.randint(1, max_end_crop)
            end -= end_crop
            print(f"Trajectory {traj.id} cropped at end by {end_crop} frames")

        # keep only valid fragment if still non-empty
        if start <= end:
            local_start = start - traj.start_frame
            local_end = end - traj.start_frame + 1
            traj.positions = traj.positions[local_start:local_end]
            traj.start_frame = start
            traj.end_frame = end
        else:
            traj.positions = []
            traj.start_frame = 0
            traj.end_frame = -1

    return frames, trajectories_GT, D_GT

def generate_frames_blinking(F, N, D=None, amp=1000, noise_gaussian=False, noise_poisson=False,
                             blink_prob=0.02, max_blink_len=2):
    particles = generate_particles(N, D=D)

    for p in particles:
        p.blink_prob = blink_prob
        p.max_blink_len = max_blink_len
        p.visible = True
        p.remaining_blink = 0

    frames = []
    trajectories_GT = [Trajectory(p.ID, start_frame=0) for p in particles]

    for f in range(F):

        if f % 10 == 0:
            print(f"Frame {f}")
    
        img = uniform_bg(128, value=100)

        for p in particles:
            update_visibility(p)

        if noise_gaussian:
            img = add_gaussian_noise(img, mean=0, std=30)
        if noise_poisson:
            img = add_poisson_noise(img)

        img_with_particles = place_particles(img, particles, frame=f, c=amp)
        frames.append(img_with_particles)

        # store full physical GT
        for i, p in enumerate(particles):
            trajectories_GT[i].add_position(p.center, frame=f)

        particles = update(particles)

    D_GT = [p.d for p in particles]
    return frames, trajectories_GT, D_GT

def crop_trajectory(traj, new_start_frame, new_end_frame):
    """
    Helper to create a cropped copy of a trajectory between absolute frames [new_start_frame, new_end_frame].
    """
    cropped = Trajectory(traj.id, start_frame=new_start_frame)
    start = max(new_start_frame, traj.start_frame)
    end = min(new_end_frame, traj.end_frame)

    if start > end:
        cropped.start_frame = 0
        cropped.end_frame = -1
        return cropped

    cropped.positions = [
        traj.get_position_at_frame(f) for f in range(start, end + 1)
    ]
    cropped.start_frame = start
    cropped.end_frame = end
    return cropped


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

def brownian_motion_adapted(
    nparticles,
    nframes,
    nposframe,
    D,
    dt,
    startAtZero=False,
    bg_size=(128, 128),
    intensity_mean=1000,
    intensity_std=50
):
    """
    Simulates Brownian motion and returns a list of Trajectory objects
    at subframe resolution.

    Each Trajectory contains one position per substep, so its frame axis is:
        0, 1, 2, ..., nframes * nposframe - 1
    """
    num_steps = nframes * nposframe
    sigma = np.sqrt(2 * D * dt / nposframe)

    trajectories = []

    for p in range(nparticles):
        # Brownian increments
        dxy = np.random.randn(num_steps, 2) * sigma

        # Build positions
        positions = np.cumsum(dxy, axis=0)

        if startAtZero:
            positions[0] = [0.0, 0.0]
        else:
            start_x = np.random.uniform(10, bg_size[0]-10) # far from borders
            start_y = np.random.uniform(10, bg_size[1]-10)
            positions += np.array([start_x, start_y])

        # Create Trajectory object
        traj = Trajectory(p, start_frame=0)

        for t in range(num_steps):
            base_intensity = max(np.random.normal(intensity_mean, intensity_std), 0.0)
            traj.add_position(tuple(positions[t]), frame=t, intensity=base_intensity)

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
        "particle_intensity": [1000, 50],
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

def save_video_as_gif(out_video, filename="movie.gif", fps=20):
    # Normalize frames to 0–255 uint8
    vid = out_video.copy()
    vid -= vid.min()
    vid /= vid.max()
    vid = (vid * 255).astype(np.uint8)

    frames = [vid[i] for i in range(vid.shape[0])]
    imageio.mimsave(filename, frames, fps=fps)




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

def animate_video(video, interval=300, cmap="gray", vmin=None, vmax=None):
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

