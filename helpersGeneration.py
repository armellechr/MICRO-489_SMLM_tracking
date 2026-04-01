import numpy as np
import random
import matplotlib.pyplot as plt
import stackview

# ----- BACKGROUND GENERATION -----

def uniform_bg(size, value=100):
    return np.full((size, size), value)

def background(x,y):
    return 100 * np.ones_like(x) # 100 to mimic microscope background intensity

def background1(x,y):
    sigma = 0.5
    return 100 * np.ones_like(x) +  5000 * (1/(2*np.pi*sigma**2)) * np.exp(-((x-0)**2 + (y-0)**2)/(2*sigma**2))

def background2(x,y):
    sigma = 0.5
    return 100 * np.ones_like(x) +  5000 * (1/(2*np.pi*sigma**2)) * np.exp(-((x-1)**2 + (y-1)**2)/(2*sigma**2))

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

# OOP approach
# class Particle:
#     def __init__(self, ID, row_c, col_c, d):
#         self.ID = ID
#         self.center = (row_c, col_c)   # image convention: (row, col)
#         self.d = d
#         self.color = (random.random(), random.random(), random.random())

#     def __str__(self):
#         return str(self.ID)

class Particle:
    def __init__(self, ID, row_c, col_c, d, start_frame=0, end_frame=None):
        self.ID = ID
        self.center = (row_c, col_c)
        self.d = d
        self.color = (random.random(), random.random(), random.random())
        self.start_frame = start_frame
        self.end_frame = end_frame
    
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


# def place_particles(img, particles, c = 5000, sigma = 0.5):
#     output = img.copy()
#     for i in range(len(particles)):
#         print(f"Particle {i}: row_c={particles[i, 1]}, col_c={particles[i, 2]}, d={particles[i, 3]}")
#         x_c, y_c = particles[i, 1], particles[i, 2]
#         for x in range(img.shape[0]):
#             for y in range(img.shape[1]):
#                 output[x, y] += g(c, x, y, x_c, y_c, sigma)
#     return output

# def place_particles(img, particles, c = 1000, sigma = 0.5):
#     output = img.copy()
#     for p in particles:
#         x_c, y_c = p.center
#         for x in range(img.shape[0]):
#             for y in range(img.shape[1]):
#                 output[x, y] += g(c, x, y, x_c, y_c, sigma)
#     return output

def place_particles(img, particles, frame, c=1000, sigma=0.5):
    output = img.copy()
    for p in particles:
        if frame < p.start_frame:
            continue
        if p.end_frame is not None and frame > p.end_frame:
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

# class Trajectory:
#     def __init__(self, id, initial_position=None):
#         self.id = id # particle ID
#         self.positions = [] # list of (x, y)
#         if initial_position is not None:
#             self.positions.append(tuple(initial_position))
#         self.color = (random.random(), random.random(), random.random()) # random color for visualization
#         self.msd = [] # to store MSD values for this trajectory
#         self.D_trajectory = 0.0 # to store estimated D for this trajectory
#         self.D_detection = 0.0 # to store estimated D from detection for this trajectory
#         self.D_localization = 0.0 # to store estimated D from localization for this trajectory
#         self.start_frame = 0 # to store the frame number where this trajectory starts
#         self.end_frame = 0 # to store the frame number where this trajectory ends

#     def set_id(self, id):
#         self.id = id

#     def add_position(self, position):
#         self.positions.append(tuple(position))

#     def get_positions(self):
#         return self.positions

#     def last_position(self):
#         return self.positions[-1] if self.positions else None
    
#     def length(self):
#         return len(self.positions)
    
#     def print_trajectory(self):
#         print(f"Trajectory {self.id}:")
#         for pos in self.positions:
#             print(float(pos[0]), float(pos[1]))

class Trajectory:
    def __init__(self, id, initial_position=None, start_frame=0):
        self.id = id # trajectory ID
        self.positions = [] # list of (x, y) positions, indexed by frame number relative to start_frame
        self.color = (random.random(), random.random(), random.random()) # random color for visualization

        self.msd = [] # to store MSD values for this trajectory
        self.D_trajectory = 0.0
        self.D_detection = 0.0
        self.D_localization = 0.0

        self.start_frame = start_frame # absolute frame number where this trajectory starts
        self.end_frame = start_frame - 1   # end frame, empty trajectory by default

        if initial_position is not None: # if provided, add initial position at start_frame
            self.positions.append(tuple(initial_position))
            self.end_frame = start_frame

    def set_id(self, id):
        self.id = id

    def add_position(self, position, frame=None):
        """
        Add a position at an absolute frame number.
        If frame is None, assumes it is the next consecutive frame.
        """
        if frame is None:
            frame = self.end_frame + 1 if self.positions else self.start_frame

        if not self.positions:
            self.start_frame = frame
            self.end_frame = frame
            self.positions.append(tuple(position))
            return

        if frame != self.end_frame + 1:
            raise ValueError(
                f"Trajectory {self.id}: expected frame {self.end_frame + 1}, got {frame}"
            )

        self.positions.append(tuple(position))
        self.end_frame = frame

    def get_positions(self):
        return self.positions

    def get_position_at_frame(self, frame):
        """
        Return position at absolute frame number, or None if outside span.
        """
        if frame < self.start_frame or frame > self.end_frame:
            return None
        return self.positions[frame - self.start_frame]

    def frames(self):
        return list(range(self.start_frame, self.end_frame + 1))

    def last_position(self):
        return self.positions[-1] if self.positions else None

    def length(self):
        return len(self.positions)

    def print_trajectory(self):
        print(f"Trajectory {self.id} (frames {self.start_frame} -> {self.end_frame}):")
        for frame, pos in zip(self.frames(), self.positions):
            print(frame, float(pos[0]), float(pos[1]))
    
def update(particles):
    particles_updated = np.empty((0, 4))
    for i in range(len(particles)):
        tag, x_c, y_c, d = particles[i]
        theta = random.uniform(0, 2*np.pi) # random direction
        x_c += d*np.cos(theta) # diffusion coeff * random direction (cos)
        y_c += d*np.sin(theta) # diffusion coeff * random direction (sin) (SIGN?)
        particles_updated = np.append(particles_updated, [(tag, x_c, y_c, d)], axis=0)
        
    return particles_updated

def update_OOP(Particles):
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
        particles = update_OOP(particles)
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
        particles = update_OOP(particles)
        for i, p in enumerate(particles):
            trajectories_GT[i].add_position(p.center, frame=f)
    return frames, trajectories_GT, D_GT

def generate_frames_setD(F, N, D, amp = 1000, noise_gaussian=False, noise_poisson=False):
    particles = generate_particles_setD(N, D) # a list of Particle objects
    frames = []
    trajectories_GT = [Trajectory(p.ID, start_frame=0) for p in particles] # GT trajectories for each particle
    D_GT = [p.d for p in particles] # GT diffusion coefficients for each particle
    for f in range(F):
        if F > 10 and f % 10 == 0: # print count every 10 frames
            print(f"Frame {f}")
        else:
            print(f"Frame {f}")
        img = uniform_bg(128, value=100) # background
        if noise_gaussian:
            img = add_gaussian_noise(img, mean=100, std=30)
        if noise_poisson:
            img = add_poisson_noise(img)
        img_with_particles = place_particles(img, particles, frame=f, c=amp)
        frames.append(img_with_particles)
        particles = update_OOP(particles)
        for i, p in enumerate(particles):
            trajectories_GT[i].add_position(p.center, frame=f)
    return frames, trajectories_GT, D_GT

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
        particles = update_OOP(particles)
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
        if random.random() < 0.2:
            start_crop = random.randint(1, max(1, int(0.3 * L)))
            start += start_crop
            print(f"Trajectory {traj.id} cropped at start by {start_crop} frames")

        # crop end
        if random.random() < 0.2:
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
    

# def show_trajectory(frames, trajectories, traj_id=0, frame_id=0,save_path=None):
#     traj = trajectories[traj_id]
#     positions = np.array(traj.get_positions())

#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.imshow(frames[frame_id], cmap="gray", vmin=0, vmax=5000)

#     if len(positions) > 1:
#         # positions assumed stored as (row, col) = (y, x)
#         y = positions[:, 0]
#         x = positions[:, 1]
#         # use the set trajectory's color
#         ax.plot(x, y, '-', color=traj.color, linewidth=2) # trajectory as line
#         ax.plot(x[0], y[0], '^', color=traj.color, markersize=5) # start point as triangle
#         ax.plot(x[-1], y[-1], '+', color=traj.color, markersize=5) # end point as plus

#     ax.set_title(f"Trajectory {traj_id} on frame {frame_id+1}")
#     ax.axis("off")

#     if save_path is not None:
#         fig.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)

#     plt.show()

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

# def show_trajectories(frames, trajectories, frame_id=0, title=None,save_path=None):
#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.imshow(frames[frame_id], cmap="gray", vmin=0, vmax=5000)

#     for traj in trajectories:
#         positions = np.array(traj.get_positions())
#         if len(positions) > 1:
#             y = positions[:, 0]
#             x = positions[:, 1]
#             ax.plot(x, y, '-', color=traj.color, linewidth=2) # trajectory as line
#             ax.plot(x[0], y[0], '^', color=traj.color, markersize=5) # start point
#             ax.plot(x[-1], y[-1], '+', color=traj.color, markersize=5) # end point

#     if title:
#         ax.set_title(title)
#     else:
#         ax.set_title(f"Trajectories on frame {frame_id}")
#     ax.axis("off")

#     if save_path is not None:
#         fig.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)

#     plt.show()

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

# def linear_trajectories_visualizer(trajectories_new, trajectories_GT):
#     """Compares the span of new and GT trajectories across the series of frames, by horizontal lines where the positions are not None 
#     for given trajectory IDs.
#     GT trajectories are solid lines, new trajectories are dashed lines. The x-axis is the frame number, the y-axis is the trajectory ID.
#     """
#     plt.figure(figsize=(12, 6))
#     for traj in trajectories_GT:
#         positions = traj.get_positions()
#         frames = [i for i, pos in enumerate(positions) if pos is not None]
#         if frames:
#             plt.hlines(traj.id, min(frames), max(frames)+2, colors=traj.color, linestyles='solid', label='GT' if traj.id == 0 else "")
#     for traj in trajectories_new:
#         positions = traj.get_positions()
#         frames = [i for i, pos in enumerate(positions) if pos is not None]
#         if frames and traj.id is not None and traj.id != -1:
#             # shift horizontally for visualization
#             plt.hlines(traj.id + 0.2, min(frames), max(frames)+2, colors=traj.color, linestyles='dashed', label='New' if traj.id == 0 else "")
#     plt.xlabel('Frame Number')
#     plt.ylabel('Trajectory ID')
#     plt.ylim(-1, max(max(traj.id for traj in trajectories_GT), max(traj.id for traj in trajectories_new if traj.id is not None and traj.id != -1)) + 1)
#     plt.title('Comparison of GT and New Trajectories')
#     plt.legend()
#     plt.grid()
#     plt.show()

def linear_trajectories_visualizer(trajectories_new, trajectories_GT):
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

    plt.title('Comparison of GT and New Trajectories')
    plt.legend()
    plt.grid()
    plt.show()
