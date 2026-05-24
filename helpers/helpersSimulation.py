from helpers.helpersGeneration import *
from helpers.helpersTracking import *
from helpers.helpersStatistics import *

class Simulator:
    def __init__(self, simulation_config, image_config=None, seed=None):
        self.simulation_config = simulation_config
        self.image_config = image_config or {}
        self.seed = seed

        
        self.video = None
        self.frames = None
        self.trajectories_HR = None
        self.trajectories_GT = None
        self.trajectories_FR = None

    def run(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        self.trajectories_HR = simulate_brownian_motion(
            **self.simulation_config
        )

        self.video = trajectories_to_global_video(
            self.trajectories_HR,
            nframes=self.simulation_config["nframes"],
            nPosPerFrame=self.simulation_config["nposframe"],
            image_props=self.image_config,
        )

        self.trajectories_GT = self.create_GT_trajectories(
            self.trajectories_HR,
            self.simulation_config["nposframe"]
        )

        self.trajectories_FR = self.create_FR_trajectories(
            self.trajectories_GT
        )

        self.frames = self.video.astype(np.float32)

        return self

    def get_outputs(self):
        return self.frames, self.trajectories_HR, self.trajectories_GT, self.video
    
    def reset(self):
        """Clear generated outputs while keeping the configuration."""
        self.trajectories_HR = None
        self.video = None
        self.frames = None
        self.trajectories_GT = None
        return self
    
    def rerun(self, seed=None):
        """Reset and rerun the simulation, optionally with a new seed."""
        if seed is not None:
            self.seed = seed

        self.reset()
        return self.run()
    
    def show_frame(self, frame_idx=0, vmin=0, vmax=5000):
        """Display one simulated frame."""
        if self.frames is None:
            raise ValueError("Simulation has not been run yet.")

        plt.imshow(self.frames[frame_idx], cmap="gray", vmin=vmin, vmax=vmax)
        plt.title(f"Frame {frame_idx}")
        plt.axis("off")
        plt.show()

    def animate(self, interval=100, cmap="gray", vmin=0, vmax=1000):
        """Animate the simulated video."""
        if self.video is None:
            raise ValueError("Simulation has not been run yet.")

        return animate_video(
            self.video,
            interval=interval,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
    
    def save_gif(self, filename="movie.gif", fps=10):
        """Save the simulated video as a GIF."""
        if self.video is None:
            raise ValueError("Simulation has not been run yet.")

        save_video_as_gif(self.video, filename=filename, fps=fps)

    # create trajectories_GT, the upsampled version of trajectories for comparison with detected and localized trajectories
    # current trajectories are of length nframes * nposframe, but we want to compare with trajectories at frame resolution (nframes)
    def create_GT_trajectories(self, trajectories, nposframe):
        trajectories_GT = copy.deepcopy(trajectories)
        output_size = self.image_config.get(
            "output_size",
            self.simulation_config.get("frame_size", (128, 128))[1],
        )

        for traj in trajectories_GT:
            traj.positions = traj.positions[::nposframe]
            traj.position_frames = list(
                range(traj.start_frame, traj.start_frame + len(traj.positions))
            )
            traj.intensities = traj.intensities[::nposframe]
            traj.sigmas = traj.sigmas[::nposframe]
            traj.states = traj.states[::nposframe]
            traj.bound_to = traj.bound_to[::nposframe]
            traj.positions = [(output_size-pos[1], pos[0]) for pos in traj.positions]
            traj.end_frame = traj.start_frame + len(traj.positions) - 1
        return trajectories_GT

    def create_FR_trajectories(self, trajectories_GT):
        trajectories_FR = copy.deepcopy(trajectories_GT)
        MSD_FR = calculateMSDtrajectories(trajectories_FR)
        D_FR_results = estimateDfromTrajectories(trajectories_FR)    
        return trajectories_FR

    def histogram(
        self,
        bins=100,
        log=False,
        density=False,
        vmin=None,
        vmax=None,
        percentile_range=None,
    ):
        """
        Plot the histogram of pixel intensities across all frames.

        Parameters
        ----------
        bins : int
            Number of histogram bins.
        log : bool
            If True, use a logarithmic y-axis.
        density : bool
            If True, normalize to probability density.
        vmin, vmax : float or None
            Optional lower and upper intensity limits.
        percentile_range : tuple or None
            Optional percentile crop, e.g. (95, 100).
        """
        if self.frames is None:
            raise ValueError("Simulation has not been run yet.")

        pixels = np.asarray(self.frames).ravel()

        if percentile_range is not None:
            pmin, pmax = percentile_range
            vmin = np.percentile(pixels, pmin)
            vmax = np.percentile(pixels, pmax)

        if vmin is not None:
            pixels = pixels[pixels >= vmin]

        if vmax is not None:
            pixels = pixels[pixels <= vmax]

        plt.figure(figsize=(8, 5))

        plt.hist(
            pixels,
            bins=bins,
            density=density,
        )

        plt.xlabel("Pixel intensity")
        plt.ylabel("Frequency" if not density else "Density")

        title = "Pixel intensity histogram"
        if vmin is not None or vmax is not None:
            title += f" cropped to [{vmin}, {vmax}]"

        plt.title(title)

        if log:
            plt.yscale("log")

        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    
    def track(self, **kwargs):
        """Run the full tracking pipeline on the simulation."""
        if self.frames is None or self.trajectories_GT is None:
            raise ValueError("Simulation has not been run yet.")

        return track(
            self.frames,
            self.trajectories_GT,
            **kwargs,
        )
    

class LigandReceptorSimulator(Simulator):
    def __init__(self, simulation_config, image_config=None, seed=None):
        super().__init__(simulation_config, image_config=image_config, seed=seed)
        self.binding_events = []

    def run(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        self.trajectories_HR, self.binding_events = simulate_ligand_receptor_motion(
            **self.simulation_config,
            return_events=True,
        )

        self.video = trajectories_to_global_video(
            self.trajectories_HR,
            nframes=self.simulation_config["nframes"],
            nPosPerFrame=self.simulation_config["nposframe"],
            image_props=self.image_config,
        )

        self.trajectories_GT = self.create_GT_trajectories(
            self.trajectories_HR,
            self.simulation_config["nposframe"],
        )

        self.trajectories_FR = self.create_FR_trajectories(
            self.trajectories_GT
        )

        self.frames = self.video.astype(np.float32)

        return self

    def reset(self):
        super().reset()
        self.binding_events = []
        return self

    def get_ligands(self, resolution="GT"):
        trajectories = self.trajectories_GT if resolution == "GT" else self.trajectories_HR
        if trajectories is None:
            raise ValueError("Simulation has not been run yet.")
        return [traj for traj in trajectories if traj.particle_type == "ligand"]

    def get_receptors(self, resolution="GT"):
        trajectories = self.trajectories_GT if resolution == "GT" else self.trajectories_HR
        if trajectories is None:
            raise ValueError("Simulation has not been run yet.")
        return [traj for traj in trajectories if traj.particle_type == "receptor"]
