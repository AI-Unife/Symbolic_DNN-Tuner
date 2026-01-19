import os
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from tonic.dataset import Dataset  # type: ignore
from tonic.datasets import DVSGesture  # type: ignore
from tonic.transforms import Compose, Downsample, ToFrame  # type:ignore


class DVSGestureROI(DVSGesture):
    """A modified version of the DVSGesture eventset that returns regions-of-interest (ROIs) and their coordinates."""

    sensor_size = (32, 32, 2)
    dtype = np.dtype(
        [("x", np.int16), ("y", np.int16), ("p", bool), ("t", np.int64)]
    )
    dtype_position = np.dtype(
        [
            ("x", np.int16),
            ("y", np.int16),
            ("s", np.int16),
            ("p", bool),
            ("t", np.float32),
        ]
    )
    ordering = dtype.names

    def __init__(
        self,
        save_to: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        position_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        Dataset.__init__(
            self,
            save_to,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )
        self.position_transform = position_transform
        self.train = train

        if train:
            self.folder_name = "train"
        else:
            self.folder_name = "test"

        self.users = []
        self.lighting = []
        self.position = []
        file_path = os.path.join(self.location_on_system, self.folder_name)
        for path, dirs, files in os.walk(file_path):
            rel_path = os.path.relpath(path, file_path)
            if rel_path != ".":
                user, lighting = rel_path.split("_", 1)
                user = int(user[4:])
                dirs.sort()
                for file in files:
                    if file.endswith(".npy"):
                        if file.endswith("_positions.npy"):
                            continue
                        self.data.append(os.path.join(path, file))
                        self.targets.append(int(file[:-4]))
                        pos_file = file.removesuffix(".npy") + (
                            "_positions.npy"
                        )
                        self.position.append(os.path.join(path, pos_file))
                        self.users.append(user)
                        self.lighting.append(lighting)

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (events, target, position) where target is the index of the target class and position is the position of the ROI in the input.
        """
        events = np.load(self.data[index])
        target = self.targets[index]
        position = np.load(self.position[index])
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.position_transform is not None:
            position = self.position_transform(position)
        if self.transforms is not None:
            events, target, position = self.transforms(
                events, target, position
            )

        return events, target, position

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return (
            self._is_file_present()
            and self._folder_contains_at_least_n_files_of_type(100, ".npy")
        )


class ROIMapTransform(object):
    """Transform ROI position events into frames for downstream processing"""

    def __init__(
        self,
        full_input_size: tuple = (128, 128, 2),  # DVSGesture size
        output_size: tuple = (32, 32, 1),  # Target size for downstream
        time_window: int = None,
        n_time_bins: int = None,
    ):
        self.downsample_transform = Downsample(
            sensor_size=full_input_size[:2] + (1,), target_size=output_size[:2]
        )
        if time_window is not None:  
            self.frame_transform = ToFrame(
                sensor_size=output_size,
                time_window=time_window,
            )
        elif n_time_bins is not None:
            self.frame_transform = ToFrame(
                sensor_size=output_size,
                n_time_bins=n_time_bins,
            )
        else:
            self.frame_transform = ToFrame(
                sensor_size=output_size,
                time_window=1000,
                )
        self.transform = Compose(
            [self.downsample_transform, self.frame_transform]
        )

    def __call__(self, events):
        return self.transform(events)


def plot_roi_animation(
    full_frames: np.ndarray, event_frames: np.ndarray, map_frames: np.ndarray
):
    # Plot full frame input, roi and map side-by-side
    # Adapted from tonic

    fig, ax = plt.subplots(nrows=1, ncols=3)

    def rearrange_frames(frames):
        if frames.shape[1] == 2:
            rgb = np.zeros((frames.shape[0], 3, *frames.shape[2:]))
            rgb[:, 1:, ...] = frames
            frames = rgb
        if frames.shape[1] in [1, 2, 3]:
            frames = np.moveaxis(frames, 1, 3)
        return frames

    full_frames = rearrange_frames(full_frames)
    event_frames = rearrange_frames(event_frames)
    map_frames = rearrange_frames(map_frames)

    ax0 = ax[0].imshow(full_frames[0])
    ax1 = ax[1].imshow(event_frames[0])
    ax2 = ax[2].imshow(map_frames[0])

    ax[0].set_title("Full input")
    ax[1].set_title("ROI")
    ax[2].set_title("Map of ROI position")

    ax[0].axis("off")
    ax[1].axis("off")
    ax[2].axis("off")

    def animate(frame):
        ax0.set_data(frame[0])
        ax1.set_data(frame[1])
        ax2.set_data(frame[2])
        return ax

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=[
            (full_frames[i], event_frames[i], map_frames[i])
            for i in range(
                min(
                    full_frames.shape[0],
                    event_frames.shape[0],
                    map_frames.shape[0],
                )
            )
        ],
        interval=100,
    )
    plt.show(block=True)
    return anim


if __name__ == "__main__":
    # Comparing the full resolution input with the ROI and its location map

    # Set this to the path where DVSGesture and DVSGestureROI are
    # First paths will be:
    #   - datasets/DVSGesture/ibmGestureTrain/user23_fluorescent/9.npy
    #   - datasets/DVSGestureROI/train/user23_fluorescent/9.npy

    DVSGESTURE_PATH = "rois_and_coordinates/datasets"

    # N.B. This can be changed but then you will see many ROIs overlapping in one frame.

    time_window = None #10000
    n_time_bins = 64
    
    dvs_training = DVSGesture(
        DVSGESTURE_PATH,
        train=False,
        transform=ToFrame(n_time_bins=n_time_bins, time_window=time_window, sensor_size=(128, 128, 2)),
    )

    dvs_training_roi = DVSGestureROI(
        str(DVSGESTURE_PATH),
        train=True,
        transform=ToFrame(n_time_bins=n_time_bins, time_window=time_window, sensor_size=(32, 32, 2)),
        position_transform=ROIMapTransform(n_time_bins=32),
    )
    print("Length of training dataset:", len(dvs_training))
    print("Length of training ROI dataset:", len(dvs_training_roi))
    exit()
    for target in range(5):
        index = dvs_training.targets.index(target)

        rois, roi_target, roi_position = dvs_training_roi[index]

        full, full_target = dvs_training[index]

        plot_roi_animation(full, rois, roi_position)
