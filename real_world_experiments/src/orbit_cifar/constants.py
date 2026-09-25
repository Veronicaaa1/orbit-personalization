from __future__ import annotations

from typing import Final

NUM_NODES: Final = 14
NUM_CLASSES: Final = 10  # Legacy CIFAR-10 compatibility alias; active code derives C from data.
FULLY_SYMMETRIC_PROFILE_NAME: Final = "balanced_global"
FULLY_SYMMETRIC_CLASS_COUNTS: Final = (200,) * NUM_CLASSES
GLOBAL_SYMMETRY_ORBITS: Final[tuple[tuple[int, ...], ...]] = (tuple(range(NUM_NODES)),)
REFERENCE_ORBITS: Final[tuple[tuple[int, ...], ...]] = (
    (0, 1),
    (2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12, 13),
)
ORBIT_SIZES: Final = (2, 3, 4, 5)
CIFAR10_CLASSES: Final = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
CLASS_PROFILES: Final[dict[str, tuple[int, ...]]] = {
    "profile_1": (600, 500, 200, 160, 140, 120, 100, 80, 60, 40),
    "profile_2": (60, 40, 600, 500, 200, 160, 140, 120, 100, 80),
    "profile_3": (100, 80, 60, 40, 600, 500, 200, 160, 140, 120),
    "profile_4": (200, 160, 140, 120, 100, 80, 60, 40, 600, 500),
}
ALIGNED_PROFILE_NAMES: Final = (
    "profile_1",
    "profile_1",
    "profile_2",
    "profile_2",
    "profile_2",
    "profile_3",
    "profile_3",
    "profile_3",
    "profile_3",
    "profile_4",
    "profile_4",
    "profile_4",
    "profile_4",
    "profile_4",
)
ALIGNED_NODE_TO_BUNDLE: Final = tuple(range(NUM_NODES))
MISALIGNED_NODE_TO_BUNDLE: Final = (0, 9, 2, 5, 10, 1, 3, 6, 11, 4, 7, 8, 12, 13)
CIFAR10_MEAN: Final = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: Final = (0.2470, 0.2435, 0.2616)
MNIST_CLASSES: Final = tuple(str(value) for value in range(10))
MNIST_MEAN: Final = (0.1307,)
MNIST_STD: Final = (0.3081,)
ACTIVE_DATASETS: Final = ("mnist", "cifar10")
ACTIVE_MODELS: Final = ("resnet20_gn", "mlp_ln")
ACTIVE_CONDITIONS: Final = ("main", "no_communication", "misaligned_data", "fully_symmetric")
ACTIVE_SEEDS: Final = (1, 2, 3)
EXPERIMENT_NAME: Final = "mnist_cifar10_resnet_mlp_3seed"


def validate_reference_orbits(orbits: tuple[tuple[int, ...], ...] = REFERENCE_ORBITS) -> None:
    """Ensure a partition covers each internal node index exactly once."""
    flat = [node for orbit in orbits for node in orbit]
    if sorted(flat) != list(range(NUM_NODES)):
        raise ValueError("Reference orbits must partition node indices 0 through 13 exactly once")
