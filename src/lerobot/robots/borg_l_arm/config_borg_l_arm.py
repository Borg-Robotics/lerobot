from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("borg_l_arm")
@dataclass
class BorgLArmConfig(RobotConfig):
    # Port to connect to the arm
    port: str
    
    disable_torque_on_disconnect: bool = True
    
    # `max_relative_target` limits the magnitude of the relative positional
    # target vector for safety purposes. Set this to a positive scalar to have
    # the same value for all motors, or a list that is the same length as the
    # number of motors in your follower arms.
    max_relative_target: int | None = None
    
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "cam_1": OpenCVCameraConfig(
                index_or_path=2,
                fps=30,
                width=480,
                height=640,
            ),
        }
    )
