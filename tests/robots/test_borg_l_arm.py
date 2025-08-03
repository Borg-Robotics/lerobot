from contextlib import contextmanager
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from lerobot.robots.borg_l_arm import (
    BorgLArm,
    BorgLArmConfig,
)

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera


def _make_bus_mock() -> MagicMock:
    """Return a bus mock with just the attributes used by the robot."""
    bus = MagicMock(name="FeetechBusMock")
    bus.is_connected = False

    def _connect():
        bus.is_connected = True

    def _disconnect(_disable=True):
        bus.is_connected = False

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect

    @contextmanager
    def _dummy_cm():
        yield

    bus.torque_disabled.side_effect = _dummy_cm

    return bus

def _camera_connect_stub(self, warmup=True):
    self.is_connected = True


@pytest.fixture
def follower():
    bus_mock = _make_bus_mock()

    def _bus_side_effect(*_args, **kwargs):
        bus_mock.motors = kwargs["motors"]
        motors_order: list[str] = list(bus_mock.motors)

        bus_mock.sync_read.return_value = {motor: idx for idx, motor in enumerate(motors_order, 1)}
        bus_mock.sync_write.return_value = None
        bus_mock.write.return_value = None
        bus_mock.disable_torque.return_value = None
        bus_mock.enable_torque.return_value = None
        bus_mock.is_calibrated = True
        return bus_mock

    with (
        patch(
            "lerobot.robots.borg_l_arm.borg_l_arm.FeetechMotorsBus",
            side_effect=_bus_side_effect,
        ),
        patch.object(BorgLArm, "configure", lambda self: None),
        patch(
            "lerobot.robots.borg_l_arm.borg_l_arm.make_cameras_from_configs",
            return_value={
                "cam1": MagicMock(is_connected=True, async_read=lambda: "fake_image"),
                "cam2": MagicMock(is_connected=True, async_read=lambda: "fake_image"),
            },
        ),
    ):
        cfg = BorgLArmConfig(port="/dev/null")
        robot = BorgLArm(cfg)
        yield robot
        if robot.is_connected:
            robot.disconnect()


def test_connect_disconnect(follower):
    assert not follower.is_connected

    follower.connect()
    assert follower.is_connected

    follower.disconnect()
    assert not follower.is_connected


def test_get_observation(follower):
    follower.connect()
    obs = follower.get_observation()

    motor_keys = {f"{m}.pos" for m in follower.bus.motors}
    obs_motor_keys = {k for k in obs if k.endswith(".pos")}
    assert obs_motor_keys == motor_keys

    for idx, motor in enumerate(follower.bus.motors, 1):
        assert obs[f"{motor}.pos"] == idx


def test_send_action(follower):
    follower.connect()

    action = {f"{m}.pos": i * 10 for i, m in enumerate(follower.bus.motors, 1)}
    returned = follower.send_action(action)

    assert returned == action

    goal_pos = {m: (i + 1) * 10 for i, m in enumerate(follower.bus.motors)}
    follower.bus.sync_write.assert_called_once_with("Goal_Position", goal_pos)
