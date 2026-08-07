import numpy as np


def test_front_view_camera_preset():
    import mujoco
    from rby1_bringup.pi05_infer import configure_view_camera

    camera = mujoco.MjvCamera()
    configure_view_camera(camera, "front")

    np.testing.assert_allclose(camera.lookat, [0.45, 0.0, 0.85])
    assert camera.distance == 1.7
    assert camera.azimuth == 180.0
    assert camera.elevation == -18.0
