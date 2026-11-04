"""Offline analysis of the hardware campaign (hardware protocol): bags -> datasets -> calibration and adequacy tests.

    io            read the bags of a session (rosbags, no ROS install needed), merge them on header stamps
    frames        rigid wrench transforms (reference point, coordinates)
    boss_solve    statically determinate boss forces from the ATI, image <-> surface registration (H2)
    depth         offline GelSight reconstruction from raw images with the session's background and model
    pressure_fit  depth -> pressure maps fitted on boss forces (H2)
    latency       camera delay and measurement ages (H1)

Conventions. Contact frame = `gelsight_contact_surface`: origin on the gel surface, y along the
sensor normal, (x, z) spanning the surface. Wrenches are (fx, fy, fz, tx, ty, tz), SI units,
expressed in a named frame and taken about its origin.
"""
