"""Generate fixed MJCF: kinematic vehicle, five sites, mocap obstacle pool, no vehicle–obstacles contact."""

from __future__ import annotations

# Vehicle box half-sizes (m), +X forward, +Y left, +Z up (body frame at geom center)
_HX, _HY, _HZ = 0.5, 0.2, 0.12
_Z_COM = 0.15
_TIMESTEP = 0.02

SLOT_NAMES = (
    "top_front_edge_l",
    "top_front_edge_r",
    "front_face_center",
    "side_front_edge_l",
    "side_front_edge_r",
)

# Local site positions on the box (on surfaces / edges)
# Front face is x = +_HX
_P_SITE = {
    "top_front_edge_l": (_HX, 0.6 * _HY, _HZ),
    "top_front_edge_r": (_HX, -0.6 * _HY, _HZ),
    "front_face_center": (_HX, 0.0, 0.0),
    "side_front_edge_l": (0.4 * _HX, _HY, 0.0),
    "side_front_edge_r": (0.4 * _HX, -_HY, 0.0),
}


def build_vehicle_mjcf(n_obstacles: int = 10) -> str:
    """
    One slide joint moves the vehicle along +X. Obstacles are mocap bodies; parking far away
    when unused. Contact excludes keep vehicle from interacting with obstacles.
    """
    n_obstacles = int(max(1, n_obstacles))
    lines: list[str] = [
        f'<mujoco model="vehicle_sensor_opt">',
        f'  <compiler angle="radian" autolimits="true"/>',
        f'  <option timestep="{_TIMESTEP}" gravity="0 0 -9.81" integrator="RK4"/>',
        f"  <worldbody>",
        f'    <light pos="0 0 8" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>',
        f'    <geom name="floor" type="plane" pos="0 0 0" size="20 20 0.01" rgba="0.7 0.7 0.7 1"/>',
        f'    <body name="vehicle" pos="0 0 {_Z_COM}">',
        f'      <joint name="vehicle_tx" type="slide" axis="1 0 0" pos="0 0 0" range="0 100"/>',
        f'      <geom name="vehicle_geom" type="box" size="{_HX} {_HY} {_HZ}" mass="1" rgba="0.15 0.4 0.8 0.7"/>',
    ]
    for slot in SLOT_NAMES:
        px, py, pz = _P_SITE[slot]
        lines.append(
            f'      <site name="{slot}" type="sphere" size="0.02" pos="{px} {py} {pz}" rgba="0.1 0.8 0.2 0.5"/>'
        )
    lines.append(f"    </body>")
    for i in range(n_obstacles):
        lines.append(
            f'    <body name="obst_{i}" mocap="true" pos="1000 0 {_Z_COM}">'
            f'      <geom name="obst_geom_{i}" type="sphere" size="0.25" rgba="0.85 0.2 0.2 0.8"/>'
            f"    </body>"
        )
    lines.append(f"  </worldbody>")
    lines.append(f"  <contact>")
    for i in range(n_obstacles):
        lines.append(
            f'    <exclude body1="vehicle" body2="obst_{i}"/>'
        )
    lines.append(f"  </contact>")
    lines.append(f"</mujoco>")
    return "\n".join(lines) + "\n"
