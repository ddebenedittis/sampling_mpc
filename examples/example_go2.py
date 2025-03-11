from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx


from hydrax.algs import MPPI
from hydrax.simulation.asynchronous import run_interactive as run_async
from hydrax.simulation.deterministic import run_interactive
from hydrax.task_base import Task

class Go2(Task):
    """TODO"""

    def __init__(
        self, planning_horizon: int = 3, sim_steps_per_control_step: int = 10
    ) -> None:
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            "assets/mujoco_menagerie/unitree_go2/scene_mjx.xml"
        )

        super().__init__(
            mj_model,
            planning_horizon=planning_horizon,
            sim_steps_per_control_step=sim_steps_per_control_step,
            trace_sites=["imu", "FL_foot", "FR_foot", "RL_foot", "RR_foot"],
        )
        
        # Get sensor and site ids
        self.orientation_sensor_id = mj_model.sensor("orientation").id
        self.velocity_sensor_id = mj_model.sensor("global_linvel").id
        self.base_id = mj_model.site("imu").id

        # Set the target height
        self.target_height = 0.3

        # home configuration
        self.qstand = jnp.array(mj_model.keyframe("home").qpos)

    def _get_base_height(self, state: mjx.Data) -> jax.Array:
        """Get the height of the base above the ground."""
        return state.site_xpos[self.base_id, 2]

    def _get_base_orientation(self, state: mjx.Data) -> jax.Array:
        """Get the rotation from the current base orientation to upright."""
        sensor_adr = self.model.sensor_adr[self.orientation_sensor_id]
        quat = state.sensordata[sensor_adr : sensor_adr + 4]
        upright = jnp.array([0.0, 0.0, 1.0])
        return mjx._src.math.rotate(upright, quat)

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        orientation_cost = jnp.sum(
            jnp.square(self._get_base_orientation(state))
        )
        height_cost = jnp.square(
            self._get_base_height(state) - self.target_height
        )
        control_cost = jnp.sum(jnp.square(control))
        nominal_cost = jnp.sum(jnp.square(state.qpos[7:] - self.qstand[7:]))
        return (
            1.0 * orientation_cost
            + 10.0 * height_cost
            + 0.1 * nominal_cost
            + 0.01 * control_cost
        )

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        return self.running_cost(state, jnp.zeros(self.model.nu))

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomize the friction parameters."""
        n_geoms = self.model.geom_friction.shape[0]
        multiplier = jax.random.uniform(rng, (n_geoms,), minval=0.5, maxval=2.0)
        new_frictions = self.model.geom_friction.at[:, 0].set(
            self.model.geom_friction[:, 0] * multiplier
        )
        return {"geom_friction": new_frictions}

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        """Randomly perturb the measured base position and velocities."""
        rng, q_rng, v_rng = jax.random.split(rng, 3)
        q_err = 0.01 * jax.random.normal(q_rng, (7,))
        v_err = 0.01 * jax.random.normal(v_rng, (6,))

        qpos = data.qpos.at[0:7].set(data.qpos[0:7] + q_err)
        qvel = data.qvel.at[0:6].set(data.qvel[0:6] + v_err)

        return {"qpos": qpos, "qvel": qvel}

asynchronous = False

# Define the task (cost and dynamics)
task = Go2()

# Set up the controller
ctrl = MPPI(
    task,
    num_samples=128,
    noise_level=1.0,
    temperature=0.1,
    num_randomizations=4,
)

# Define the model used for simulation
mj_model = task.mj_model
mj_model.opt.timestep = 0.01

# Set the initial state so the robot falls and needs to stand back up
mj_data = mujoco.MjData(mj_model)
mj_data.qpos[:] = mj_model.keyframe("home").qpos
mj_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

# Run the interactive simulation
if asynchronous:
    print("Running asynchronous simulation")

    # Tighten up the simulator parameters, since it's running on CPU and
    # therefore won't slow down the planner
    mj_model.opt.timestep = 0.005
    mj_model.opt.iterations = 100
    mj_model.opt.ls_iterations = 50
    mj_model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC

    run_async(
        ctrl,
        mj_model,
        mj_data,
    )
else:
    print("Running deterministic simulation")
    run_interactive(
        ctrl,
        mj_model,
        mj_data,
        frequency=50,
        show_traces=False,
    )
