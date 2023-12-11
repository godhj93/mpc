
import do_mpc
from models import SingleInvertedPendulum, DoubleIInvertedPendulum
from utils import pendulum_bars
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

SIP = SingleInvertedPendulum()
DIP = DoubleIInvertedPendulum()

print(SIP.model.x)
print(DIP.model.x)
plt.ion()
rcParams['text.usetex'] = False
rcParams['axes.grid'] = True
rcParams['lines.linewidth'] = 2.0
rcParams['axes.labelsize'] = 'xx-large'
rcParams['xtick.labelsize'] = 'xx-large'
rcParams['ytick.labelsize'] = 'xx-large'

mpc_graphics = do_mpc.graphics.Graphics(SIP.controller.data)


# %%capture


fig = plt.figure(figsize=(16,9))

ax1 = plt.subplot2grid((4, 2), (0, 0), rowspan=4)
ax2 = plt.subplot2grid((4, 2), (0, 1))
ax3 = plt.subplot2grid((4, 2), (1, 1))
ax4 = plt.subplot2grid((4, 2), (2, 1))
ax5 = plt.subplot2grid((4, 2), (3, 1))

ax2.set_ylabel('$E_{kin}$ [J]')
ax3.set_ylabel('$E_{pot}$ [J]')
ax4.set_ylabel('Angle  [rad]')
ax5.set_ylabel('Input force [N]')

# Axis on the right.
for ax in [ax2, ax3, ax4, ax5]:
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    if ax != ax5:
        ax.xaxis.set_ticklabels([])

ax5.set_xlabel('time [s]')

mpc_graphics.add_line(var_type='_aux', var_name='E_kin', axis=ax2)
mpc_graphics.add_line(var_type='_aux', var_name='E_pot', axis=ax3)
mpc_graphics.add_line(var_type='_x', var_name='theta', axis=ax4)
mpc_graphics.add_line(var_type='_u', var_name='force', axis=ax5)

ax1.axhline(0,color='black')

bar1 = ax1.plot([],[], '-o', linewidth=5, markersize=10)
bar2 = ax1.plot([],[], '-o', linewidth=5, markersize=10)

ax1.set_xlim(-1.8,1.8)
ax1.set_ylim(-1.2,1.2)
ax1.set_axis_off()

fig.align_ylabels()
fig.tight_layout()

# %%capture
# Quickly reset the history of the MPC data object.
SIP.controller.reset_history()

x0 = SIP.controller.x0

n_steps = 300
for k in range(n_steps):

    u0 = SIP.controller.make_step(x0)
    
    # Add Neural Network to compensate input from wrong model
    y_next = DIP.simulator.make_step(u0)
    
    x0 = DIP.estimator.make_step(y_next)
    # Rearange states to match SIP
    x0 = np.array([x0[0], x0[1], x0[3], x0[4]])



print("DONE!!!")
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Your existing code for the animation setup
# x_arr = SIP.controller.data['_x']
x_arr = DIP.simulator.data['_x']
def update(t_ind):
    line1, line2 = pendulum_bars(x_arr[t_ind])
    bar1[0].set_data(line1[0],line1[1])
    bar2[0].set_data(line2[0],line2[1])
    mpc_graphics.plot_results(t_ind)
    mpc_graphics.plot_predictions(t_ind)
    mpc_graphics.reset_axes()

anim = FuncAnimation(fig, update, frames=n_steps, repeat=False)

# Use FFMpegWriter to save the animation
ffmpeg_writer = FFMpegWriter(fps=20)
anim.save('anim_dip2.mp4', writer=ffmpeg_writer)  # Saving as an MP4 file
