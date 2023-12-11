
import do_mpc
from models import SingleInvertedPendulum, DoubleIInvertedPendulum
from utils import pendulum_bars
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import torch

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
x0_DIP = DIP.controller.x0
#simulator.x0['theta'] = 0.99*np.pi

# Let's make a training dataset for the neural network.
# The neural network will predict optimal control input u* given the current state x and the control input u0 from the wrong MPC.
# the target u* is the control input from the correct MPC(DIP).
train_data_x = []
train_data_y = []

n_steps = 1000
for k in range(n_steps+1):

    u0 = SIP.controller.make_step(x0)
    u_target = DIP.controller.make_step(x0_DIP)
    # Data generation
    if k != 0:
        input_data = np.array([x0[0], x0[1], x0[2], x0[3], u0[0]])

        
        train_input = torch.tensor(input_data)
        train_output = torch.tensor(u_target)
        
        train_data_x.append(train_input)
        train_data_y.append(train_output)
        
    y_next = DIP.simulator.make_step(u0)
    x0_DIP = DIP.estimator.make_step(y_next)
    # Rearange states to match SIP
    x0 = np.array([x0_DIP[0], x0_DIP[1], x0_DIP[3], x0_DIP[4]])
    # break
# Let's save the training data
train_data_x = torch.stack(train_data_x)
train_data_y = torch.stack(train_data_y)
torch.save(train_data_x, 'train_data_x.pt')
torch.save(train_data_y, 'train_data_y.pt')
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