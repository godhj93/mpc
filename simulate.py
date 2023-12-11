
import do_mpc
from models import SingleInvertedPendulum, DoubleIInvertedPendulum
from utils import pendulum_bars
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import torch
import tensorboard
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

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



# Let's make a training dataset for the neural network.
# The neural network will predict optimal control input u* given the current state x and the control input u0 from the wrong MPC.
# the target u* is the control input from the correct MPC(DIP).

# Reinforcement learning configuration
from minimalRL.sac import QNet, PolicyNet, ReplayBuffer, calc_target

#Hyperparameters
lr_pi           = 0.0005
lr_q            = 0.001
init_alpha      = 0.01
gamma           = 0.98
batch_size      = 32
buffer_limit    = 50000
tau             = 0.01 # for target network soft update
target_entropy  = -1.0 # for automated alpha update
lr_alpha        = 0.001  # for automated alpha update

memory = ReplayBuffer()
q1, q2, q1_target, q2_target = QNet(lr_q), QNet(lr_q), QNet(lr_q), QNet(lr_q)
pi = PolicyNet(lr_pi)
q1_target.load_state_dict(q1.state_dict())
q2_target.load_state_dict(q2.state_dict())

def get_reward(x):
    '''
    We will use an energy-based formulation for the objective. 
    If we think of energy in terms of potential and kinetic energy it is clear that we want to maximize the potential energy (up-up position) 
    and minimize the kinetic energy (stabilization).
    mterm = model.aux['E_kin'] - model.aux['E_pot'] # terminal cost
    lterm = model.aux['E_kin'] - model.aux['E_pot'] # stage cost
      E_kin_p1 = 1 / 2 * m1 * (
            (dpos + l1 * dtheta[0] * cos(theta[0]))**2 +
            (l1 * dtheta[0] * sin(theta[0]))**2) + 1 / 2 * J1 * dtheta[0]**2
    x[0] = position of the cart -> pos
    x[1] = theta of the first pendulum -> theta
    x[2] = dposition of the cart -> dpos
    x[3] = dtheta of the first pendulum -> dtheta
    ''' 
    pos, theta, dpos, dtheta = x[0], x[1], x[2], x[3]
    l1 = 0.5/2
    J1 = (0.6 * l1 ** 2) /3 
    m0 = 0.6
    m1 = 0.2
    E_kin_cart = 1 / 2 * m0 * dpos**2
    E_kin_p1 =  1/2 * m1 *(dpos + l1 * dtheta * np.cos(theta))**2 + (l1 * dtheta * np.sin(theta))**2 + 1 / 2 * J1 * dtheta**2
    E_pos =  9.81 * l1 * np.cos(theta)
    
    return E_pos - (E_kin_p1+E_kin_cart)

for epi in range(10000):
    n_steps = 1000
    
    # Quickly reset the history of the MPC data object.
    SIP.controller.reset_history()
    #simulator.x0['theta'] = 0.99*np.pi
    SIP.controller.x0['theta'] = np.random.uniform(-np.pi, np.pi)
    DIP.controller.x0['theta'] = SIP.controller.x0['theta'] 
    x0 = SIP.controller.x0
    x0_DIP = DIP.controller.x0

    for k in range(n_steps+1):

        u0 = SIP.controller.make_step(x0)
        
        
        # Data generation
        if k != 0:
            # State
            s = np.array([[x0[0]], [x0[1]], [x0[2]], [x0[3]], [u0[0]]])
            # Action
            optimal_u, log_prob = pi(torch.from_numpy(s).float().flatten())
            # next state
            y_next = DIP.simulator.make_step(optimal_u.detach().numpy().reshape(1,1))
            x0_DIP = DIP.estimator.make_step(y_next)
            u0prime = SIP.controller.make_step(np.array([x0_DIP[0], x0_DIP[1], x0_DIP[3], x0_DIP[4]]))
            optimal_u_prime = pi(torch.from_numpy(np.array([x0_DIP[0], x0_DIP[1], x0_DIP[3], x0_DIP[4], u0prime[0]])).float().flatten())[0].detach().numpy()
            # print([x0_DIP[0], x0_DIP[1], x0_DIP[3], x0_DIP[4], optimal_u_prime])
            s_prime = np.array([[x0_DIP[0]], [x0_DIP[1]], [x0_DIP[3]], [x0_DIP[4]], [optimal_u_prime]])
            print(s_prime)
            # raise ValueError()
            # Reward
            r = get_reward(s[:-1])
            
            memory.put((s.reshape(-1,5), optimal_u.detach().numpy().reshape(1,1), r, s_prime.reshape(-1,5), False))
            
            # Logging by tensorboard
            writer.add_scalar("reward", r, epi*n_steps+k)
            writer.add_scalar("theta(deg)", np.rad2deg(x0[1]), epi*n_steps+k)
            writer.add_scalar("input force", u0[0], epi*n_steps+k)
            writer.add_scalar("log_prob", log_prob, epi*n_steps+k)
            if memory.size() > 1000:
                for i in range(20):
                    mini_batch = memory.sample(batch_size)
                    # print(f"mini batch shape: {len(mini_batch)}")
                    # print(f"{memory.buffer}")
                    td_target = calc_target(pi, q1_target, q2_target, mini_batch)
                    q1.train_net(td_target, mini_batch)
                    q2.train_net(td_target, mini_batch)
                    entropy = pi.train_net(q1, q2, mini_batch)
                    q1.soft_update(q1_target)
                    q2.soft_update(q2_target)
                    
        else:
            y_next = DIP.simulator.make_step(u0)
            x0_DIP = DIP.estimator.make_step(y_next)
            

        
        # Rearange states to match SIP
        x0 = np.array([x0_DIP[0], x0_DIP[1], x0_DIP[3], x0_DIP[4]])
    # break
# Let's save the trained network
torch.save(pi.state_dict(), 'pi.pth')
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
# %%
