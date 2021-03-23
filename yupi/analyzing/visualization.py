import matplotlib.pyplot as plt
import numpy as np

def plot_trajectories(trajectories, max_trajectories=None, title="", legend=True, plot=True):
    # TODO: Check if trajectories is list of Trajectory
    # or Trajectory, if the second case traj = [traj]
    # If none of both case raise exception

    if max_trajectories is None:
        max_trajectories = len(trajectories)


    for i, t in enumerate(trajectories):
        if i == max_trajectories:
            break
        # plotting
        traj_plot = plt.plot(t.x, t.y, '-')
        color = traj_plot[-1].get_color()
        plt.plot(t.x[0], t.y[0], 'o', mfc='white', zorder=2,
                 label=f'{t.id} initial position', color=color)
        plt.plot(t.x[-1], t.y[-1], 'o', mfc='white', zorder=2,
                 color=color)
        plt.plot(t.x[-1], t.y[-1], 'o', alpha=.5,
                 label=f'{t.id} final position', color=color)

        if legend:
            plt.legend(fontsize=12)

        plt.title(title)
        plt.tick_params(direction='in', labelsize=12)
        plt.axis('equal')
        plt.grid(True)
        plt.xlabel('x [m]', fontsize=12)
        plt.ylabel('y [m]', fontsize=12)
        
    if plot:
        plt.show()


def plot_velocity_hist(v, bins=20, plot=True):
    plt.hist(v, bins, density=True, ec='k', color='#00cc55')
    plt.xlabel('speed [m/s]')
    plt.ylabel('pdf')
    if plot:
        plt.show()


def plot_angle_distribution(theta, bins=50, ax=None, plot=True):
    
    if ax is None:        
        # fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax = plt.gca(projection='polar')
    plt.hist(theta, bins, density=True, ec='k', color='.85')
    ax.set_theta_zero_location('N')
    ax.set_rlabel_position(135)
    ax.set_axisbelow(True)
    plt.xlabel('turning angles pdf')
    if plot:
        plt.show()


def plot_msd(msd, msd_std, dt, lag=30, plot=True):
    lag_t_msd = dt * np.arange(lag)
    plt.plot(lag_t_msd, msd, color='.2')
    plt.fill_between(lag_t_msd, msd + msd_std, 
            msd - msd_std, color='#afc0da')
    plt.xlabel('lag time [s]')
    plt.ylabel('$\mathrm{msd \; [m^2/s]}$')
    if plot:
        plt.show()


def plot_kurtosis(kurtosis, dt=None, t_array=None, plot=True):
    if dt:
        t_array = np.linspace(0, dt*len(kurtosis), len(kurtosis))
    if t_array is not None:
        plt.plot(t_array, kurtosis)
        plt.xlabel('time [s]')
    else:
        plt.plot(kurtosis)

    plt.ylabel('kurtosis')
    if plot:
        plt.show()
