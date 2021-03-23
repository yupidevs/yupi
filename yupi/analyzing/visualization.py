import matplotlib.pyplot as plt

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