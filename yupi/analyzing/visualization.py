import matplotlib.pyplot as plt

def plot_trajectories(trajectories):
    # TODO: Check if trajectories is list of Trajectory
    # or Trajectory, if the second case traj = [traj]
    # If none of both case raise exception
    for t in trajectories:

        # plotting
        plt.plot(t.x_arr, t.y_arr, '-')
        plt.plot(t.x_arr[0], t.y_arr[0], 'go', mfc='white', zorder=2)
        plt.plot(t.x_arr[-1], t.y_arr[-1], 'ro', mfc='white', zorder=2)
        plt.plot(t.x_arr[0], t.y_arr[0], 'go', mec='g', alpha=.5, label='{} initial position'.format(t.id))
        plt.plot(t.x_arr[-1], t.y_arr[-1], 'ro', mec='r', alpha=.5, label='{} final position'.format(t.id))
        plt.legend(fontsize=12)

        plt.title('Trajectories in global coordinates')
        plt.tick_params(direction='in', labelsize=12)
        plt.axis('equal')
        plt.grid(True)
        plt.xlabel('x [m]', fontsize=12)
        plt.ylabel('y [m]', fontsize=12)

        plt.show()