from vo_pipeline.continuousVO import ContinuousVO
from utils.loadData import Dataset
import matplotlib.pyplot as plt
from utils.matrix import *
import matplotlib.animation as animation
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.cm as cm

trajectory = np.array([0, 0])
tracked_kps = np.array([0, 0])


def plt_online(continuousVO: ContinuousVO, dataset: Dataset):
    fig = plt.figure()
    # Img subplot. We plot the landmarks and keypoints of the current frame
    ax_img = fig.add_subplot(221)
    sc_landmarks = ax_img.scatter([], [], s=10, color="green", marker="x", label="landmarks")
    sc_keypoints = ax_img.scatter([], [], s=10, color="red", marker="x", label="keypoints")

    # Full trajectory subplot
    ax_full_traj = fig.add_subplot(246)
    sc_full_traj, = ax_full_traj.plot([], [], color="blue", label="trajectory", lw=2)

    # Number if matched landmarks
    ax_tracked_kps = fig.add_subplot(245)
    sc_tracked_kps, = ax_tracked_kps.plot([], [], color='red', label="matched lks", lw=2)

    # Trajectory of last 20 frames and landmarks subplot
    ax_local_traj = fig.add_subplot(122)
    sc_local_traj = ax_local_traj.scatter([], [], s=5, color="blue", marker="o", label="trajectory")
    sc_local_lks = ax_local_traj.scatter([], [], s=3, color="black", marker="*", label="landmarks")

    def animate(i):
        global trajectory
        global tracked_kps
        continuousVO.step()
        if len(continuousVO.keypoint_trajectories.landmarks) > 0:

            # Get current pose
            p = hom_inv(continuousVO.frame_queue.get_head().pose)[0:3, 3]
            trajectory = np.vstack((trajectory, p[[0, 2]]))

            # Tracked kps subplot
            num_tracked_kps = continuousVO.frame_queue.get_head().num_tracked_kps
            tracked_kps = np.vstack((tracked_kps, np.array([i+1, num_tracked_kps])))
            sc_tracked_kps.set_data(tracked_kps[-100:, 0], tracked_kps[-100:, 1])

            # Plot full trajectory
            sc_full_traj.set_data(trajectory[:, 0], trajectory[:, 1])

            # Plot local trajectory and landmarks
            sc_local_traj.set_offsets(trajectory[-20:, :])

            # Get active landmarks
            landmarks = continuousVO.keypoint_trajectories.get_all()
            landmarks = np.array(landmarks)
            if landmarks.size > 0:
                sc_local_lks.set_offsets(landmarks[:, [0, 2]])

            # Image, landmarks and keypoints subplot
            ax_img.imshow(continuousVO.frame_queue.get_head().img, cmap='gray')
            keypoints, _, _ = continuousVO.keypoint_trajectories.at_frame(continuousVO.keypoint_trajectories.latest_frame)
            if keypoints.size > 0:
                sc_keypoints.set_offsets(keypoints)

            M = continuousVO.K @ continuousVO.frame_queue.get_head().pose[0:3, 0:4]
            active = continuousVO.keypoint_trajectories.get_active()
            active = np.array(active)
            if active.size > 0:
                active_hom = np.hstack((active, np.ones((active.shape[0], 1))))
                img_pts = (M @ active_hom.T).T
                img_pts = (img_pts.T / img_pts[:, 2]).T
                sc_landmarks.set_offsets(img_pts[:, 0:2])

            xy_min = np.min(trajectory, axis=0)
            xy_max = np.max(trajectory, axis=0)
            ax_full_traj.set_xlim(xy_min[0]-2, xy_max[0]+2)
            ax_full_traj.set_ylim(xy_min[1]-2, xy_max[1]+2)

            all_points = np.vstack((trajectory[-20:, :], active[:, [0, 2]]))
            x_min = np.min(all_points[:, 0])
            x_max = np.max(all_points[:, 0])
            y_min = np.min(all_points[:, 1])
            y_max = np.max(all_points[:, 1])

            ax_local_traj.set_xlim(x_min-2, x_max+2)
            ax_local_traj.set_ylim(y_min-2, y_max+2)
            # ax_local_traj.axis('equal')

            if tracked_kps.shape[0] < 100:
                ax_tracked_kps.set_xlim(0, tracked_kps[-1, 0])
            else:
                ax_tracked_kps.set_xlim(tracked_kps[-100, 0], tracked_kps[-1, 0])

            ax_tracked_kps.set_ylim(0, np.max(tracked_kps[-100:, 1])+20)
            plt.show()

    ani = animation.FuncAnimation(fig, animate)

    ax_img.legend()
    ax_img.set_title("Current frame")
    ax_local_traj.set_title("Trajectory of last 20 frames and landmarks")
    ax_full_traj.set_title("Full trajectory")
    ax_tracked_kps.set_title("Tracked keypoints")
    plt.tight_layout()
    plt.show()


def plt_trajectory_landmarks(continuousVO: ContinuousVO, dataset: Dataset):
    fig = plt.figure()
    ax_3d = fig.add_subplot(121, projection="3d")
    sc_active = ax_3d.scatter([], [], [], label="active")
    sc_ego = ax_3d.scatter([], [], [], color="green", label="$T_i$")
    sc_ego_key = ax_3d.scatter([], [], [],  color="red", label="$T^{key}_i$", marker="*")
    sc_gt = ax_3d.scatter([], [], [],  color="fuchsia", label="$T^{gt}$",marker="o")

    ax_img = fig.add_subplot(122)
    sc_landmarks = ax_img.scatter([], [], s=1, color="red", marker="*", label="landmarks")
    sc_keypoints = ax_img.scatter([], [], s=0.5, color="yellow", marker="*", label="keypoints")
    title = ax_3d.set_title("Reconstructed points, t=0")

    def animate(i):
        continuousVO.step()
        if len(continuousVO.keypoint_trajectories.landmarks) > 0:
            # plot 3D
            active = continuousVO.keypoint_trajectories.get_active()
            active = np.array(active)
            if active.size > 0:
                sc_active._offsets3d = (active[:, 0],active[:, 1],active[:, 2])

            p = np.array([hom_inv(k.pose)[0:3, 3] for k in continuousVO.frame_queue])
            # sc_ego_key._offsets3d = (p[:,0], p[:, 1], p[:, 2])
            sc_ego._offsets3d = (p[:,0], p[:, 1], p[:, 2])

            # gt_scale = np.linalg.norm(keyframes[0]) / np.linalg.norm(dataset.T[continuousVO.frames_to_skip - 1, 0:3, 3])
            gt = dataset.T[:i, 0:3, 3]
            sc_gt._offsets3d = (gt[:, 0], gt[:, 1], gt[:, 2])

            # plot images
            ax_img.imshow(continuousVO.frame_queue.get_head().img)
            # keypoints, _, _ = continuousVO.keypoint_trajectories.at_frame(continuousVO.keypoint_trajectories.latest_frame)
            # if keypoints.size > 0:
            #     sc_keypoints.set_offsets(keypoints)

            M = continuousVO.K @ continuousVO.frame_queue.get_head().pose[0:3, 0:4]
            if active.size > 0:
                active_hom = np.hstack((active, np.ones((active.shape[0], 1))))
                img_pts = (M @ active_hom.T).T
                img_pts = (img_pts.T / img_pts[:, 2]).T
                sc_landmarks.set_offsets(img_pts[:, 0:2])

            title.set_text(f"Reconstructed points, t={i}")

    ani = animation.FuncAnimation(fig, animate)
    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_zlabel("z")
    ax_3d.set_xlim(-5, 5)
    ax_3d.set_ylim(-5, 5)
    ax_3d.set_zlim(-5, 20)
    ax_3d.legend()

    ax_img.legend()
    plt.tight_layout()
    plt.show()




def plt_trajectory(continuousVO: ContinuousVO, dataset: Dataset, plot_frame_indices=False):
    fig = plt.figure()
    ax_traj_pred = fig.add_subplot(221)
    ax_traj_true = fig.add_subplot(222)
    ax_err_scale = fig.add_subplot(223)
    ax_err_trans = fig.add_subplot(224)

    IDX = 200
    OFFSET = 4

    frame_states = []
    for i in range(len(dataset.T)):
        frame_state = continuousVO.step()
        frame_states.append(frame_state)

    # get estimated and true poses
    p = np.array([hom_inv(k.pose)[0:3, 3] for k in frame_states])
    frame_indices = np.array([state.idx for state in frame_states])
    p = p[OFFSET:IDX]
    frame_indices = frame_indices[OFFSET:IDX]
    gt = dataset.T[OFFSET:IDX, 0:3, 3]

    gt_max, gt_min = np.max(gt, axis=0), np.min(gt, axis=0)
    p_max, p_min = np.max(p, axis=0), np.min(p, axis=0)
    x_max = np.max([p_max[0], gt_max[0]])
    z_max = np.max([p_max[2], gt_max[2]])
    x_min = np.min([p_min[0], gt_min[0]])
    z_min = np.min([p_min[2], gt_min[2]])

    # plot estimated trajectory and the points where we bootstrapped again

    ax_traj_pred.scatter(p[:, 0], p[:, 2], label="$T_p$", c=np.linspace(0, 1, p.shape[0]), cmap=cm.get_cmap("viridis"))
    if plot_frame_indices:
        for i, pt in enumerate(p):
            ax_traj_pred.text(pt[0], pt[2], f"{frame_indices[i]}")

    bootstrap_x_idx = [x-OFFSET for x in continuousVO.bootstrap_idx if OFFSET <= x < IDX]
    # [ax_traj_pred.axvline(p[x_idx, 0]) for x_idx in bootstrap_x_idx]
    ax_traj_pred.set_xlabel("x [m]")
    ax_traj_pred.set_ylabel("z [m]")
    #ax_traj_pred.set_xlim(x_min, x_max)
    #ax_traj_pred.set_ylim(z_min, z_max)
    ax_traj_pred.set_title('Predicted Trajectory')
    ax_traj_pred.legend()

    # plot true trajectory
    ax_traj_true.scatter(gt[:, 0], gt[:, 2], label="$T_t$", c=np.linspace(0, 1, gt.shape[0]), cmap=cm.get_cmap("viridis"))
    ax_traj_true.set_xlabel("x [m]")
    ax_traj_true.set_ylabel("z [m]")
    ax_traj_true.set_xlim(x_min, x_max)
    ax_traj_true.set_ylim(z_min, z_max)
    ax_traj_true.set_title('Ground Truth Trajectory')
    ax_traj_true.legend()

    # translational error
    z_true_inter = np.interp(p[:, 0], xp=gt[:, 0], fp=gt[:, 2])
    ax_err_trans.plot(p[:, 0], (z_true_inter-p[:, 2])/z_true_inter)
    ax_err_trans.set_xlabel("x [m]")
    ax_err_trans.set_ylabel("Translational error [%]")
    ax_err_trans.set_title("Translational error over distance")
    ax_err_trans.legend()

    # scale drift determined by yaw angle error
    def _get_yaw_angle(rot_mat):
        rot = R.from_matrix(rot_mat)
        zyx = rot.as_euler('zyx', degrees=True)
        return zyx[0]

    yaw_pred = np.array([_get_yaw_angle(hom_inv(k.pose)[0:3, 0:3]) for k in frame_states])
    yaw_true = np.array([_get_yaw_angle(T_i[0:3, 0:3]) for T_i in dataset.T])
    yaw_true = yaw_true[OFFSET:IDX]
    yaw_pred = yaw_pred[OFFSET:IDX]

    yaw_pred_inter = np.interp(p[:, 0], xp=gt[:, 0], fp=yaw_pred)
    ax_err_scale.plot(p[:, 0], yaw_pred_inter-yaw_true)
    ax_err_scale.set_xlabel("x [m]")
    ax_err_scale.set_ylabel("Yaw error [deg]")
    ax_err_scale.set_title("Yaw error over distance")
    ax_err_scale.legend()

    plt.yticks(rotation=90)
    plt.tight_layout()
    plt.show()
