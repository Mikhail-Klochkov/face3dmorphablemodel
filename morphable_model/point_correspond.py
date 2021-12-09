import logging

import numpy as np
from sklearn.metrics import pairwise_distances


class PointCorresponder():


    def point_correspond_3d_to_2d(self, projected_pts, visible_indces, keypoints, top_closest=1):
        """
        Args:
            projected_pts:
            visible_indces:
            keypoints:
            top_closest:

        Returns:
            Dictionary with idx keypoint : top closest indeces of mean_shape points
        """
        visible_idxs = list(visible_indces.keys())
        assert projected_pts.shape[1] == keypoints.shape[1], 'Points should be 2d.'
        # 68 x N_vis_points filtered only visible pts
        projected_pts_visible = [projected_pts[idx] for idx in visible_idxs]
        logging.info(f'We have {len(projected_pts_visible)} visible pts on image!')
        distances = pairwise_distances(keypoints, projected_pts_visible, metric='euclidean')
        # 68 x top
        top_idx_vis_pts = np.argsort(distances, axis=1)[:, :top_closest]
        # indeces of mean_shape points
        top_idx_mean_shape = np.empty((keypoints.shape[0], top_closest))
        for idx, top_idxs in enumerate(top_idx_vis_pts):
            top_idx_mean_shape[idx] = [visible_idxs[i] for i in top_idxs]
        correspondence_dict = {idx: top_idxs.astype(np.int).tolist() for idx, top_idxs in
                               zip(range(len(keypoints)), top_idx_mean_shape)}
        return correspondence_dict
