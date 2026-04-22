import os
import numpy as np
import math
import torch
from utils.normalization import normalize_min_max
from torch.nn import functional as F
from einops import rearrange


def seq_collate_egotraj(batch):
    """
    collate the batch with the variable number of agents
    1. add the past gt: past_traj_gt, past_traj_gt_orig
    2. add the past valid: past_traj_valid
    """
    (
        index,
        num_peds,
        past_traj,
        fut_traj,
        past_traj_valid,
        past_traj_orig,
        fut_traj_orig,
        traj_vel,
        past_traj_gt,
        past_traj_gt_orig,
        past_theta,
    ) = zip(*batch)

    batch_max_agents = max(num_peds)
    mask_list = []
    past_traj_list = []
    fut_traj_list = []
    past_traj_orig_list = []
    fut_traj_orig_list = []
    traj_vel_list = []
    past_traj_gt_list = []
    past_traj_gt_orig_list = []
    past_traj_valid_list = []
    past_theta_list = []

    for i in range(len(batch)):

        pad_len = batch_max_agents - num_peds[i]
        if pad_len > 0:
            past_traj_list.append(
                F.pad(past_traj[i], (0, 0, 0, 0, 0, 0, 0, pad_len), "constant", 0)
            )
            fut_traj_list.append(
                F.pad(fut_traj[i], (0, 0, 0, 0, 0, 0, 0, pad_len), "constant", 0)
            )
            past_traj_orig_list.append(
                F.pad(past_traj_orig[i], (0, 0, 0, 0, 0, 0, 0, pad_len), "constant", 0)
            )
            fut_traj_orig_list.append(
                F.pad(fut_traj_orig[i], (0, 0, 0, 0, 0, 0, 0, pad_len), "constant", 0)
            )
            traj_vel_list.append(
                F.pad(traj_vel[i], (0, 0, 0, 0, 0, 0, 0, pad_len), "constant", 0)
            )
            mask_list.append(torch.cat([torch.ones(num_peds[i]), torch.zeros(pad_len)]))

            past_traj_gt_list.append(
                F.pad(past_traj_gt[i], (0, 0, 0, 0, 0, 0, 0, pad_len), "constant", 0)
            )
            past_traj_gt_orig_list.append(
                F.pad(
                    past_traj_gt_orig[i], (0, 0, 0, 0, 0, 0, 0, pad_len), "constant", 0
                )
            )
            past_traj_valid_list.append(
                F.pad(past_traj_valid[i], (0, 0, 0, 0, 0, pad_len), "constant", 0)
            )
            past_theta_list.append(F.pad(past_theta[i], (0, pad_len), "constant", 0))
        else:
            past_traj_list.append(past_traj[i])
            fut_traj_list.append(fut_traj[i])
            past_traj_orig_list.append(past_traj_orig[i])
            fut_traj_orig_list.append(fut_traj_orig[i])
            traj_vel_list.append(traj_vel[i])
            mask_list.append(torch.ones(num_peds[i]))
            past_traj_gt_list.append(past_traj_gt[i])
            past_traj_gt_orig_list.append(past_traj_gt_orig[i])
            past_traj_valid_list.append(past_traj_valid[i])
            past_theta_list.append(past_theta[i])

    indexes = torch.stack(index, dim=0)
    pre_motion_3D = torch.stack(past_traj_list, dim=0).squeeze(
        dim=2
    )  # [B, A_max, P, 6]
    fut_motion_3D = torch.stack(fut_traj_list, dim=0).squeeze(dim=2)
    pre_motion_3D_orig = torch.stack(past_traj_orig_list, dim=0).squeeze(dim=2)
    fut_motion_3D_orig = torch.stack(fut_traj_orig_list, dim=0).squeeze(dim=2)
    fut_traj_vel = torch.stack(traj_vel_list, dim=0).squeeze(dim=2)
    mask = torch.stack(mask_list, dim=0)
    batch_size = torch.tensor(pre_motion_3D.shape[0])  ### bt
    pre_motion_3D_gt = torch.stack(past_traj_gt_list, dim=0).squeeze(dim=2)
    pre_motion_3D_gt_orig = torch.stack(past_traj_gt_orig_list, dim=0).squeeze(dim=2)
    past_traj_valid = torch.stack(past_traj_valid_list, dim=0).squeeze(
        dim=2
    )  # [B, A_max, P]
    past_theta = torch.stack(past_theta_list, dim=0)  # [B, A_max]

    data = {
        "indexes": indexes,
        "batch_size": batch_size,
        "past_traj": pre_motion_3D,
        "fut_traj": fut_motion_3D,
        "past_traj_original_scale": pre_motion_3D_orig,
        "fut_traj_original_scale": fut_motion_3D_orig,
        "fut_traj_vel": fut_traj_vel,
        "agent_mask": mask,
        "past_traj_gt": pre_motion_3D_gt,
        "past_traj_gt_original_scale": pre_motion_3D_gt_orig,
        "past_traj_valid": past_traj_valid,
        "past_theta": past_theta,
    }

    return data


def rotate_traj(
    past_rel,
    future_rel,
    past_abs,
    past_rel_gt,
    past_abs_gt,
    agents=2,
    rotate_time_frame=0,
    subset="eth",
):
    """
    add the rotation of the past gt
    """
    past_rel = rearrange(past_rel, "b a p d -> (b a) p d")  # [A, P, 2]
    past_abs = rearrange(past_abs, "b a p d -> (b a) p d")  # [A, P, 2]
    future_rel = rearrange(future_rel, "b a f d -> (b a) f d")  # [A, F, 2]
    past_rel_gt = rearrange(past_rel_gt, "b a p d -> (b a) p d")  # [A, P, 2]
    past_abs_gt = rearrange(past_abs_gt, "b a p d -> (b a) p d")  # [A, P, 2]

    def calculate_rotate_matrix(past_rel_reference, rotate_time_frame):
        # movement between frame 6 and 7
        past_diff = past_rel_reference[:, rotate_time_frame]  # [A, 2]
        # past_diff = past[:, rotate_time_frame] - past[:, rotate_time_frame-1]

        # atan(y / x) = angle between the vector and x-axis
        past_theta = torch.atan(torch.div(past_diff[:, 1], past_diff[:, 0] + 1e-5))
        # Corrects quadrant ambiguity
        past_theta = torch.where(
            (past_diff[:, 0] < 0), past_theta + math.pi, past_theta
        )
        # Rotation matrix
        rotate_matrix = torch.zeros((past_theta.size(0), 2, 2)).to(past_theta.device)
        rotate_matrix[:, 0, 0] = torch.cos(past_theta)
        rotate_matrix[:, 0, 1] = torch.sin(past_theta)
        rotate_matrix[:, 1, 0] = -torch.sin(past_theta)
        rotate_matrix[:, 1, 1] = torch.cos(past_theta)
        ### store the inverse of this rotate_matrix
        # inverse_rotate_matrix = rotate_matrix.transpose(1, 2)
        # np.save(f'inverse_rotate_matrix_{subset}.npy', inverse_rotate_matrix.detach().cpu().numpy())
        # exit()
        return rotate_matrix, past_theta

    rotate_matrix, past_theta = calculate_rotate_matrix(
        past_rel, rotate_time_frame
    )

    # Applies the same rotation to both past and future
    past_after = torch.matmul(rotate_matrix, past_rel.transpose(1, 2)).transpose(1, 2)
    future_after = torch.matmul(rotate_matrix, future_rel.transpose(1, 2)).transpose(
        1, 2
    )
    past_abs_after = torch.matmul(rotate_matrix, past_abs.transpose(1, 2)).transpose(
        1, 2
    )
    past_abs_gt_after = torch.matmul(
        rotate_matrix, past_abs_gt.transpose(1, 2)
    ).transpose(1, 2)
    past_rel_gt_after = torch.matmul(
        rotate_matrix, past_rel_gt.transpose(1, 2)
    ).transpose(1, 2)

    past_after = rearrange(past_after, "(b a) p d -> b a p d", a=agents)
    future_after = rearrange(future_after, "(b a) f d -> b a f d", a=agents)
    past_abs_after = rearrange(past_abs_after, "(b a) p d -> b a p d", a=agents)
    past_abs_gt_after = rearrange(past_abs_gt_after, "(b a) p d -> b a p d", a=agents)
    past_rel_gt_after = rearrange(past_rel_gt_after, "(b a) p d -> b a p d", a=agents)

    return (
        past_after,
        future_after,
        past_abs_after,
        past_abs_gt_after,
        past_rel_gt_after,
        past_theta,  # [A]
    )


class EgoTrajDataset(object):
    def __init__(
        self,
        cfg=None,
        split="train",
        data_dir=None,
        rotate_time_frame=0,
        type="original",
        source="tbd",
    ):
        # Resolve data file path based on type and source
        if type == "original" and source == "tbd":
            data_file_path = os.path.join(data_dir, f"egotraj_tbd_{split}.npz")
        elif type == "original_bal" and source != "tbd":
            data_file_path = os.path.join(data_dir, f"t2fpv_{source}_{split}.npz")
        else:
            raise ValueError(f"Invalid type/source combination: type={type}, source={source}")

        dset = np.load(data_file_path)
        self.all_obs = torch.from_numpy(dset["all_obs"])
        self.all_obs = self.all_obs[:, None, :, :]  # [N, 1, 8, 7]
        self.all_pred = torch.from_numpy(dset["all_pred"])
        self.all_pred = self.all_pred[:, None, :, :]  # [N, 1, 20, 7]
        self.num_peds_in_seq = torch.from_numpy(dset["num_peds"])  # [n_seq]
        self.seq_start_end = torch.from_numpy(dset["seq_start_end"])  # [n_seq, 2]

        self.cfg = cfg
        self.rotate_time_frame = rotate_time_frame
        self.split = split

        ### set the agent_num in the cfg
        # max_agents: 13 -> use the customed value 16
        max_agents = max(self.num_peds_in_seq)
        assert (
            cfg.MODEL.CONTEXT_ENCODER.AGENTS >= max_agents
        ), f"cfg.MODEL.CONTEXT_ENCODER.AGENTS: {cfg.MODEL.CONTEXT_ENCODER.AGENTS} < max_agents in {max_agents}"

        # set for the rotate_traj
        cfg.agents = self.all_obs.shape[1]

        assert self.all_obs.shape[2] == cfg.past_frames

        ### compute past and future trajectories
        past_traj_abs = self.all_obs[:, :, :, :2]  # [A, 1, P, 2]
        initial_pos = past_traj_abs[:, :, -1:]  # [A, 1, 1, 2]
        past_traj_rel = (past_traj_abs - initial_pos).contiguous()
        fut_traj = (
            self.all_pred[:, :, cfg.past_frames :, :2] - initial_pos
        ).contiguous()  # relative future traj

        ### add the past gt
        past_traj_abs_gt = self.all_pred[:, :, : cfg.past_frames, :2]  # [A, 1, P, 2]
        initial_pos_gt = past_traj_abs_gt[:, :, -1:]  # [A, 1, 1, 2]
        past_traj_rel_gt = (past_traj_abs_gt - initial_pos_gt).contiguous()
        # column names: [x, y, orient/yaw, img_x, img_y, valid, agent_id]
        self.past_traj_valid = self.all_obs[:, :, :, 5]

        if cfg.rotate:  # the normalization
            (
                past_traj_rel,
                fut_traj,
                past_traj_abs,
                past_traj_abs_gt,
                past_traj_rel_gt,
                past_theta,
            ) = rotate_traj(
                past_rel=past_traj_rel,
                future_rel=fut_traj,
                past_abs=past_traj_abs,
                past_rel_gt=past_traj_rel_gt,
                past_abs_gt=past_traj_abs_gt,
                agents=cfg.agents,
                rotate_time_frame=rotate_time_frame,
                subset=None,
            )

        ### save for the visualization
        if split == "test":
            self.past_theta = past_theta

        past_traj_vel = torch.cat(
            (
                past_traj_rel[:, :, 1:] - past_traj_rel[:, :, :-1],
                torch.zeros_like(past_traj_rel[:, :, -1:]),
            ),
            dim=2,
        )
        past_traj_vel_gt = torch.cat(
            (
                past_traj_rel_gt[:, :, 1:] - past_traj_rel_gt[:, :, :-1],
                torch.zeros_like(past_traj_rel_gt[:, :, -1:]),
            ),
            dim=2,
        )
        past_traj = torch.cat((past_traj_abs, past_traj_rel, past_traj_vel), dim=-1)
        past_traj_gt = torch.cat(
            (past_traj_abs_gt, past_traj_rel_gt, past_traj_vel_gt), dim=-1
        )
        self.fut_traj_vel = torch.cat(
            (
                fut_traj[:, :, 1:] - fut_traj[:, :, :-1],
                torch.zeros_like(fut_traj[:, :, -1:]),
            ),
            dim=2,
        )

        # Compute and persist normalization stats (stored in yml_dict for YAML serialization)
        if split == "train":
            cfg.yml_dict["fut_traj_max"] = fut_traj.max().item()
            cfg.yml_dict["fut_traj_min"] = fut_traj.min().item()
            cfg.yml_dict["past_traj_max"] = past_traj.max().item()
            cfg.yml_dict["past_traj_min"] = past_traj.min().item()
            cfg.yml_dict["past_traj_gt_max"] = past_traj_gt.max().item()
            cfg.yml_dict["past_traj_gt_min"] = past_traj_gt.min().item()
        elif cfg.get("past_traj_min", None) is None:
            # Fallback: compute from current data when train stats are unavailable
            cfg.yml_dict["fut_traj_max"] = fut_traj.max().item()
            cfg.yml_dict["fut_traj_min"] = fut_traj.min().item()
            cfg.yml_dict["past_traj_max"] = past_traj.max().item()
            cfg.yml_dict["past_traj_min"] = past_traj.min().item()
            cfg.yml_dict["past_traj_gt_max"] = past_traj_gt.max().item()
            cfg.yml_dict["past_traj_gt_min"] = past_traj_gt.min().item()

        ### record the original to avoid numerical errors
        self.past_traj_original_scale = past_traj
        self.fut_traj_original_scale = fut_traj
        self.past_traj_gt_original_scale = past_traj_gt

        ### min-max linear normalization
        if cfg.data_norm == "min_max":
            self.past_traj = normalize_min_max(
                past_traj, cfg.past_traj_min, cfg.past_traj_max, -1, 1
            ).contiguous()  # [A, 1, P, 6]
            self.fut_traj = normalize_min_max(
                fut_traj, cfg.fut_traj_min, cfg.fut_traj_max, -1, 1
            ).contiguous()  # [A, 1, F, 2]
            self.past_traj_gt = normalize_min_max(
                past_traj_gt, cfg.past_traj_gt_min, cfg.past_traj_gt_max, -1, 1
            ).contiguous()  # [A, 1, P, 6]
        elif cfg.data_norm == "original":
            self.past_traj = past_traj
            self.fut_traj = fut_traj
            self.past_traj_gt = past_traj_gt

    def __len__(self):
        return len(self.num_peds_in_seq)

    def __getitem__(self, item):
        seq_start, seq_end = self.seq_start_end[item]
        num_peds = self.num_peds_in_seq[item]

        ### past traj, future traj, number of pedestrians (presumbly?), index
        past_traj_norm_scale = self.past_traj[
            seq_start:seq_end
        ]  # [A, P, 6] abs + rel + vel; rotate; norm
        fut_traj_norm_scale = self.fut_traj[
            seq_start:seq_end
        ]  # [A, F, 2] just rel; rotate; norm
        past_traj_original_scale = self.past_traj_original_scale[
            seq_start:seq_end
        ]  # [A, P, 6] abs + rel + vel; rotate
        fut_traj_original_scale = self.fut_traj_original_scale[
            seq_start:seq_end
        ]  # [A, F, 2] just rel; rotate
        fut_traj_vel = self.fut_traj_vel[
            seq_start:seq_end
        ]  # [A, F, 2] just vel; rotate

        past_traj_gt_norm_scale = self.past_traj_gt[
            seq_start:seq_end
        ]  # [A, P, 6] abs + rel + vel; rotate; norm
        past_traj_gt_original_scale = self.past_traj_gt_original_scale[
            seq_start:seq_end
        ]  # [A, P, 6] abs + rel + vel; rotate
        past_traj_valid = self.past_traj_valid[seq_start:seq_end]

        if self.split == "test":
            past_theta = self.past_theta[seq_start:seq_end]
        else:
            past_theta = torch.zeros(
                past_traj_norm_scale.size(0), device=past_traj_norm_scale.device
            )  # dummy variable

        out = [
            torch.Tensor([item]).to(torch.int32),
            torch.Tensor([num_peds]).to(torch.int32),
            past_traj_norm_scale,  # [A, P, 6] -> eth A = 1
            fut_traj_norm_scale,
            past_traj_valid,
            past_traj_original_scale,
            fut_traj_original_scale,
            fut_traj_vel,
            past_traj_gt_norm_scale,
            past_traj_gt_original_scale,
            past_theta,
        ]
        return out
