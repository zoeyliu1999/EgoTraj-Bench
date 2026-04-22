import os
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from pathlib import Path

import torch
import torch.nn as nn

from einops import rearrange, reduce, repeat

from accelerate import Accelerator
from ema_pytorch import EMA

from tqdm.auto import tqdm

from utils.common import exists, default
from utils.utils import set_random_seed
from utils.normalization import unnormalize_min_max


def cycle(dl):
    while True:
        for data in dl:
            yield data


def build_scheduler(optimizer, opt_cfg, total_iters_each_epoch):
    total_epochs = opt_cfg.NUM_EPOCHS
    decay_steps = [
        x * total_iters_each_epoch
        for x in opt_cfg.get("DECAY_STEP_LIST", [5, 10, 15, 20])
    ]

    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in decay_steps:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * opt_cfg.LR_DECAY
        return max(cur_decay, opt_cfg.LR_CLIP / opt_cfg.LR)

    if opt_cfg.get("SCHEDULER", None) == "cosineAnnealingLRwithWarmup":
        total_iterations = total_epochs * total_iters_each_epoch
        warmup_iterations = max(1, int(total_iterations * 0.05))
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: max(opt_cfg.LR_CLIP / opt_cfg.LR, step / warmup_iterations),
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_iterations - warmup_iterations,
            eta_min=opt_cfg.LR_CLIP,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_iterations],
        )
    elif opt_cfg.get("SCHEDULER", None) == "lambdaLR":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lbmd)
    elif opt_cfg.get("SCHEDULER", None) == "linearLR":
        total_iters = total_iters_each_epoch * total_epochs
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=opt_cfg.LR_CLIP / opt_cfg.LR,
            total_iters=total_iters,
        )
    elif opt_cfg.get("SCHEDULER", None) == "stepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=opt_cfg.DECAY_STEP, gamma=opt_cfg.DECAY_GAMMA
        )
    elif opt_cfg.get("SCHEDULER", None) == "cosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=opt_cfg.LR_CLIP
        )
    else:
        scheduler = None
    return scheduler


def build_optimizer(model, opt_cfg):
    if opt_cfg.OPTIMIZER == "Adam":
        optimizer = torch.optim.Adam(
            [each[1] for each in model.named_parameters()],
            lr=opt_cfg.LR,
            weight_decay=opt_cfg.get("WEIGHT_DECAY", 0),
        )
    elif opt_cfg.OPTIMIZER == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt_cfg.LR,
            weight_decay=opt_cfg.get("WEIGHT_DECAY", 0),
        )
    else:
        assert False
    return optimizer


class BiFlowTrainer(object):
    def __init__(
        self,
        cfg,
        denoiser,
        train_loader,
        test_loader,
        val_loader=None,
        tb_log=None,
        logger=None,
        gradient_accumulate_every=1,
        ema_decay=0.995,
        ema_update_every=1,
        save_samples=False,
        *awgs,
        **kwargs,
    ):
        super().__init__()

        # init
        self.cfg = cfg
        self.denoiser = denoiser
        self.train_loader = train_loader
        self.test_loader = test_loader
        # use the test as val if val is not provided
        self.val_loader = default(val_loader, test_loader)
        self.tb_log = tb_log
        self.logger = logger

        self.gradient_accumulate_every = gradient_accumulate_every
        self.ema_decay = ema_decay
        self.ema_update_every = ema_update_every

        # config fields
        if cfg.denoising_method == "fm":
            self.denoising_steps = cfg.sampling_steps
            self.denoising_schedule = cfg.t_schedule
        else:
            raise NotImplementedError(
                f"Denoising method [{cfg.denoising_method}] is not implemented yet."
            )

        self.save_dir = Path(cfg.cfg_dir)

        # sampling and training hyperparameters
        self.gradient_accumulate_every = gradient_accumulate_every
        self.save_samples = save_samples

        # accelerator
        self.accelerator = Accelerator(split_batches=True, mixed_precision="no")

        # EMA model
        if self.accelerator.is_main_process:
            self.ema = EMA(denoiser, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        if train_loader is not None:
            self.save_and_sample_every = cfg.checkpt_freq * len(train_loader)
            self.train_num_steps = cfg.OPTIMIZATION.NUM_EPOCHS * len(train_loader)

            # optimizer
            self.opt = build_optimizer(self.denoiser, self.cfg.OPTIMIZATION)
            self.scheduler = build_scheduler(
                self.opt, self.cfg.OPTIMIZATION, len(self.train_loader)
            )

            # prepare model, dataloader, optimizer with accelerator
            self.denoiser, self.opt = self.accelerator.prepare(self.denoiser, self.opt)

            train_dl_ = self.accelerator.prepare(train_loader)
            self.train_loader = train_dl_
            self.dl = cycle(train_dl_)
        else:
            self.save_and_sample_every = 0
            self.train_num_steps = 1  # avoid division by zero in eval
            self.denoiser = self.accelerator.prepare(self.denoiser)

        self.test_loader = self.accelerator.prepare(test_loader)

        val_loader = default(val_loader, test_loader)
        self.val_loader = self.accelerator.prepare(val_loader)

        # set counters and training states
        self.step = 0
        self.best_ade_min = float("inf")

        # print the number of model parameters
        self.print_model_params(self.denoiser, "Stage One Model")

    def print_model_params(self, model: nn.Module, name: str):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(
            f"[{name}] Trainable/Total Params: {trainable_num}/{total_num}"
        )

    @property
    def device(self):
        return self.cfg.device

    def save_ckpt(self, ckpt_name):
        if not self.accelerator.is_local_main_process:
            return
        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.denoiser),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": (
                self.accelerator.scaler.state_dict()
                if exists(self.accelerator.scaler)
                else None
            ),
        }
        torch.save(data, os.path.join(self.cfg.model_dir, f"{ckpt_name}.pt"))

    def save_last_ckpt(self):
        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.denoiser),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        torch.save(data, os.path.join(self.cfg.model_dir, "checkpoint_last.pt"))

    def load(self, ckpt_name):
        accelerator = self.accelerator

        new_epochs = self.cfg.OPTIMIZATION.NUM_EPOCHS
        old_epochs = self.cfg.RESUME.start_epoch
        model_dir = self.cfg.model_dir.replace(f"EP{new_epochs}", f"EP{old_epochs}")

        data = torch.load(
            os.path.join(model_dir, f"{ckpt_name}.pt"),
            map_location=self.device,
            weights_only=True,
        )

        model = self.accelerator.unwrap_model(self.denoiser)
        model.load_state_dict(data["model"])

        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])
        if new_epochs == old_epochs:
            self.scheduler.load_state_dict(data["scheduler"])

        if self.accelerator.is_main_process:
            # pass
            self.ema.load_state_dict(data["ema"])

        if "version" in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data["scaler"]):
            self.accelerator.scaler.load_state_dict(data["scaler"])

    def train(self):
        """
        Training loop
        """

        # init
        accelerator = self.accelerator
        # load ckpt if needed
        if self.cfg.RESUME.resume:
            ckpt_name = self.cfg.RESUME.get("ckpt_name", "checkpoint_last")
            self.load(ckpt_name)
            self.logger.info(f"Resuming training from {ckpt_name}.pt")

        self.logger.info("training start")
        iter_per_epoch = self.train_num_steps // self.cfg.OPTIMIZATION.NUM_EPOCHS

        if self.cfg.RESUME.get("early_stop", -1) > 0:
            self.early_stop_num_steps = (
                self.cfg.RESUME.get("early_stop", -1) * iter_per_epoch
            )
        else:
            self.early_stop_num_steps = self.train_num_steps

        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not accelerator.is_main_process,
        ) as pbar:
            while (
                self.step < self.train_num_steps
                and self.step < self.early_stop_num_steps
            ):
                # init per-iteration variables
                total_loss = 0.0
                self.denoiser.train()
                self.ema.ema_model.train()

                for _ in range(self.gradient_accumulate_every):
                    data = {k: v.to(self.device) for k, v in next(self.dl).items()}

                    log_dict = {"cur_epoch": self.step // iter_per_epoch}

                    # compute the loss
                    with self.accelerator.autocast():
                        (
                            loss,
                            loss_reg_fut,
                            loss_cls_fut,
                            _,
                            loss_reg_pst,
                            loss_cls_pst,
                            _,
                        ) = self.denoiser(data, log_dict)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                    # log to tensorboard
                    if self.tb_log is not None:
                        self.tb_log.add_scalar(
                            "train/loss_total", loss.item(), self.step
                        )
                        self.tb_log.add_scalar(
                            "train/loss_reg_fut", loss_reg_fut.item(), self.step
                        )
                        self.tb_log.add_scalar(
                            "train/loss_cls_fut", loss_cls_fut.item(), self.step
                        )
                        self.tb_log.add_scalar(
                            "train/loss_reg_pst", loss_reg_pst.item(), self.step
                        )
                        self.tb_log.add_scalar(
                            "train/loss_cls_pst", loss_cls_pst.item(), self.step
                        )
                        self.tb_log.add_scalar(
                            "train/learning_rate",
                            self.opt.param_groups[0]["lr"],
                            self.step,
                        )

                pbar.set_description(
                    f'curr step: {self.step}/{self.train_num_steps}, total loss: {total_loss:.4f}, loss_reg_fut: {loss_reg_fut:.4f}, loss_cls_fut: {loss_cls_fut:.4f}, loss_reg_pst: {loss_reg_pst:.4f}, loss_cls_pst: {loss_cls_pst:.4f}, lr: {self.opt.param_groups[0]["lr"]:.6f}'
                )

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(
                    self.denoiser.parameters(), self.cfg.OPTIMIZATION.GRAD_NORM_CLIP
                )

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                if accelerator.is_main_process:
                    self.ema.update()
                    # checkpt test and save the best validation model
                    if (self.step + 1) >= self.save_and_sample_every and (
                        self.step + 1
                    ) % self.save_and_sample_every == 0:

                        fut_traj_gt, performance, n_samples = self.eval_dataloader(
                            testing_mode=False, training_err_check=False
                        )

                        cur_epoch = self.step // iter_per_epoch
                        select_idx = len(self.cfg.K_LIST) - 1
                        cur_ade_min = performance["ADE_min"][select_idx] / n_samples

                        # update the best model
                        if cur_ade_min < self.best_ade_min:
                            self.best_ade_min = cur_ade_min
                            self.logger.info(
                                f"Current best ADE_MIN: {self.best_ade_min}"
                            )
                            self.save_ckpt(f"checkpoint_best")

                        # save the model and remove the old models

                        ckpt_list = glob(
                            os.path.join(self.cfg.model_dir, "checkpoint_epoch_*.pt*")
                        )
                        ckpt_list.sort(key=os.path.getmtime)

                        if ckpt_list.__len__() >= self.cfg.max_num_ckpts:
                            for cur_file_idx in range(
                                0, len(ckpt_list) - self.cfg.max_num_ckpts + 1
                            ):
                                os.remove(ckpt_list[cur_file_idx])

                        self.save_ckpt("checkpoint_epoch_%d" % cur_epoch)

                self.step += 1
                pbar.update(1)
                self.scheduler.step()

                # end of one training iteration
            # end of training loop

        self.save_last_ckpt()

        self.logger.info("training complete")

    def compute_ADE_FDE(self, distances, end_frame):
        """
        Helper function to compute ADE and FDE
        distances: [b*num_agents, k_preds, future_frames] or [b*num_agents, timestamps, k_preds, future_frames]
        ade_frames: int
        fde_frame: int
        """
        ade_best = (
            (distances[..., :end_frame]).mean(dim=-1).min(dim=-1).values.sum(dim=0)
        )
        fde_best = (distances[..., end_frame - 1]).min(dim=-1).values.sum(dim=0)
        ade_avg = (distances[..., :end_frame]).mean(dim=-1).mean(dim=-1).sum(dim=0)
        fde_avg = (distances[..., end_frame - 1]).mean(dim=-1).sum(dim=0)
        return ade_best, fde_best, ade_avg, fde_avg

    ### Based on https://arxiv.org/abs/2305.06292 Joint metric for ADE and FDE
    def compute_JADE_JFDE(self, distances, end_frame):
        """
        Helper function to compute JADE and JFDE
        distances: [b*num_agents, k_preds, future_frames] or [b*num_agents, timestamps, k_preds, future_frames]
        ade_frames: int
        fde_frame: int
        """
        jade_best = (
            (distances[..., :end_frame]).mean(dim=-1).sum(dim=0).min(dim=-1).values
        )
        jfde_best = (distances[..., end_frame - 1]).sum(dim=0).min(dim=-1).values
        jade_avg = (distances[..., :end_frame]).mean(dim=-1).sum(dim=0).mean(dim=0)
        jfde_avg = (distances[..., end_frame - 1]).sum(dim=0).mean(dim=-1)
        return jade_best, jfde_best, jade_avg, jfde_avg

    def compute_avar_fvar(self, pred_trajs, end_frame):
        """
        Helper function to compute AVar and FVar
        predictions: [b*num_agents,k_preds, future_frames, dim]
        ade_frames: int
        fde_frame: int
        """
        a_var = pred_trajs[..., :end_frame, :].var(dim=(1, 3)).mean(dim=1).sum()
        f_var = pred_trajs[..., end_frame - 1, :].var(dim=(1, 2)).sum()
        return a_var, f_var

    def compute_MASD(self, pred_trajs, end_frame):
        """
        Helper function to compute MASD
        predictions: [b*num_agents,k_preds, future_frames, dim]
        ade_frames: int
        fde_frame: int
        """
        # Reshape for pairwise computation: (B, T, N, D)
        predictions = pred_trajs[:, :, :end_frame, :].permute(
            0, 2, 1, 3
        )  # Shape: (B, T, N, D)

        # Compute pairwise L2 distances among N samples at each (B, T)
        pairwise_distances = torch.cdist(
            predictions, predictions, p=2
        )  # Shape: (B, T, N, N)

        # Get the maximum squared distance among all pairs (excluding diagonal)
        max_squared_distance = pairwise_distances.max(dim=-1)[0].max(dim=-1)[
            0
        ]  # Shape: (B, T)

        # Compute the final MASD metric
        masd = max_squared_distance.mean(dim=-1).sum()
        return masd

    @torch.no_grad()
    def test(self, mode, eval_on_train=False, save_for_vis=False):
        # init
        self.logger.info(f"testing start with the {mode} ckpt")
        self.mode = mode
        self.save_for_vis = save_for_vis

        set_random_seed(42)

        if mode == "last":
            ckpt_states = torch.load(
                os.path.join(self.cfg.model_dir, "checkpoint_last.pt"),
                map_location=self.device,
                weights_only=True,
            )
        elif mode == "best":
            ckpt_states = torch.load(
                os.path.join(self.cfg.model_dir, "checkpoint_best.pt"),
                map_location=self.device,
                weights_only=True,
            )
        elif isinstance(mode, int):
            ckpt_states = torch.load(
                os.path.join(self.cfg.model_dir, f"checkpoint_best_{mode}.pt"),
                map_location=self.device,
                weights_only=True,
            )
        else:
            raise ValueError(f"unknown mode: {mode}")

        self.denoiser = self.accelerator.unwrap_model(self.denoiser)
        self.denoiser.load_state_dict(ckpt_states["model"])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(ckpt_states["ema"])

        # testing_mode=False, training_err_check=False
        if eval_on_train:
            fut_traj_gt, _, _ = self.eval_dataloader(training_err_check=True)
        else:
            fut_traj_gt, _, _ = self.eval_dataloader(testing_mode=True)
        self.logger.info(f"testing complete with the {mode} ckpt")

    def sample_from_denoising_model(self, data):
        """
        Return the samples from denoising model in normal scale
        """
        # y_t, y_data_at_t_ls, t_ls, y_t_ls, model_preds.pred_score
        # [B, K, A, T*F], [B, S, K, A, T*F], ,[B, S, K, A, T*F], [B, K, A]
        (
            pred_traj_y,
            pred_traj_y_at_t,
            y_t_seq,
            pred_traj_x,
            pred_traj_x_at_t,
            x_t_seq,
            t_seq,
            pred_score_y,
            pred_score_x,
        ) = self.denoiser.sample(
            data,
            num_trajs=self.cfg.denoising_head_preds,
            return_all_states=self.save_samples,
        )
        # assert list(pred_traj.shape[2:]) == [
        #     self.cfg.agents,
        #     self.cfg.MODEL.MODEL_OUT_DIM,
        # ]
        assert pred_traj_y.shape[3] == self.cfg.MODEL.MODEL_OUT_DIM
        assert pred_traj_x.shape[3] == self.cfg.MODEL.MODEL_IN_DIM

        pred_traj_y = rearrange(
            pred_traj_y, "b k a (f d) -> (b a) k f d", f=self.cfg.future_frames
        )[
            ..., 0:2
        ]  # [B, k_preds, 11, 24] -> [B * 11, k_preds, 12, 2]
        pred_traj_x = rearrange(
            pred_traj_x, "b k a (p d) -> (b a) k p d", p=self.cfg.past_frames
        )[
            ..., 0:2
        ]  # [B, k_preds, 11, 16] -> [B * 11, k_preds, 8, 2]

        pred_traj_y_at_t = rearrange(
            pred_traj_y_at_t, "b t k a (f d) -> (b a) t k f d", f=self.cfg.future_frames
        )[
            ..., 0:2
        ]  # [B, k_preds, 11, 24] -> [B * 11, k_preds, 12, 2]
        if self.cfg.get("data_norm", None) == "min_max":
            pred_traj_y = unnormalize_min_max(
                pred_traj_y, self.cfg.fut_traj_min, self.cfg.fut_traj_max, -1, 1
            )
            pred_traj_y_at_t = unnormalize_min_max(
                pred_traj_y_at_t, self.cfg.fut_traj_min, self.cfg.fut_traj_max, -1, 1
            )
            pred_traj_x = unnormalize_min_max(
                pred_traj_x, self.cfg.past_traj_min, self.cfg.past_traj_max, -1, 1
            )
            pred_traj_x_at_t = None
            # pred_traj_y_at_t = None
        else:
            raise NotImplementedError(
                f"Data normalization [{self.cfg.data_norm}] is not implemented yet."
            )

        return (
            pred_traj_y,
            pred_traj_x,
            pred_traj_y_at_t,
            pred_traj_x_at_t,
            t_seq,
            y_t_seq,
            x_t_seq,
            pred_score_y,
            pred_score_x,
        )

    def save_latent_states(
        self, t_seq_ls, y_t_seq_ls, y_pred_data_ls, x_data_ls, pred_score_ls, file_name
    ):
        self.logger.info("Begin to save the denoising samples...")

        if self.cfg.dataset in ["tbd", "t2fpv", "nba", "sdd", "eth_ucy"]:
            keys_to_save = [
                "past_traj",
                "fut_traj",
                "past_traj_original_scale",
                "fut_traj_original_scale",
                "fut_traj_vel",
            ]
        else:
            raise NotImplementedError(
                f"Dataset [{self.cfg.dataset}] is not implemented yet."
            )

        states_to_save = {k: [] for k in keys_to_save}

        states_to_save["t"] = []
        states_to_save["y_t"] = []
        states_to_save["y_pred_data"] = []
        states_to_save["pred_score"] = []

        for i_batch, (t_seq, y_t_seq, y_pred_data, x_data, pred_score) in enumerate(
            zip(t_seq_ls, y_t_seq_ls, y_pred_data_ls, x_data_ls, pred_score_ls)
        ):
            t = t_seq.detach().cpu().numpy().reshape(1, -1)
            states_to_save["t"].append(t)

            y_t_seq = y_t_seq.detach().cpu().numpy()
            states_to_save["y_t"].append(y_t_seq)

            y_pred_data = y_pred_data.detach().cpu().numpy()
            states_to_save["y_pred_data"].append(y_pred_data)

            pred_score = pred_score.detach().cpu().numpy()
            states_to_save["pred_score"].append(pred_score)

            for key in keys_to_save:
                x_data_val_ = x_data[key].detach().cpu().numpy()
                assert len(y_t_seq) == len(x_data_val_)
                states_to_save[key].append(x_data_val_)

        for key in states_to_save:
            states_to_save[key] = np.concatenate(states_to_save[key], axis=0)

        # clean up the cfg and remove any path related fields
        cfg_ = copy.deepcopy(self.cfg.yml_dict)

        def _remove_path_fields(cfg):
            for k in list(cfg.keys()):
                if "path" in k or "dir" in k:
                    cfg.pop(k)
                elif isinstance(cfg[k], dict):
                    _remove_path_fields(cfg[k])
                else:
                    try:
                        if os.path.isdir(cfg[k]) or os.path.isfile(cfg[k]):
                            cfg.pop(k)
                    except:
                        pass

        _remove_path_fields(cfg_)

        num_datapoints = len(states_to_save["y_t"])
        meta_data = {"cfg": cfg_, "size": num_datapoints}

        states_to_save["meta_data"] = meta_data

        # save_path = os.path.join(self.cfg.sample_dir, f'{file_name}.npz')
        # np.savez_compressed(save_path, **states_to_save)

        save_path = os.path.join(self.cfg.sample_dir, f"{file_name}.pkl")
        self.logger.info("Saving the denoising samples to {}".format(save_path))
        pickle.dump(states_to_save, open(save_path, "wb"))

    def compute_pearson_corr(self, ade, score):
        """
        Compute the Pearson correlation between ADE and score
        """
        ade = ade.flatten()
        score = score.flatten()
        combi = torch.stack([ade, score], dim=0)
        return torch.corrcoef(combi)[0, 1]

    def compute_k_agent_from_distance(
        self, gt_traj, pred_traj, agent_mask, k, pred_score
    ):
        """
        Compute the ADE/FDE for the top k predictions.
        (Note we use the num_trajs in the following log, so use sum)
        @param gt_traj: [B*A, K, T, D]
        @param pred_traj: [B*A, K, T, D]
        @param agent_mask: [B, A]
        @param k: int, top k score in the pred_score
        @param pred_score: [B, K, A]
        """
        # extract the valid trajectories only
        valid_idx = agent_mask.bool().view(-1)
        gt_traj = gt_traj[valid_idx]
        pred_traj = pred_traj[valid_idx]
        pred_score = rearrange(pred_score, "b k a -> (b a) k").unsqueeze(-1)
        pred_score = pred_score[valid_idx]

        distances = (gt_traj - pred_traj).norm(p=2, dim=-1)  # [valid_idx, K, T]
        num_traj = distances.shape[0]

        metrics_k_mode = self.cfg.get("metrics_k_mode", "min")
        if metrics_k_mode == "min":
            _, selected_k = pred_score.topk(k, dim=1)
            selected_k = selected_k.repeat(1, 1, distances.shape[2])
            distances_k = distances.gather(1, selected_k)
        elif metrics_k_mode == "randn":

            selected_k = torch.randint(
                0,
                self.cfg.denoising_head_preds,
                (num_traj, k, distances.shape[2]),
                device=distances.device,
            )
            distances_k = distances.gather(1, selected_k)
        else:
            raise ValueError(f"Unknown metrics_k_mode: {metrics_k_mode}")

        # average over the t frames, min over the k modes, sum over the valid agents
        # in the log stage we add the demon of the valid agents
        ade_best = distances_k.mean(dim=-1).min(dim=-1).values.sum(dim=0)
        fde_best = distances_k[..., -1].min(dim=-1).values.sum(dim=0)
        ade_avg = distances_k.mean(dim=-1).mean(dim=-1).sum(dim=0)
        fde_avg = distances_k[..., -1].mean(dim=-1).sum(dim=0)

        return ade_best, fde_best, ade_avg, fde_avg

    def compute_k_scene_from_distance(
        self, gt_traj, pred_traj, agent_mask, k, pred_score
    ):
        """
        Compute the JADE/JFDE for the top k predictions
        (Note we use the batch_size in the following log, and calculate the scene-level metrics)
        @param gt_traj: [B*A, K, T, D]
        @param pred_traj: [B*A, K, T, D]
        @param agent_mask: [B, A]
        @param k: int, top k score in the pred_score
        @param pred_score: [B, K, A]
        """
        B, A = agent_mask.shape

        gt_traj = rearrange(gt_traj, "(b a) k t d -> b a k t d", b=B, a=A)
        pred_traj = rearrange(pred_traj, "(b a) k t d -> b a k t d", b=B, a=A)
        distances = (gt_traj - pred_traj).norm(p=2, dim=-1)  # [B, A, K, T]

        metrics_k_mode = self.cfg.get("metrics_k_mode", "min")
        pred_score = rearrange(pred_score, "b k a -> b a k")
        if metrics_k_mode == "min":
            _, selected_k = pred_score.topk(k, dim=-1)  # [B, A, k]
            selected_k = repeat(selected_k, "b a k ->  b a k t", t=distances.shape[-1])
            distances_k = distances.gather(2, selected_k)  # [B, A, k, T]
        elif metrics_k_mode == "randn":
            selected_k = torch.randint(
                0,
                self.cfg.denoising_head_preds,
                (B, A, k, distances.shape[-1]),
                device=distances.device,
            )
            distances_k = distances.gather(2, selected_k)
        else:
            raise ValueError(f"Unknown metrics_k_mode: {metrics_k_mode}")

        agent_mask_ = agent_mask
        agent_mask = repeat(agent_mask, "b a -> b a k t", k=k, t=distances.shape[-1])
        assert agent_mask_.sum(dim=1).min() > 0

        jade = (distances_k * agent_mask).mean(dim=-1).sum(dim=1) / agent_mask_.sum(
            dim=1
        ).unsqueeze(-1)
        jade_best = jade.sum(dim=0).min(dim=-1).values
        jade_avg = jade.sum(dim=0).mean(dim=-1)

        jfde = (distances_k[..., -1] * agent_mask[..., -1]).sum(
            dim=1
        ) / agent_mask_.sum(dim=1).unsqueeze(-1)
        jfde_best = jfde.sum(dim=0).min(dim=-1).values
        jfde_avg = jfde.sum(dim=0).mean(dim=-1)

        return jade_best, jfde_best, jade_avg, jfde_avg

    def compute_k_metrics(
        self, gt_traj, pred_traj, agent_mask, pred_score, performance, performance_joint
    ):
        gt_traj = rearrange(gt_traj, "b a t d -> (b a) t d")
        gt_traj = gt_traj.unsqueeze(1).repeat(1, self.cfg.denoising_head_preds, 1, 1)

        k_list = self.cfg.get("K_LIST", [1, 3, 5, 20])
        for idx, k in enumerate(k_list):
            ade, fde, ade_avg, fde_avg = self.compute_k_agent_from_distance(
                gt_traj, pred_traj, agent_mask, k, pred_score
            )
            performance["ADE_min"][idx] += ade.item()
            performance["FDE_min"][idx] += fde.item()
            performance["ADE_avg"][idx] += ade_avg.item()
            performance["FDE_avg"][idx] += fde_avg.item()

            jade, jfde, jade_avg, jfde_avg = self.compute_k_scene_from_distance(
                gt_traj, pred_traj, agent_mask, k, pred_score
            )

            performance_joint["JADE_min"][idx] += jade.item()
            performance_joint["JFDE_min"][idx] += jfde.item()
            performance_joint["JADE_avg"][idx] += jade_avg.item()
            performance_joint["JFDE_avg"][idx] += jfde_avg.item()

        return performance, performance_joint

    def eval_dataloader(self, testing_mode=False, training_err_check=False):
        """
        General API to evaluate the dataloader/dataset
        """
        ### turn on the eval mode
        self.denoiser.eval()
        self.ema.ema_model.eval()
        self.logger.info(f"Record the statistics of samples from the denoising model")

        if testing_mode:
            self.logger.info(f"Start recording test set ADE/FDE...")
            status = "test"
            dl = self.test_loader
        elif training_err_check:
            self.logger.info(f"Start recording training set ADE/FDE...")
            status = "train"
            dl = self.train_loader
        else:
            self.logger.info(f"Start recording validation set ADE/FDE...")
            status = "val"
            dl = self.val_loader

        ### setup the performance dict
        performance_future = {
            "FDE_min": [0, 0, 0, 0],
            "ADE_min": [0, 0, 0, 0],
            "FDE_avg": [0, 0, 0, 0],
            "ADE_avg": [0, 0, 0, 0],
        }
        performance_joint_future = {
            "JFDE_min": [0, 0, 0, 0],
            "JADE_min": [0, 0, 0, 0],
            "JFDE_avg": [0, 0, 0, 0],
            "JADE_avg": [0, 0, 0, 0],
        }
        performance_past = {
            "ADE_min": [0, 0, 0, 0],
            "FDE_min": [0, 0, 0, 0],
            "ADE_avg": [0, 0, 0, 0],
            "FDE_avg": [0, 0, 0, 0],
        }
        performance_joint_past = {
            "JFDE_min": [0, 0, 0, 0],
            "JADE_min": [0, 0, 0, 0],
            "JFDE_avg": [0, 0, 0, 0],
            "JADE_avg": [0, 0, 0, 0],
        }
        num_trajs = 0
        num_scenes = 0

        t_seq_ls, y_t_seq_ls, y_pred_data_ls, x_data_ls = [], [], [], []
        ### record running time
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for i_batch, data in enumerate(dl):
            bs = int(data["batch_size"])
            data = {k: v.to(self.device) for k, v in data.items()}

            (
                pred_traj_y,
                pred_traj_x,
                pred_traj_y_at_t,
                pred_traj_x_at_t,
                t_seq,
                y_t_seq,
                x_t_seq,
                pred_score_y,
                pred_score_x,
            ) = self.sample_from_denoising_model(
                data
            )  # pred_traj: [B*A, K, F, 2], pred_traj_t: [B*A, T, K, F, 2]

            if self.cfg.get("use_ablation_dataset", False):
                B, A = data["past_traj"].shape[:2]
                agent_mask = torch.ones((B, A), device=pred_traj_y.device)
            else:
                agent_mask = data["agent_mask"]

            performance_future, performance_joint_future = self.compute_k_metrics(
                data["fut_traj_original_scale"],
                pred_traj_y,
                agent_mask,
                pred_score_y,
                performance_future,
                performance_joint_future,
            )
            performance_past, performance_joint_past = self.compute_k_metrics(
                data["past_traj_gt_original_scale"][..., :2],
                pred_traj_x,
                agent_mask,
                pred_score_x,
                performance_past,
                performance_joint_past,
            )

            num_trajs += agent_mask.sum().item()  # the valid agents
            num_scenes += bs  # the batch size B

            # save the denoising samples for IMLE
            if self.save_samples:
                raise NotImplementedError("Not implemented yet")

            # save visual samples
            if testing_mode and self.save_for_vis:
                save_tensors = dict()
                save_tensors["hist_obs"] = data["past_traj_original_scale"][
                    ..., :2
                ]  # [B, A, F, 2]
                save_tensors["hist_gt"] = data["past_traj_gt_original_scale"][
                    ..., :2
                ]  # [B, A, F, 2]
                save_tensors["fut_gt"] = data["fut_traj_original_scale"]  # [B, A, T, 2]

                save_tensors["fut_pred"] = pred_traj_y  # [B*A, K, F, 2]
                save_tensors["fut_pred_at_t"] = pred_traj_y_at_t
                save_tensors["hist_pred"] = pred_traj_x
                save_tensors["agent_mask"] = agent_mask
                save_tensors["past_theta"] = data["past_theta"]

                self.save_batch_samples(save_tensors, i_batch)

        end.record()
        torch.cuda.synchronize()
        self.logger.info(f"Total runtime: {start.elapsed_time(end):5f} ms")
        self.logger.info(
            f"Runtime per scene: {start.elapsed_time(end)/len(dl.dataset):5f} ms"
        )
        self.logger.info(f"Number of scenes: {dl.dataset}")
        steps_per_epoch = self.train_num_steps // max(
            self.cfg.OPTIMIZATION.NUM_EPOCHS, 1
        )
        cur_epoch = self.step // steps_per_epoch if steps_per_epoch > 0 else 0

        k_list = self.cfg.get("K_LIST", [1, 3, 5, 20])

        if not testing_mode:
            self.logger.info(
                f"{self.step}/{self.train_num_steps}, running inference on {num_trajs} agents (trajectories)"
            )
            # although no factor added, but still same meaning for the time
            for idx, k in enumerate(k_list):
                if self.tb_log:
                    self.tb_log.add_scalar(
                        f"eval_{status}/F_ADE_min_k{k}",
                        performance_future["ADE_min"][idx] / num_trajs,
                        cur_epoch,
                    )
                    self.tb_log.add_scalar(
                        f"eval_{status}/F_FDE_min_k{k}",
                        performance_future["FDE_min"][idx] / num_trajs,
                        cur_epoch,
                    )
                    self.tb_log.add_scalar(
                        f"eval_{status}/F_ADE_avg_k{k}",
                        performance_future["ADE_avg"][idx] / num_trajs,
                        cur_epoch,
                    )
                    self.tb_log.add_scalar(
                        f"eval_{status}/F_FDE_avg_k{k}",
                        performance_future["FDE_avg"][idx] / num_trajs,
                        cur_epoch,
                    )
                    self.tb_log.add_scalar(
                        f"eval_{status}/J_JADE_min_k{k}",
                        performance_joint_future["JADE_min"][idx] / num_scenes,
                        cur_epoch,
                    )
                    self.tb_log.add_scalar(
                        f"eval_{status}/J_JFDE_min_k{k}",
                        performance_joint_future["JFDE_min"][idx] / num_scenes,
                        cur_epoch,
                    )
                    self.tb_log.add_scalar(
                        f"eval_{status}/J_JADE_avg_k{k}",
                        performance_joint_future["JADE_avg"][idx] / num_scenes,
                        cur_epoch,
                    )
                    self.tb_log.add_scalar(
                        f"eval_{status}/J_JFDE_avg_k{k}",
                        performance_joint_future["JFDE_avg"][idx] / num_scenes,
                        cur_epoch,
                    )

                    self.tb_log.add_scalar(
                        f"eval_{status}/P_ADE_min_k{k}",
                        performance_past["ADE_min"][idx] / num_trajs,
                        cur_epoch,
                    )
                    self.tb_log.add_scalar(
                        f"eval_{status}/P_FDE_min_k{k}",
                        performance_past["FDE_min"][idx] / num_trajs,
                        cur_epoch,
                    )
                    self.tb_log.add_scalar(
                        f"eval_{status}/P_ADE_avg_k{k}",
                        performance_past["ADE_avg"][idx] / num_trajs,
                        cur_epoch,
                    )
                    self.tb_log.add_scalar(
                        f"eval_{status}/P_FDE_avg_k{k}",
                        performance_past["FDE_avg"][idx] / num_trajs,
                        cur_epoch,
                    )
                    self.tb_log.add_scalar(
                        f"eval_{status}/P_JADE_min_k{k}",
                        performance_joint_past["JADE_min"][idx] / num_scenes,
                        cur_epoch,
                    )
                    self.tb_log.add_scalar(
                        f"eval_{status}/P_JFDE_min_k{k}",
                        performance_joint_past["JFDE_min"][idx] / num_scenes,
                        cur_epoch,
                    )
                    self.tb_log.add_scalar(
                        f"eval_{status}/P_JADE_avg_k{k}",
                        performance_joint_past["JADE_avg"][idx] / num_scenes,
                        cur_epoch,
                    )
                    self.tb_log.add_scalar(
                        f"eval_{status}/P_JFDE_avg_k{k}",
                        performance_joint_past["JFDE_avg"][idx] / num_scenes,
                        cur_epoch,
                    )

        # print out the performance
        for idx, k in enumerate(k_list):
            self.logger.info(
                "--Future--ADE_min(k{}): {:.7f}\t--FDE_min(k{}): {:.7f}".format(
                    k,
                    performance_future["ADE_min"][idx] / num_trajs,
                    k,
                    performance_future["FDE_min"][idx] / num_trajs,
                )
            )

        for idx, k in enumerate(k_list):
            self.logger.info(
                "--Future--ADE_avg(k{}): {:.7f}\t--FDE_avg(k{}): {:.7f}".format(
                    k,
                    performance_future["ADE_avg"][idx] / num_trajs,
                    k,
                    performance_future["FDE_avg"][idx] / num_trajs,
                )
            )

        for idx, k in enumerate(k_list):
            self.logger.info(
                "--Future--JADE_min(k{}): {:.7f}\t--JFDE_min(k{}): {:.7f}".format(
                    k,
                    performance_joint_future["JADE_min"][idx] / num_scenes,
                    k,
                    performance_joint_future["JFDE_min"][idx] / num_scenes,
                )
            )

        for idx, k in enumerate(k_list):
            self.logger.info(
                "--Future--JADE_avg(k{}): {:.7f}\t--JFDE_avg(k{}): {:.7f}".format(
                    k,
                    performance_joint_future["JADE_avg"][idx] / num_scenes,
                    k,
                    performance_joint_future["JFDE_avg"][idx] / num_scenes,
                )
            )

        for idx, k in enumerate(k_list):
            self.logger.info(
                "--Past--ADE_min(k{}): {:.7f}\t--FDE_min(k{}): {:.7f}".format(
                    k,
                    performance_past["ADE_min"][idx] / num_trajs,
                    k,
                    performance_past["FDE_min"][idx] / num_trajs,
                )
            )

        for idx, k in enumerate(k_list):
            self.logger.info(
                "--Past--ADE_avg(k{}): {:.7f}\t--FDE_avg(k{}): {:.7f}".format(
                    k,
                    performance_past["ADE_avg"][idx] / num_trajs,
                    k,
                    performance_past["FDE_avg"][idx] / num_trajs,
                )
            )

        for idx, k in enumerate(k_list):
            self.logger.info(
                "--Past--JADE_min(k{}): {:.7f}\t--JFDE_min(k{}): {:.7f}".format(
                    k,
                    performance_joint_past["JADE_min"][idx] / num_scenes,
                    k,
                    performance_joint_past["JFDE_min"][idx] / num_scenes,
                )
            )

        for idx, k in enumerate(k_list):
            self.logger.info(
                "--Past--JADE_avg(k{}): {:.7f}\t--JFDE_avg(k{}): {:.7f}".format(
                    k,
                    performance_joint_past["JADE_avg"][idx] / num_scenes,
                    k,
                    performance_joint_past["JFDE_avg"][idx] / num_scenes,
                )
            )

        fut_traj_gt = rearrange(data["fut_traj_original_scale"], "b a t d -> (b a) t d")
        return fut_traj_gt, performance_future, num_trajs

    def save_batch_samples(self, save_tensors, i_batch, prefix="visual"):
        """save the samples"""
        save_path = os.path.join(
            self.cfg.sample_dir, f"{prefix}_mode_{self.mode}_batch_{i_batch}.pkl"
        )

        fut_gt = rearrange(save_tensors["fut_gt"], "b a f d -> (b a) f d")
        fut_gt_ = repeat(fut_gt, "b f d -> b k f d", k=self.cfg.denoising_head_preds)
        fut_pred = save_tensors["fut_pred"]
        fut_pred_at_t = save_tensors["fut_pred_at_t"]  # [B*A, K, T_denoising, F, 2]

        hist_gt = rearrange(save_tensors["hist_gt"], "b a f d -> (b a) f d")
        hist_gt_ = repeat(hist_gt, "b f d -> b k f d", k=self.cfg.denoising_head_preds)
        hist_pred = save_tensors["hist_pred"]
        hist_obs = rearrange(save_tensors["hist_obs"], "b a f d -> (b a) f d")

        valid_idx = save_tensors["agent_mask"].bool().flatten()
        past_theta = save_tensors["past_theta"].flatten()[valid_idx]
        fut_gt = fut_gt[valid_idx]
        fut_pred = fut_pred[valid_idx]
        fut_pred_at_t = fut_pred_at_t[valid_idx]
        hist_obs = hist_obs[valid_idx]
        hist_gt = hist_gt[valid_idx]
        hist_pred = hist_pred[valid_idx]

        save_tensors["fut_gt"] = fut_gt
        save_tensors["fut_pred"] = fut_pred
        save_tensors["fut_pred_at_t"] = fut_pred_at_t
        save_tensors["hist_obs"] = hist_obs
        save_tensors["hist_gt"] = hist_gt
        save_tensors["hist_pred"] = hist_pred
        save_tensors["past_theta"] = past_theta

        fut_gt_ = fut_gt_[valid_idx]
        hist_gt_ = hist_gt_[valid_idx]

        distances_fut = (fut_gt_ - fut_pred).norm(p=2, dim=-1).sum(dim=-1)
        save_tensors["fut_best_idx"] = distances_fut.argmin(dim=-1)

        distances_hist = (hist_gt_ - hist_pred).norm(p=2, dim=-1).sum(dim=-1)
        save_tensors["hist_best_idx"] = distances_hist.argmin(dim=-1)

        for k, v in save_tensors.items():
            save_tensors[k] = v.detach().cpu().numpy()

        pickle.dump(save_tensors, open(save_path, "wb"))
        return
