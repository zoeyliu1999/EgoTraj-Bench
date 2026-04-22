import yaml
import os
import glob
import numpy as np
from easydict import EasyDict
from .utils import create_logger


def noise_module_inherit_cfg(yml_dict):

    def inherit_cfg(model_cfg, cfg_dict):
        for key, value in cfg_dict.items():
            if key not in model_cfg:
                model_cfg[key] = value
        return model_cfg

    if yml_dict.MODEL.get("PAST_TRAJ_MODE", None) == "social_context_regen":
        past_frames = yml_dict.past_frames
        future_frames = yml_dict.future_frames

        cfg_dict = {
            "past_frames": past_frames,
            "future_frames": future_frames,
        }
        yml_dict.MODEL = inherit_cfg(yml_dict.MODEL, cfg_dict)

    return yml_dict


class Config:
    def __init__(self, cfg_path, tag, train_mode=True):
        self.cfg_path = cfg_path
        self.cfg_name = (
            os.path.basename(cfg_path).replace(".yaml", "").replace(".yml", "")
        )
        self.tag = tag
        self.train_mode = train_mode
        files = glob.glob(cfg_path, recursive=True)
        assert len(files) == 1, "YAML file [{}] does not exist!".format(cfg_path)
        yml_dict_ = EasyDict(yaml.safe_load(open(files[0], "r")))
        if train_mode:
            self.yml_dict = yml_dict_
            self.results_root_dir = os.path.expanduser(
                self.yml_dict["results_root_dir"]
            )
        else:
            yml_dict_.cfg_path = cfg_path  # renew the config with the non-train mode
            yml_dict_.cfg_name = self.cfg_name
            yml_dict_.tag = tag
            yml_dict_.train_mode = train_mode
            self.yml_dict = yml_dict_
        self.ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # add the copy_dup_conf function to the yml_dict
        self.yml_dict = noise_module_inherit_cfg(self.yml_dict)

    def create_dirs(self, tag_suffix=None):
        # results dirs
        tag = self.tag if tag_suffix is None else self.tag + tag_suffix
        if self.train_mode:
            self.cfg_dir = "%s/%s/%s" % (self.results_root_dir, self.cfg_name, tag)
        else:
            # extracted from the ckpt model dir
            self.cfg_dir = os.path.dirname(self.cfg_path)

        self.model_dir = "%s/models" % self.cfg_dir
        self.log_dir = "%s/log" % self.cfg_dir
        self.sample_dir = "%s/samples" % self.cfg_dir
        self.model_path = os.path.join(self.model_dir, "model_%04d.p")

        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self.model_files = glob.glob(os.path.join(self.model_dir, "model_*.p"))

        if self.train_mode:
            log_file = os.path.join(self.log_dir, "log.txt")
        else:
            log_file = os.path.join(
                self.log_dir, "log_eval_{:s}.txt".format(tag).replace("__", "_")
            )

        logger = create_logger(log_file)
        self.logger = logger

        # update the yaml file
        for key in sorted(dir(self)):
            if not key.startswith("__") and not callable(getattr(self, key)):
                if key in ["yml_dict", "logger"]:
                    continue
                if key not in self.yml_dict:
                    logger.info("New key {} ---> {}".format(key, getattr(self, key)))
                    self.yml_dict[key] = getattr(self, key)
                else:
                    orig_val = self.yml_dict[key]
                    new_val = getattr(self, key)
                    if orig_val != new_val:
                        logger.info(
                            "Existing key {} ---> {} from {}".format(
                                key, new_val, orig_val
                            )
                        )
                        self.yml_dict[key] = new_val

        if self.train_mode:
            # save the updated yaml file
            os.system(
                "cp %s %s" % (self.cfg_path, self.cfg_dir)
            )  # copy original config
            self.save_updated_yml()

        return logger

    def save_updated_yml(self):
        """Dump the current yml_dict to {cfg_name}_updated.yml in cfg_dir."""

        def easydict_to_dict(easydict_obj):
            result = {}
            for key, value in easydict_obj.items():
                if isinstance(value, EasyDict):
                    result[key] = easydict_to_dict(value)
                else:
                    result[key] = value
            return result

        nested_dict = easydict_to_dict(self.yml_dict)
        with open(
            os.path.join(self.cfg_dir, "{:s}_updated.yml".format(self.cfg_name)),
            "w",
        ) as f:
            yaml.dump(nested_dict, f)

    def get_last_epoch(self):
        model_files = glob.glob(os.path.join(self.model_dir, "model_*.p"))
        if len(model_files) == 0:
            return None
        else:
            model_file = os.path.basename(model_files[-1])
            epoch = int(os.path.splitext(model_file)[0].split("model_")[-1])
            return epoch

    def get_latest_ckpt(self):
        model_files = glob.glob(os.path.join(self.model_dir, "model_*.p"))
        if len(model_files) == 0:
            return None
        else:
            epochs = np.array(
                [int(os.path.splitext(f)[0].split("model_")[-1]) for f in model_files]
            )
            last_epoch = epochs.max()
            fp = os.path.join(self.model_dir, "model_%04d.p" % last_epoch)
            return fp

    def __getattribute__(self, name):
        try:
            yml_dict = super().__getattribute__("yml_dict")
        except AttributeError:
            return super().__getattribute__(
                name
            )  # Return default attribute if yml_dict is not set
        if name in yml_dict:
            return yml_dict[name]
        else:
            return super().__getattribute__(name)

    def __setattr__(self, name, value):
        try:
            yml_dict = super().__getattribute__("yml_dict")
        except AttributeError:
            return super().__setattr__(name, value)
        if name in yml_dict:
            yml_dict[name] = value
        else:
            return super().__setattr__(name, value)

    def get(self, name, default=None):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            return default
