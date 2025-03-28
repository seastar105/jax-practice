import copy

from src.config import Config

if __name__ == "__main__":
    # baseline
    base_cfg = Config()
    base_cfg.train_config.wandb_run_name = "baseline"
    base_cfg.to_json("configs/base.json", indent=4)

    # rmsnorm
    rmsnorm_cfg = copy.deepcopy(base_cfg)
    rmsnorm_cfg.transformer_config.norm_class = "rmsnorm"
    rmsnorm_cfg.train_config.wandb_run_name = "rmsnorm"
    rmsnorm_cfg.to_json("configs/rmsnorm.json", indent=4)

    # qk_norm
    qk_norm_cfg = copy.deepcopy(base_cfg)
    qk_norm_cfg.transformer_config.qk_norm = True
    qk_norm_cfg.train_config.wandb_run_name = "qk_norm"
    qk_norm_cfg.to_json("configs/qk_norm.json", indent=4)

    # double_head_dim
    double_head_dim_cfg = copy.deepcopy(base_cfg)
    double_head_dim_cfg.transformer_config.num_heads //= 2
    double_head_dim_cfg.train_config.wandb_run_name = "double_head_dim"
    double_head_dim_cfg.to_json("configs/double_head_dim.json", indent=4)

    # rotary
    rotary_cfg = copy.deepcopy(base_cfg)
    rotary_cfg.transformer_config.pos_emb = "rotary"
    rotary_cfg.train_config.wandb_run_name = "rotary"
    rotary_cfg.to_json("configs/rotary.json", indent=4)

    # untie
    untie_cfg = copy.deepcopy(base_cfg)
    untie_cfg.transformer_config.tie_embedding = False
    untie_cfg.train_config.wandb_run_name = "untie"
    untie_cfg.to_json("configs/untie.json", indent=4)

    # wsd schedule
    wsd_cfg = copy.deepcopy(base_cfg)
    wsd_cfg.train_config.learning_rate_scheduler = "linear_warmup_stable_decay"
    wsd_cfg.train_config.wandb_run_name = "wsd"
    wsd_cfg.to_json("configs/wsd.json", indent=4)

    # 3x lr
    lr3x_cfg = copy.deepcopy(base_cfg)
    lr3x_cfg.train_config.learning_rate *= 3
    lr3x_cfg.train_config.wandb_run_name = "lr3x"
    lr3x_cfg.to_json("configs/lr3x.json", indent=4)

    # double batch size
    double_batch_cfg = copy.deepcopy(base_cfg)
    double_batch_cfg.train_config.per_device_batch_size *= 2
    double_batch_cfg.transformer_config.use_remat = True
    double_batch_cfg.train_config.wandb_run_name = "double_batch"
    double_batch_cfg.to_json("configs/double_batch.json", indent=4)

    # wsd + 3x lr + double head + rms + qk_norm + rotary
    combined_cfg = copy.deepcopy(base_cfg)
    combined_cfg.transformer_config.num_heads //= 2
    combined_cfg.transformer_config.pos_emb = "rotary"
    combined_cfg.transformer_config.norm_class = "rmsnorm"
    combined_cfg.transformer_config.qk_norm = True
    combined_cfg.train_config.learning_rate_scheduler = "linear_warmup_stable_decay"
    combined_cfg.train_config.learning_rate *= 3
    combined_cfg.to_json("configs/combined.json", indent=4)
