from ml_collections.config_dict import ConfigDict


def get_config():

    cfg = ConfigDict()

    # ----------------
    # Train
    # ----------------

    training = cfg.training = ConfigDict()
    training.num_epochs = 100
    training.batch_size = 64
    training.save_ckpt_freq = 10
    training.eval_freq = 10

    # ----------------
    # Model
    # ----------------

    model = cfg.model = ConfigDict()
    model.clip_grad_norm = 1.
    model.ema = True
    model.ema_rate = 0.999
    model.ema_steps = 1

    # ----------------
    # Optimization
    # ----------------

    cfg.optim = optim = ConfigDict()
    optim.optimizer = 'AdamW'
    optim.schedule = 'CosineAnnealingLR'
    optim.grad_clip = 1.
    optim.initial_lr = 0.0005
    optim.weight_decay = 0.0001
    optim.min_lr = 0.001 * optim.initial_lr
    optim.warmup_epochs = None

    # ----------------
    # Data
    # ----------------

    cfg.data = data = ConfigDict()
    data.num_workers = 2
    data.prefetch_factor = 1

    cfg.seed = 42
    cfg.distributed = True
    cfg.use_deterministic_algorithms = True
    cfg.debug = False

    return cfg