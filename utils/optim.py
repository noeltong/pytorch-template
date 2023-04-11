import torch


def get_optim(model, config):
    init_lr = config.optim.initial_lr * config.training.batch_size / 256

    if config.optim.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=init_lr,
            weight_decay=config.optim.weight_decay,
        )
    elif config.optim.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=init_lr,
            weight_decay=config.optim.weight_decay
        )
    else:
        raise NotImplementedError(
            f'{config.optim.optimizer} is not supported.')

    if config.optim.schedule.lower() is not None and config.optim.schedule.lower() == 'cosineannealinglr':
        if config.optim.warmup_epochs is None:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.training.num_epochs,
                eta_min=config.optim.min_lr,
            )
        else:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=config.optim.initial_lr / config.optim.warmup_epochs,
                total_iters=config.optim.warmup_epochs
            )
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.training.num_epochs - config.optim.warmup_epochs,
                eta_min=config.optim.min_lr
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[config.optim.warmup_epochs]
            )
    else:
        raise ValueError(f'{config.optim.schedule} is not supported.')
    
    return optimizer, scheduler