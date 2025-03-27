import optax


def linear_warmup_stable_decay(lr: float, warmup_steps: int, decay_steps: int, total_steps: int, **kwargs):
    warmup_phase = optax.linear_schedule(init_value=0.0, end_value=lr, transition_steps=warmup_steps)
    stable_phase = optax.constant_schedule(lr)
    decay_phase = optax.linear_schedule(init_value=lr, end_value=0.0, transition_steps=decay_steps)
    schedule = optax.join_schedules(
        schedules=[warmup_phase, stable_phase, decay_phase],
        boundaries=[warmup_steps, total_steps - decay_steps],
    )
    return schedule


def linear_warmup_linear_decay(lr: float, warmup_steps: int, total_steps: int, **kwargs):
    warmup_phase = optax.linear_schedule(init_value=0.0, end_value=lr, transition_steps=warmup_steps)
    decay_phase = optax.linear_schedule(init_value=lr, end_value=0.0, transition_steps=total_steps - warmup_steps)
    schedule = optax.join_schedules(
        schedules=[warmup_phase, decay_phase],
        boundaries=[warmup_steps],
    )
    return schedule


def linear_warmup_constant(lr: float, warmup_steps: int, total_steps: int, **kwargs):
    warmup_phase = optax.linear_schedule(init_value=0.0, end_value=lr, transition_steps=warmup_steps)
    constant_phase = optax.constant_schedule(lr)
    schedule = optax.join_schedules(
        schedules=[warmup_phase, constant_phase],
        boundaries=[warmup_steps],
    )
    return schedule


def linear_warmup_cosine_decay(lr: float, warmup_steps: int, total_steps: int, **kwargs):
    warmup_phase = optax.linear_schedule(init_value=0.0, end_value=lr, transition_steps=warmup_steps)
    decay_phase = optax.cosine_decay_schedule(init_value=lr, decay_steps=total_steps - warmup_steps)
    schedule = optax.join_schedules(
        schedules=[warmup_phase, decay_phase],
        boundaries=[warmup_steps],
    )
    return schedule


LEARNING_RATE_SCHEDULES = {
    "linear_warmup_stable_decay": linear_warmup_stable_decay,
    "linear_warmup_linear_decay": linear_warmup_linear_decay,
    "linear_warmup_constant": linear_warmup_constant,
    "linear_warmup_cosine_decay": linear_warmup_cosine_decay,
}


def get_learning_rate_scheduler(scheduler_name: str, **kwargs):
    if scheduler_name not in LEARNING_RATE_SCHEDULES:
        raise ValueError(f"Unknown learning rate scheduler: {scheduler_name}")
    return LEARNING_RATE_SCHEDULES[scheduler_name](**kwargs)
