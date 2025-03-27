import logging
import sys

import jax.experimental
import jax.experimental.multihost_utils

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
import argparse
import glob
import time
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import torch
from tqdm.auto import tqdm

import wandb
from src.config import Config
from src.lr_scheduler import get_learning_rate_scheduler
from src.model import Transformer


def save_state(state, out: str):
    checkpointer = ocp.StandardCheckpointer()
    # nnx using jax.random.key, but it seems orbax not supporting now. Workaround is do not save rng_state
    graphdef, rng_state, other_state = nnx.split(state, nnx.RngState, ...)
    checkpointer.save(out, other_state)
    checkpointer.wait_until_finished()


def load_state(src: str, state):
    checkpointer = ocp.StandardCheckpointer()
    graphdef, rng_state, other_state = nnx.split(state, nnx.RngState, ...)
    state_restored = checkpointer.restore(src, other_state)
    nnx.update(state, state_restored)


def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)  # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])  # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=False)  # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())  # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


def data_generator(filename_pattern: str, batch_size: int, context_length: int):
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    flatten_batch_size = batch_size * context_length
    file_iter = iter(files)  # use itertools.cycle(files) instead if you want to do multi-epoch training
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + flatten_batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos : pos + flatten_batch_size + 1]
        inputs = buf[:-1].reshape(batch_size, context_length)
        targets = buf[1:].reshape(batch_size, context_length)
        pos += flatten_batch_size
        yield inputs.numpy(), targets.numpy()


def train(config: Config):
    seed = 998244353
    rngs = nnx.Rngs(params=seed, dropout=seed)

    num_devices = jax.local_device_count()
    mesh = jax.sharding.Mesh(jax.experimental.mesh_utils.create_device_mesh((num_devices,)), ("data",))
    model_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))

    model_config = config.transformer_config
    train_config = config.train_config

    model = Transformer(model_config, rngs)

    params = nnx.state(model, nnx.Param)
    shapes = jax.tree_util.tree_map(lambda x: jnp.prod(jnp.array(jnp.shape(x))), params)
    num_params = jax.tree_util.tree_reduce(lambda x, y: x + y, shapes)
    logging.info(f"Number of parameters: {num_params/1e6:.2f}M")

    # apply weight decay on 2D shaped parameters
    decay_mask = jax.tree_util.tree_map(lambda x: len(x.shape) == 2, params)

    learning_rate_fn = get_learning_rate_scheduler(
        train_config.learning_rate_scheduler,
        lr=train_config.learning_rate,
        warmup_steps=train_config.warmup_steps,
        decay_steps=train_config.decay_steps,
        total_steps=train_config.total_steps,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(train_config.max_grad_norm),
        optax.adamw(
            learning_rate=learning_rate_fn,
            b1=train_config.beta1,
            b2=train_config.beta2,
            weight_decay=train_config.weight_decay,
            mask=decay_mask,
        ),
    )
    state = nnx.state((model, optimizer))
    state = jax.device_put(state, model_sharding)
    nnx.update((model, optimizer), state)

    state = nnx.Optimizer(model, optimizer)

    vocab_size = model_config.vocab_size

    def loss_fn(model, x, y):
        # Do not care about IGNORE_INDEX now
        logits = model(x)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits.reshape(-1, vocab_size), y.reshape(-1))
        return loss.mean()

    @nnx.jit
    def train_step(state, x, y):
        grad_fn = nnx.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.model, x, y)
        # x and y is sharded on each device, but mean performs all-reduce, so loss is on every device now
        # also grads is on every device without sharding
        total_norm = optax.global_norm(grads)
        state.update(grads)
        return {"loss": loss, "grad_norm": total_norm}

    @nnx.jit
    def validation_step(model, x, y):
        loss = loss_fn(model, x, y)
        return loss

    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        grad_norm=nnx.metrics.Average("grad_norm"),
    )

    per_device_batch_size = train_config.per_device_batch_size
    context_length = model_config.context_length

    num_devices = jax.device_count()
    train_dataloader = data_generator(
        "data/finewebedu10B/finewebedu_train_*.bin", per_device_batch_size * num_devices, context_length
    )
    elapsed_time = 0

    log_wandb = False
    if train_config.wandb_project_name:
        run = wandb.init(
            project=train_config.wandb_project_name,
            config=config.to_dict(),
            name=train_config.wandb_run_name,
        )
        log_wandb = True

    t0 = time.perf_counter()
    for step in tqdm(range(1, train_config.total_steps + 1), desc="Training"):
        inputs, targets = next(train_dataloader)
        inputs, targets = jax.device_put((inputs, targets), data_sharding)

        step_metrics = train_step(state, inputs, targets)
        metrics.update(loss=step_metrics["loss"], grad_norm=step_metrics["grad_norm"])
        if step % train_config.log_interval == 0:
            jax.experimental.multihost_utils.sync_global_devices("barrier")
            log_dict = metrics.compute()
            elapsed_time += time.perf_counter() - t0
            throughput = per_device_batch_size * context_length * num_devices * train_config.log_interval / elapsed_time
            elapsed_time = 0
            log_dict.update(learning_rate=learning_rate_fn(step), throughput=throughput)
            metrics.reset()
            if log_wandb:
                run.log(log_dict, step=step)
            logging.info(f"Step {step}: {log_dict}")
            t0 = time.perf_counter()

        if step % train_config.val_interval == 0:
            jax.experimental.multihost_utils.sync_global_devices("barrier")
            elapsed_time += time.perf_counter() - t0

            model.eval()
            val_dataloader = data_generator(
                "data/finewebedu10B/finewebedu_val_*.bin", per_device_batch_size * num_devices, context_length
            )
            val_steps = 100000000 // (per_device_batch_size * num_devices * context_length)
            val_losses = []
            for _ in tqdm(range(val_steps), desc="Validation", leave=False):
                inputs, targets = next(val_dataloader)
                inputs, targets = jax.device_put((inputs, targets), data_sharding)
                loss = validation_step(state.model, inputs, targets)
                val_losses.append(loss)
            val_loss = jnp.mean(jnp.array(val_losses))
            logging.info(f"Validation Loss at step {step}: {val_loss:.4f}")
            if log_wandb:
                run.log({"val_loss": val_loss}, step=step)
            model.train()
            t0 = time.perf_counter()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str, required=True)

    args = parser.parse_args()
    config = Config.from_json(args.config_path)
    train(config)
