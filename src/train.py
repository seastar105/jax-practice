import logging
import sys

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
import argparse
import time
from typing import Optional, Union

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from tqdm.auto import tqdm

from src.model import Transformer
from src.optimizer import LEARNING_RATE_SCHEDULES, get_learning_rate_scheduler


class NumpyDataset:
    def __init__(self, data: Union[str, np.ndarray], seed: Optional[int] = None):
        if isinstance(data, str):
            data = np.load(data, mmap_mode="r")

        self.data = data
        self.seed = seed
        self.data_length = len(self.data)
        self.generator = None if seed is None else np.random.RandomState(seed)

    def get_sample(self, context_length: int):
        if self.generator is not None:
            start_idx = self.generator.randint(0, self.data_length - context_length)
        else:
            start_idx = np.random.randint(0, self.data_length - context_length)
        sample = self.data[start_idx : start_idx + context_length + 1]
        x = sample[:-1]
        y = sample[1:]
        assert x.shape == y.shape
        return x.astype(np.int64), y.astype(np.int64)

    def get_batch(self, batch_size: int, context_length: int):
        batch_length = batch_size * context_length
        if self.generator is not None:
            offset = self.generator.randint(0, self.data_length - batch_length)
        else:
            offset = np.random.randint(0, self.data_length - batch_length)
        x_batch = self.data[offset : offset + batch_length].astype(np.int64)
        y_batch = self.data[offset + 1 : offset + batch_length + 1].astype(np.int64)
        return x_batch.reshape(batch_size, context_length), y_batch.reshape(batch_size, context_length)
        return jnp.array(x_batch).reshape(batch_size, context_length), jnp.array(y_batch).reshape(
            batch_size, context_length
        )

    def get_validation_batches(self, batch_size: int, context_length: int):
        batch_length = batch_size * context_length
        num_batches = (self.data_length - 1) // batch_length
        for batch_idx in tqdm(range(num_batches), desc="Validation", leave=False):
            offset = batch_idx * batch_length
            x_batch = self.data[offset : offset + batch_length].astype(np.int64)
            y_batch = self.data[offset + 1 : offset + batch_length + 1].astype(np.int64)
            yield x_batch.reshape(batch_size, context_length), y_batch.reshape(batch_size, context_length)
            # yield jnp.array(x_batch).reshape(batch_size, context_length), jnp.array(y_batch).reshape(
            #     batch_size, context_length
            # )


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


def train(args):
    seed = 998244353
    train_data = NumpyDataset(args.train_data, seed=seed)
    val_data = NumpyDataset(args.val_data)

    rngs = nnx.Rngs(params=seed, dropout=seed)

    model = Transformer(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        dim=args.dim,
        dim_ff=args.dim_ff,
        num_heads=args.num_heads,
        rngs=rngs,
        context_length=args.context_length,
        ff_activation=args.activation,
        ff_dropout=args.ff_pdrop,
        attention_dropout=args.attn_pdrop,
        residual_dropout=args.residual_pdrop,
        use_bias=args.use_bias,
        norm_class=args.norm_class,
        use_glu=args.use_glu,
    )
    _, params, _ = nnx.split(model, nnx.Param, ...)
    params = jax.tree_util.tree_map(lambda x: jnp.prod(jnp.array(jnp.shape(x))), params)
    num_params = jax.tree_util.tree_reduce(lambda x, y: x + y, params)
    logging.info(f"Number of parameters: {num_params/1e6:.2f}M")

    learning_rate_fn = get_learning_rate_scheduler(
        args.learning_rate_scheduler,
        lr=args.learning_rate,
        warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps,
        total_steps=args.total_steps,
    )

    optimizer = optax.adamw(
        learning_rate=learning_rate_fn,
        b1=args.beta1,
        b2=args.beta2,
        weight_decay=args.weight_decay,
    )

    state = nnx.Optimizer(model, optimizer)

    def loss_fn(model, x, y):
        # Do not care about IGNORE_INDEX now
        logits = model(x)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits.reshape(-1, args.vocab_size), y.reshape(-1))
        return loss.mean()

    @nnx.jit
    def train_step(state, x, y):
        grad_fn = nnx.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.model, x, y)
        total_norm = optax.global_norm(grads)

        def clip_fn(t):
            return jax.lax.select(jnp.squeeze(total_norm < args.max_grad_norm), t, t * args.max_grad_norm / total_norm)

        grads = jax.tree_util.tree_map(clip_fn, grads)
        state.update(grads)
        return {"loss": loss, "grad_norm": total_norm}

    @nnx.jit
    def validation_step(model, x, y):
        loss = loss_fn(model, x, y)
        return loss

    def validation_loop(model, val_data, args):
        model.eval()
        losses = []
        for inputs, targets in val_data.get_validation_batches(args.batch_size, args.context_length):
            loss = validation_step(model, inputs, targets)
            losses.append(loss)
        val_loss = jnp.stack(losses).mean().item()
        model.train()
        return val_loss

    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        grad_norm=nnx.metrics.Average("grad_norm"),
    )
    model.train()
    start = time.perf_counter()
    for step in tqdm(range(1, args.total_steps + 1), desc="Training"):
        inputs, targets = train_data.get_batch(args.batch_size, args.context_length)

        step_metrics = train_step(state, inputs, targets)
        metrics.update(loss=step_metrics["loss"], grad_norm=step_metrics["grad_norm"])
        if step % args.log_interval == 0:
            end = time.perf_counter()
            throughput = args.log_interval * args.batch_size * args.context_length / (end - start)
            log_dict = metrics.compute()
            log_dict.update(learning_rate=learning_rate_fn(step), step=step, throughput=throughput)
            metrics.reset()
            logging.info(log_dict)
            start = time.perf_counter()

        if step % args.val_interval == 0:
            val_loss = validation_loop(state.model, val_data, args)
            logging.info(f"Validation Loss at step {step}: {val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # I/O Parameters
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default=None)

    # Model Parameters
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--dim_ff", type=int, default=2048)
    parser.add_argument("--attn_pdrop", type=float, default=0.1)
    parser.add_argument("--ff_pdrop", type=float, default=0.1)
    parser.add_argument("--residual_pdrop", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--use_bias", action="store_true")
    parser.add_argument("--use_glu", action="store_true")
    parser.add_argument("--norm_class", type=str, default="rmsnorm", choices=["rmsnorm", "layernorm"])

    # Training Parameters
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_interval", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--learning_rate_scheduler",
        type=str,
        default="linear_warmup_cosine_decay",
        choices=list(LEARNING_RATE_SCHEDULES.keys()),
    )
    parser.add_argument("--total_steps", type=int, default=10000)
    parser.add_argument("--warmup_steps", type=int, default=400)
    parser.add_argument("--decay_steps", type=int, default=400)

    args = parser.parse_args()
    train(args)
