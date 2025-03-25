import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import wandb
from cs336_basics.model import TransformerLM, cross_entropy_loss
from cs336_basics.optimizer import AdamW, clip_grad_norm, get_cosine_schedule_lr
from tqdm.auto import tqdm

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)


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

    def get_batch(self, batch_size: int, context_length: int, device: Union[torch.device, str]):
        batch_length = batch_size * context_length
        if self.generator is not None:
            offset = self.generator.randint(0, self.data_length - batch_length)
        else:
            offset = np.random.randint(0, self.data_length - batch_length)
        x_batch = self.data[offset : offset + batch_length].astype(np.int64)
        y_batch = self.data[offset + 1 : offset + batch_length + 1].astype(np.int64)
        return torch.from_numpy(x_batch).reshape(batch_size, context_length).to(device), torch.from_numpy(
            y_batch
        ).reshape(batch_size, context_length).to(device)

    def get_validation_batches(self, batch_size: int, context_length: int, device: Union[torch.device, str]):
        batch_length = batch_size * context_length
        num_batches = (self.data_length - 1) // batch_length
        for batch_idx in tqdm(range(num_batches), desc="Validation", leave=False):
            offset = batch_idx * batch_length
            x_batch = self.data[offset : offset + batch_length].astype(np.int64)
            y_batch = self.data[offset + 1 : offset + batch_length + 1].astype(np.int64)
            yield torch.from_numpy(x_batch).reshape(batch_size, context_length).to(device), torch.from_numpy(
                y_batch
            ).reshape(batch_size, context_length).to(device)


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(src: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> int:
    checkpoint = torch.load(src, "cpu")
    model_state_dict = dict()
    if list(checkpoint["model"].keys())[0].startswith("_orig_mod."):
        for k, v in checkpoint["model"].items():
            model_state_dict[k[len("_orig_mod.") :]] = v
    else:
        model_state_dict = checkpoint["model"]
    model.load_state_dict(model_state_dict)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]


def validation_loop(model, val_data, args) -> float:
    model.eval()
    losses = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for inputs, targets in val_data.get_validation_batches(args.batch_size, args.context_length, device):
            logits = model(inputs)
            loss = cross_entropy_loss(logits, targets)
            losses.append(loss)
    val_loss = torch.stack(losses).mean().item()
    model.train()
    return val_loss


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = NumpyDataset(args.train_data, seed=998244353)
    val_data = NumpyDataset(args.val_data)

    model = TransformerLM(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        context_length=args.context_length,
        dim=args.dim,
        num_heads=args.num_heads,
        dim_ff=args.dim_ff,
        attn_pdrop=args.attn_pdrop,
        residual_pdrop=args.residual_pdrop,
        epsilon=args.epsilon,
        activation=args.activation,
        norm_class=args.norm_class,
        norm_type=args.norm_type,
        parallel_block=args.parallel_block,
    )
    logging.info(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    total_steps = args.total_steps
    warmup_steps = args.warmup_steps

    args_dict = vars(args)
    run = wandb.init(
        project="cs336",
        config=args_dict,
        name=args.run_name,
    )

    logging.info("***** Running training *****")
    logging.info(f"  Total Steps = {total_steps}")
    logging.info(f"  Tokens per batch = {args.batch_size * args.context_length}")
    logging.info(f"  Total tokens = {total_steps * args.batch_size * args.context_length}")

    model = model.to(device).train()
    model = torch.compile(model)

    running_losses = []
    running_norms = []

    start = time.perf_counter()
    for step in tqdm(range(1, total_steps + 1), desc="Training"):
        inputs, targets = train_data.get_batch(args.batch_size, args.context_length, device)

        # Calculate loss
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            logits = model(inputs)
            loss = cross_entropy_loss(logits, targets)
        loss.backward()

        # Update parameters
        total_norm = clip_grad_norm(model.parameters(), args.max_grad_norm)
        optimizer.update_lr(get_cosine_schedule_lr(step, args.learning_rate, 0.0, warmup_steps, total_steps))
        optimizer.step()
        optimizer.zero_grad()

        # Logging
        running_losses.append(loss)
        running_norms.append(total_norm)

        if step % args.log_interval == 0:
            running_losses = torch.stack(running_losses)
            running_norms = torch.tensor(running_norms)

            learning_rate = get_cosine_schedule_lr(step, args.learning_rate, 0.0, warmup_steps, total_steps)
            torch.cuda.synchronize()
            end = time.perf_counter()
            throughput = args.log_interval * args.batch_size * args.context_length / (end - start)

            log_dict = {
                "train_loss": running_losses.mean().item(),
                "grad_norm": running_norms.mean().item(),
                "learning_rate": learning_rate,
                "throughput": throughput,
            }
            run.log(log_dict, step=step)
            logging.info(f"Step: {step}: {log_dict}")

            running_losses = []
            running_norms = []
            start = time.perf_counter()

        if step % args.val_interval == 0:
            val_loss = validation_loop(model, val_data, args)
            logging.info(f"Validation Loss at step {step}: {val_loss:.4f}")
            run.log({"val_loss": val_loss}, step=step)

    if args.save_path is not None:
        save_checkpoint(model, optimizer, total_steps, args.save_path)
        run.log_artifact(args.save_path)
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # I/O Parameters
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_path", type=str, default=None)

    # Model Parameters
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--dim_ff", type=int, default=2048)
    parser.add_argument("--attn_pdrop", type=float, default=0.1)
    parser.add_argument("--residual_pdrop", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--parallel_block", action="store_true")
    parser.add_argument("--norm_class", type=str, default="rms_norm", choices=["rms_norm", "layer_norm", "none"])
    parser.add_argument("--norm_type", type=str, default="pre_norm", choices=["pre_norm", "post_norm"])

    # Training Parameters
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--total_steps", type=int, default=5000)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--val_interval", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    args = parser.parse_args()
    train(args)
