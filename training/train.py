"""
Fine-tune a pretrained ConvNeXt-Tiny to recognize car make/model/year-range.

THE BIG PICTURE
---------------
Training is a loop of guess -> check -> adjust:

  1. Show the model a BATCH of 128 training photos.
  2. It outputs 1,168 scores per photo (one per class). These raw scores are
     called LOGITS; higher = "more likely this class".
  3. Measure how wrong it was with a LOSS function (one number: 0 = perfect,
     bigger = worse).
  4. BACKPROPAGATION works out, for every one of the model's ~28 million
     weights, which direction would have made the loss smaller.
  5. The OPTIMIZER nudges every weight a small step in that direction.
     The size of the step is the LEARNING RATE.
  6. Repeat. One pass over all 194K training photos is one EPOCH
     (~1,500 batches). We run 20 epochs.

After each epoch we check accuracy on the VAL pile (photos the model never
learns from). If val accuracy keeps going up, the model is learning real
patterns. If train accuracy keeps rising but val accuracy stalls or drops,
the model is memorizing the training photos instead ("overfitting").

Where this fits:
    build_subset.py -> [train.py] -> evaluate.py / predict.py -> export_onnx.py -> Java app

Outputs:
    training/checkpoints/best.pt   the model from the epoch with the best val accuracy
                                   (this is the one evaluate/predict/export use)
    training/checkpoints/last.pt   everything needed to resume after a crash or Ctrl+C
    training/runs/train/log.csv    one row of metrics per epoch
    training/runs/train/curves.png loss and accuracy charts

Usage (from the project root):
    python training/train.py            # full training run (~1-2 hours on an RTX 4090)
    python training/train.py --quick    # ~2-minute smoke test on a few batches
    python training/train.py --resume   # continue from training/checkpoints/last.pt
"""

import argparse
import math
import time

import matplotlib

matplotlib.use("Agg")  # draw charts straight to files, no pop-up window
import matplotlib.pyplot as plt
import pandas as pd
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from common import load_config, project_path
from dataset import CarDataset, build_transforms, load_class_names


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------
def parse_args():
    """Read command-line flags (like parsing String[] args in Java's main)."""
    parser = argparse.ArgumentParser(description="Fine-tune ConvNeXt-Tiny on the VMMRdb subset.")
    parser.add_argument("--quick", action="store_true",
                        help="smoke test: 1 short epoch on a few batches, saved separately")
    parser.add_argument("--resume", action="store_true",
                        help="continue training from last.pt")
    return parser.parse_args()


def build_model(model_name, num_classes, drop_path_rate):
    """Download the pretrained model and replace its last layer for our classes.

    THIS IS THE TRANSFER LEARNING STEP. pretrained=True loads weights that already
    recognize edges, shapes, wheels, windows, etc. num_classes tells timm to
    throw away the original 1,000-class ImageNet output layer and attach a fresh
    one with one output per class. Only that new layer starts from random numbers;
    everything else starts from what ImageNet already taught it.

    drop_path_rate (also called "stochastic depth"): during training, each of the
    model's residual blocks is randomly skipped for some images, up to this chance for
    the deepest blocks. The model can't lean on any single block, which makes it
    harder to memorize training photos. It only acts during training; model.eval()
    turns it off, so evaluation, export and the Java app are unaffected.

    The weights come from Hugging Face's official timm/ organization and are
    cached in ~/.cache/huggingface after the first download.
    """
    return timm.create_model(model_name, pretrained=True, num_classes=num_classes,
                             drop_path_rate=drop_path_rate)


def make_loaders(config, class_names, mean, std, quick):
    """Create the DataLoaders that feed batches of images to the GPU.

    A DataLoader wraps a Dataset and handles: grouping items into batches,
    shuffling the order every epoch, and loading in parallel worker processes,
    so the CPU prepares the next batch while the GPU works on the current one.
    """
    paths, settings = config["paths"], config["training"]
    splits_dir = project_path(paths["splits_dir"])
    dataset_dir = project_path(paths["dataset_dir"])
    size = settings["image_size"]

    train_set = CarDataset(splits_dir / "train.csv", dataset_dir, class_names,
                           build_transforms(size, mean, std, train=True))
    val_set = CarDataset(splits_dir / "val.csv", dataset_dir, class_names,
                         build_transforms(size, mean, std, train=False))

    if quick:
        # Smoke test: just enough data to prove the whole pipeline runs end to end.
        # Subset(dataset, indices) is a view that only exposes those items.
        generator = torch.Generator().manual_seed(0)
        train_set = Subset(train_set, torch.randperm(len(train_set), generator=generator)[:50 * settings["batch_size"]].tolist())
        val_set = Subset(val_set, torch.randperm(len(val_set), generator=generator)[:10 * settings["batch_size"]].tolist())

    common_args = dict(
        batch_size=settings["batch_size"],
        num_workers=settings["num_workers"],
        # pin_memory puts batches in page-locked RAM, which makes copying them to the GPU faster.
        pin_memory=torch.cuda.is_available(),
        # Keep worker processes alive between epochs. On Windows, starting them is slow
        # (each one re-imports this whole file), so reusing them saves time every epoch.
        persistent_workers=settings["num_workers"] > 0,
    )
    # shuffle=True for training: a different order every epoch, so the model can't
    # learn anything from the order itself. Validation order doesn't matter.
    # drop_last=True skips a final, smaller batch, which keeps every step the same size.
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **common_args)
    val_loader = DataLoader(val_set, shuffle=False, **common_args)
    return train_loader, val_loader


def make_lr_schedule(optimizer, warmup_steps, total_steps):
    """Learning rate that ramps up linearly, then decays smoothly to ~0 ("warmup + cosine").

    WHY WARMUP: at the very start, the new last layer is random, so its error
    signals are large and chaotic. Big steps then could wreck the carefully
    pretrained weights underneath. Starting tiny and ramping up avoids that.

    WHY DECAY: early on, big steps make fast progress. Near the end, small steps
    let the model settle into a good spot instead of bouncing around it.

    LambdaLR multiplies the configured learning rate by whatever this function
    returns for the current step (0.0 to 1.0):

        1.0 |     ____
            |    /    ‾‾--__
            |   /           ‾-_
        0.0 |__/________________‾-__
             warmup     cosine decay
    """
    def lr_multiplier(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)  # 0.0 -> 1.0
        return 0.5 * (1 + math.cos(math.pi * progress))  # 1.0 -> 0.0 along a cosine curve

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)


# ---------------------------------------------------------------------------
# One epoch of training, and one pass of validation
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, device, use_amp, epoch_label):
    """Run guess -> check -> adjust over every training batch once. Returns (avg loss, accuracy)."""
    model.train()  # turns on training-only behavior inside the model (e.g. dropout)
    total_loss, total_correct, total_seen = 0.0, 0, 0

    progress = tqdm(loader, desc=epoch_label, unit="batch", leave=False)
    for images, labels in progress:
        # Move the batch from CPU memory to the GPU. non_blocking lets the copy
        # overlap with other work (it works together with pin_memory above).
        # channels_last is a memory layout that's faster for convolutions on NVIDIA GPUs.
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)

        # MIXED PRECISION: inside this block, most math runs in bfloat16 (16-bit
        # numbers) instead of float32. The 4090's tensor cores are much faster at
        # 16-bit, and it halves memory use. bfloat16 keeps float32's range, so unlike
        # float16 it doesn't need any extra "loss scaling" tricks.
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(images)            # 1. guess: shape [128, 1168]
            loss = loss_fn(logits, labels)    # 2. check: one number for the whole batch

        optimizer.zero_grad(set_to_none=True)  # clear the previous step's gradients
        loss.backward()    # 3. backpropagation: compute the gradient for every weight
        optimizer.step()   # 4. adjust: nudge every weight against its gradient
        scheduler.step()   # move the learning rate one step along its schedule

        # Bookkeeping for the progress bar and the log.
        # .item() copies a single number from the GPU back to a normal Python float.
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()  # argmax = the top-scoring class
        total_seen += batch_size
        progress.set_postfix(loss=f"{total_loss / total_seen:.3f}", acc=f"{total_correct / total_seen:.1%}")

    return total_loss / total_seen, total_correct / total_seen


@torch.no_grad()  # decorator: don't track gradients in here (we're not learning), saving memory and time
def validate(model, loader, loss_fn, device, use_amp):
    """Measure loss, top-1 and top-5 accuracy on the val pile. No learning happens here.

    TOP-1 accuracy: the model's single best guess is correct.
    TOP-5 accuracy: the correct class is somewhere in its 5 best guesses.
    Top-5 is useful for us because the app shows 5 suggestions: if the right
    answer is in the list, the app is still helpful.
    """
    model.eval()  # turns off training-only behavior, so results are deterministic
    total_loss, top1, top5, total_seen = 0.0, 0, 0, 0

    for images, labels in tqdm(loader, desc="  validating", unit="batch", leave=False):
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, labels)

        # topk(5) returns the indices of the 5 highest scores for each image: shape [batch, 5].
        best5 = logits.topk(5, dim=1).indices
        # labels.unsqueeze(1) turns shape [batch] into [batch, 1] so it can be compared
        # against all 5 guesses at once (broadcasting). The result is [batch, 5] True/False.
        hits = best5 == labels.unsqueeze(1)
        top1 += hits[:, 0].sum().item()        # column 0 = the single best guess
        top5 += hits.any(dim=1).sum().item()   # correct anywhere in the 5

        total_loss += loss.item() * labels.size(0)
        total_seen += labels.size(0)

    return total_loss / total_seen, top1 / total_seen, top5 / total_seen


# ---------------------------------------------------------------------------
# Saving progress
# ---------------------------------------------------------------------------
def save_best_model(path, model, class_names, settings, mean, std, epoch, val_top1):
    """Save everything needed to USE the model later (evaluate, predict, ONNX export).

    We save the weights (state_dict) plus the facts needed to rebuild the model
    and prepare images identically: class order, model name, image size,
    normalization mean/std. A weights file without those facts is useless.
    """
    torch.save({
        "model_state": model.state_dict(),   # every weight, by name
        "class_names": class_names,          # index -> class name, in order
        "model_name": settings["model_name"],
        "image_size": settings["image_size"],
        "mean": list(mean),
        "std": list(std),
        "epoch": epoch,
        "val_top1": val_top1,
    }, path)


def save_resume_point(path, model, optimizer, scheduler, epoch, best_top1, history):
    """Save everything needed to CONTINUE training exactly where it stopped.

    Besides the weights, the optimizer has its own memory (running averages of
    past gradients), and the scheduler knows which step it's on. Without them,
    a resumed run would behave differently from an uninterrupted one.
    """
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "best_top1": best_top1,
        "history": history,
    }, path)


def plot_curves(history, out_path):
    """Two charts, train vs val: loss (should go down) and accuracy (should go up)."""
    df = pd.DataFrame(history)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    train_color, val_color = "#2a78d6", "#eb6834"  # blue = train, orange = val

    ax_loss.plot(df["epoch"], df["train_loss"], color=train_color, linewidth=2, label="train")
    ax_loss.plot(df["epoch"], df["val_loss"], color=val_color, linewidth=2, label="val")
    ax_loss.set_title("Loss (lower is better)", loc="left")

    ax_acc.plot(df["epoch"], df["train_acc"], color=train_color, linewidth=2, label="train top-1")
    ax_acc.plot(df["epoch"], df["val_top1"], color=val_color, linewidth=2, label="val top-1")
    ax_acc.plot(df["epoch"], df["val_top5"], color=val_color, linewidth=2, linestyle="--", label="val top-5")
    ax_acc.set_title("Accuracy (higher is better)", loc="left")
    ax_acc.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

    for ax in (ax_loss, ax_acc):
        ax.set_xlabel("Epoch", color="#52514e")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))  # whole-number epochs only
        ax.grid(color="#e1e0d9", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors="#898781")
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    config = load_config()
    settings = config["training"]
    torch.manual_seed(settings["seed"])

    # Quick runs go to their own folders so they never overwrite a real model.
    run_name = "train_quick" if args.quick else "train"
    checkpoints_dir = project_path(config["paths"]["checkpoints_dir"]) / ("quick" if args.quick else "")
    runs_dir = project_path(config["paths"]["runs_dir"]) / run_name
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Use the GPU if there is one. Mixed precision only helps on the GPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if use_amp else " - training on CPU will be very slow"))

    class_names = load_class_names(project_path(config["paths"]["splits_dir"]))
    model = build_model(settings["model_name"], num_classes=len(class_names),
                        drop_path_rate=settings["drop_path_rate"])
    model = model.to(device, memory_format=torch.channels_last)

    # Read the normalization numbers from the pretrained model's own config, so they
    # always match the weights we downloaded (rather than typing them in by hand).
    mean, std = model.pretrained_cfg["mean"], model.pretrained_cfg["std"]
    print(f"Model: {settings['model_name']} ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M weights), "
          f"{len(class_names):,} classes")

    train_loader, val_loader = make_loaders(config, class_names, mean, std, args.quick)
    epochs = 1 if args.quick else settings["epochs"]
    print(f"Train: {len(train_loader.dataset):,} images ({len(train_loader):,} batches/epoch), "
          f"val: {len(val_loader.dataset):,} images, epochs: {epochs}")

    # LOSS: cross-entropy is the standard loss for "pick one class out of many".
    # It's small when the model gives the correct class a high probability.
    # LABEL SMOOTHING (0.1) changes the target from "100% this class" to
    # "90% this class, 10% spread over the rest". With noisy labels like
    # VMMRdb's (some photos are simply mislabeled), this stops the model from
    # becoming overconfident in answers that might be wrong.
    loss_fn = nn.CrossEntropyLoss(label_smoothing=settings["label_smoothing"])

    # OPTIMIZER: AdamW is the standard choice for fine-tuning modern models.
    # It adapts the step size for each weight individually, and its "weight decay"
    # slightly shrinks all weights every step, which discourages memorization.
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    warmup_steps = (0 if args.quick else settings["warmup_epochs"]) * len(train_loader)
    scheduler = make_lr_schedule(optimizer, warmup_steps, total_steps=epochs * len(train_loader))

    # Resume support: reload the weights, optimizer, scheduler, and metrics so far.
    start_epoch, best_top1, history = 1, 0.0, []
    if args.resume:
        # weights_only=True only allows plain data (tensors, numbers, strings, lists, dicts)
        # to be loaded - never arbitrary code. Always load checkpoints this way.
        state = torch.load(checkpoints_dir / "last.pt", map_location=device, weights_only=True)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        start_epoch, best_top1, history = state["epoch"] + 1, state["best_top1"], state["history"]
        print(f"Resuming from epoch {start_epoch} (best val top-1 so far: {best_top1:.1%})")

    for epoch in range(start_epoch, epochs + 1):
        started = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler,
                                                device, use_amp, f"Epoch {epoch}/{epochs}")
        val_loss, val_top1, val_top5 = validate(model, val_loader, loss_fn, device, use_amp)
        seconds = time.time() - started

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_top1": val_top1, "val_top5": val_top5,
            "lr": optimizer.param_groups[0]["lr"], "seconds": round(seconds),
        })

        # Keep the best model by VAL accuracy, not the last one: if later epochs start
        # overfitting, we still have the version that did best on unseen photos.
        improved = val_top1 > best_top1
        if improved:
            best_top1 = val_top1
            save_best_model(checkpoints_dir / "best.pt", model, class_names, settings, mean, std, epoch, val_top1)
        save_resume_point(checkpoints_dir / "last.pt", model, optimizer, scheduler, epoch, best_top1, history)

        # Rewrite the log and chart after every epoch, so you can watch progress mid-run.
        pd.DataFrame(history).to_csv(runs_dir / "log.csv", index=False)
        plot_curves(history, runs_dir / "curves.png")

        images_per_sec = len(train_loader) * settings["batch_size"] / seconds
        remaining = (epochs - epoch) * seconds / 60
        print(f"Epoch {epoch:>2}/{epochs} | train loss {train_loss:.3f} acc {train_acc:.1%} | "
              f"val loss {val_loss:.3f} top-1 {val_top1:.1%} top-5 {val_top5:.1%} | "
              f"{seconds / 60:.1f} min, {images_per_sec:,.0f} img/s, ~{remaining:.0f} min left"
              + ("  * best" if improved else ""))

    print(f"\nDone. Best val top-1: {best_top1:.1%}")
    print(f"Best model: {checkpoints_dir / 'best.pt'}")
    print(f"Log + charts: {runs_dir}")


# The __main__ guard is REQUIRED here on Windows, not just good style: each DataLoader
# worker process re-imports this file, and without the guard every worker would
# start its own training run.
if __name__ == "__main__":
    main()
