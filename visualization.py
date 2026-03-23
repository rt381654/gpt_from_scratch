"""
visualization.py — Loss Curve Plotting
========================================
Plots training and validation loss curves collected during the training loop
and saves the figure to disk.

Keeping visualization here means train.py stays focused on the training loop
and this module can be extended (e.g. learning-rate schedule plots) without
touching training code.
"""

import matplotlib.pyplot as plt  # standard Python plotting library


def plot_loss_curves(
    train_losses: list,   # list of (step, loss) tuples from training checkpoints
    val_losses:   list,   # list of (step, loss) tuples from validation checkpoints
    save_path:    str = "loss_curve.png",  # file path for the saved figure
):
    """
    Plot train and val loss against training step and save to a PNG file.

    Args:
        train_losses: List of (step, train_loss) pairs recorded at eval checkpoints.
        val_losses:   List of (step, val_loss)   pairs recorded at eval checkpoints.
        save_path:    Destination path for the saved PNG image.
    """
    # Unzip the (step, loss) pairs into separate lists for the x and y axes.
    steps_recorded, train_loss_values = zip(*train_losses)
    _,              val_loss_values   = zip(*val_losses)

    plt.figure()                                          # create a new figure
    plt.plot(steps_recorded, train_loss_values, label="train loss")  # blue line by default
    plt.plot(steps_recorded, val_loss_values,   label="val loss")    # orange line by default
    plt.xlabel("Step")                                    # label the x-axis
    plt.ylabel("Loss")                                    # label the y-axis
    plt.title("Training and Validation Loss")             # chart title
    plt.legend()                                          # show the label legend
    plt.tight_layout()                                    # avoid clipping labels
    plt.savefig(save_path)                                # save to disk instead of blocking on plt.show()
    print(f"\nLoss curve saved to {save_path}")
