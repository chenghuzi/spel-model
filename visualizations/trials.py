"""Trial diagnostics shared by training and evaluation."""

import matplotlib.pyplot as plt


def plot_trial(hiddens, outputs, inputs, lengths, path, title, actions=None):
    # Display the same first-trial signals for both training and evaluation.
    stop = lengths[0]
    plt.close("all")
    fig, axes = plt.subplots(
        3 if actions is None else 4, 1, sharex=True, figsize=(8, 6 if actions is None else 8)
    )
    axes[0].plot(hiddens[0].detach().cpu().numpy()[:stop])
    axes[0].set_title("RNN hidden")
    axes[1].imshow(
        outputs[0].detach().cpu().numpy()[:stop].T,
        aspect="auto",
        interpolation="none",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("RNN output")
    if actions is not None:
        axes[2].imshow(
            actions[0].detach().cpu().numpy()[:stop].T,
            aspect="auto",
            interpolation="none",
            vmin=0,
            vmax=1,
        )
        axes[2].set_title("target")
    values = inputs[0].detach().cpu().numpy()[:stop].T
    axes[-1].imshow(values, aspect="auto", interpolation="none")
    axes[-1].set_title(f"RNN input ({values.min():.2f} to {values.max():.2f})")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
