import pandas as pd
import matplotlib.pyplot as plt
import os
from src.fea_gnn.utils import load_config


def plot_training_curves():
    cfg = load_config()
    csv_path = os.path.join(cfg["env"]["save_path"], "training_history.csv")

    if not os.path.exists(csv_path):
        print(f"Erreur: Pas d'historique trouvé dans {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Configuration esthétique
    plt.style.use("ggplot")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Courbes de Loss (Log scale pour mieux voir la convergence fine)
    ax1.plot(
        df["epoch"], df["total_loss"], label="Total Loss", color="black", linewidth=2
    )
    ax1.plot(df["epoch"], df["data_loss"], label="Data (MSE)", linestyle="--")
    ax1.plot(df["epoch"], df["phys_loss"], label="Physics (PINN)", linestyle=":")

    ax1.set_yscale("log")  # Très important pour voir les erreurs de 10^-4 à 10^-6
    ax1.set_title("Convergence de l'Entraînement")
    ax1.set_xlabel("Époque")
    ax1.set_ylabel("Loss (Log Scale)")
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.5)

    # 2. Learning Rate
    ax2.plot(df["epoch"], df["lr"], color="orange")
    ax2.set_title("Adaptation du Learning Rate")
    ax2.set_xlabel("Époque")
    ax2.set_ylabel("Learning Rate")

    output_path = "visualization/training_metrics.png"
    plt.savefig(output_path, dpi=300)
    print(f"Courbes sauvegardées dans {output_path}")
    plt.show()


if __name__ == "__main__":
    plot_training_curves()
