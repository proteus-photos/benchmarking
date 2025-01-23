import optuna
import subprocess
import sys
from tqdm import tqdm

class TqdmCallback:
    def __init__(self, n_trials):
        self.n_trials = n_trials
        self.pbar = tqdm(total=n_trials)

    def __call__(self, study, trial):
        self.pbar.update(1)

    def close(self):
        self.pbar.close()


def run_experiment(learning_rate, weight_decay, n_iter_range):
    """
    Run the experiment by calling an external script.

    Args:
        learning_rate (float): Learning rate for the model.
        weight_decay (float): Weight decay for the optimizer.

    Returns:
        float: The accuracy of the experiment (to be maximized).
    """

    command = [
        sys.executable,
        "./apgd_train.py",
        f"--lr={learning_rate}",
        f"--weight_decay={weight_decay}",
        f"--n_iter_range={n_iter_range}",
        "--image_dir=./diffusion_data",
        "--n_iter=10",
        "--n_epochs=1",
    ]

    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        output = result.stdout.strip().splitlines()

        accuracy = float(output[-1])
        print(accuracy)
        return accuracy
    except subprocess.CalledProcessError as e:
        print("Error while running experiment:", e.stderr)
        return 404
    except Exception as e:
        print("Unexpected error:", str(e))
        return 404

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-7, 1e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True)
    n_iter_range = trial.suggest_int("n_iter_range", 0, 5)

    accuracy = run_experiment(learning_rate, weight_decay, n_iter_range)
    return accuracy


n_trials = 120
study = optuna.create_study(direction="minimize")
tqdm_callback = TqdmCallback(n_trials)

study.optimize(objective, n_trials=n_trials, callbacks=[tqdm_callback])

tqdm_callback.close()

print("Best hyperparameters:", study.best_params)
print("Best accuracy:", study.best_value)

df = study.trials_dataframe()
df.to_csv("study_results.csv", index=False)

print("Study results saved to study_results.csv")