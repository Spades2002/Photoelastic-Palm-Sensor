"""MLP (deep neural network), run any time after build_features.py has produced its cache."""
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import CONFIG, run_model

model = make_pipeline(
    StandardScaler(),
    MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=2000,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=CONFIG["seed"],
        verbose=True,  # prints training loss every iteration as it fits
    ),
)
run_model("mlp", model)
