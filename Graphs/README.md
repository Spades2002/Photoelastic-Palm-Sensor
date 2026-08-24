THESIS PLOTTING SCRIPTS
=======================

These scripts are configured for:

    D:\ERP\Graphs

Place the following files in that folder with these exact filenames:

    force_conditioned_predictions.xlsx
    joint_multitask_predictions.xlsx
    mlp_predictions.csv
    two_stage_predictions.csv

Place these Python files in the same folder as well:

    plot_style.py
    01_fz_predicted_vs_ground_truth.py
    02_fz_mae_architecture_comparison.py
    03_contact_localisation_2d.py
    04_two_stage_depth_prediction.py
    05_fz_absolute_error_vs_depth.py
    06_fz_absolute_error_distributions.py
    run_all_plots.py

The scripts create:

    D:\ERP\Graphs\figures

and save every figure as both PNG (300 dpi) and PDF.

Install dependencies with:

    pip install pandas matplotlib numpy scikit-learn openpyxl

Run everything with:

    python run_all_plots.py

Or run any numbered script individually.

IMPORTANT MLP NOTE
------------------
The folder shown contains mlp_predictions.csv, which is a summary-metrics file rather than a full per-sample prediction file. The MLP is therefore included in the architecture-level Fz MAE comparison, but not in predicted-vs-ground-truth, error-vs-depth, or error-distribution plots. If you later add random_mlp_sample_predictions.csv, those plots can be extended to include the MLP sample as well.
