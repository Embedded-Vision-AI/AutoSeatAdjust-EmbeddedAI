import json
import matplotlib.pyplot as plt

# Load the latest metrics run
with open("logs/training_metrics.json") as f:
    logs = json.load(f)
last = logs[-1]
targets = list(last["targets"].keys())

#learning curves
for target, m in last["targets"].items():
    plt.figure()
    plt.plot(m["train_rmse_curve"], label="Train RMSE")
    plt.plot(m["val_rmse_curve"],   label="Val   RMSE")
    plt.title(f"Learning Curve – {target}")
    plt.xlabel("Boosting Round")
    plt.ylabel("RMSE")
    plt.legend()

#training time per target
train_times = [last["targets"][t]["train_time_s"] for t in targets]
plt.figure()
plt.bar(targets, train_times)
plt.title("Training Time per Target")
plt.ylabel("Time (s)")

#Final RMSE (train vs. validation)
train_rmse_final = [last["targets"][t]["train_rmse_final"] for t in targets]
val_rmse_final   = [last["targets"][t]["val_rmse_final"]   for t in targets]
x = range(len(targets))
plt.figure()
plt.bar(x, train_rmse_final, width=0.4, label="Train RMSE Final")
plt.bar([i+0.4 for i in x], val_rmse_final, width=0.4, label="Val RMSE Final")
plt.xticks([i+0.2 for i in x], targets)
plt.title("Final RMSE Comparison")
plt.ylabel("RMSE")
plt.legend()

#cross-validation RMSE with error bars
cv_means = [last["targets"][t]["cv_rmse_mean"] for t in targets]
cv_stds  = [last["targets"][t]["cv_rmse_std"]  for t in targets]
plt.figure()
plt.bar(targets, cv_means, yerr=cv_stds, capsize=5)
plt.title("Cross-Validation RMSE")
plt.ylabel("RMSE")

#validation error metrics: RMSE, variance, max error
val_rmse       = [last["targets"][t]["val_rmse"] for t in targets]
resid_variance = [last["targets"][t]["residual_variance"] for t in targets]
max_error      = [last["targets"][t]["max_error"] for t in targets]
plt.figure()
x = range(len(targets))
plt.bar(x, val_rmse,       width=0.3, label="Val RMSE")
plt.bar([i+0.3 for i in x], resid_variance, width=0.3, label="Residual Variance")
plt.bar([i+0.6 for i in x], max_error,      width=0.3, label="Max Error")
plt.xticks([i+0.3 for i in x], targets)
plt.title("Additional Validation Metrics")
plt.legend()

plt.tight_layout()
plt.show()
