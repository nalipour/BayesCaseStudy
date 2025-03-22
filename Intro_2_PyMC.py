#%%
import arviz as az
import pandas as pd
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

# %%
print(f"Using PyMC version: {pm.__version__}")

# %%

rng = np.random.default_rng(42)
data = rng.standard_normal(500)
print(f"Observed data with mean {np.mean(data)} and sd {np.std(data)}")
# %%
with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=1)
    obs = pm.Normal("obs", mu=mu, sigma=1, observed=data)
# %%
model.basic_RVs

# %%
model.free_RVs

# %%
model.observed_RVs

# %%
with pm.Model() as model:
    x = pm.Normal("x", mu=0, sigma=1)
    y = pm.Gamma("y", alpha=1, beta=1)
    plus_2 = x + 2
    summed = x + y
    squared = x**2
# %%
with pm.Model():
    x = pm.Normal("x", mu=0, sigma=1)
    plus_2 = pm.Deterministic("x plus 2", x + 2)
# %%
coords = {"cities": ["Santiago", "Mumbai", "Tokyo"]}
with pm.Model(coords=coords) as model:
    x = pm.Normal("x", mu=0, sigma=1, dims="cities")
    y = x[0] * x[1]
# %%
with pm.Model() as model:
    mu = pm.Normal("mu", mu=1, sigma=1)
    sd = pm.HalfNormal("sd", sigma=1)
    obs = pm.Normal("obs", mu=mu, sigma=sd, observed=data)

    posterior_samples = pm.sample(random_seed=rng)
    prior_samples = pm.sample_prior_predictive(samples=1000, random_seed=rng)

# %%
for var in ["mu", "sd"]:
    az.plot_dist(prior_samples.prior[var])
    plt.title(f"Prior for {var}")
    plt.show()
# %%
plt.figure(figsize=(12, 6))
plt.hist(
    np.array(prior_samples.prior["mu"][0, :]),
    bins=50,
    color="cornflowerblue",
    alpha=0.7,
    edgecolor="white",
    density=True,
)

plt.grid(True, alpha=0.3)
plt.title("Distribution of Prior Samples (μ)", fontsize=14, pad=15)
plt.xlabel("μ Values", fontsize=12)
plt.ylabel("Density", fontsize=12)

# Add mean line and annotation
mean_mu = np.mean(prior_samples.prior["mu"][0, :])
plt.axvline(mean_mu, color="red", linestyle="--", alpha=0.5)
plt.text(mean_mu * 1.1, plt.gca().get_ylim()[1] * 0.9, f"Mean: {mean_mu:.2f}", fontsize=10)

plt.tight_layout()
plt.show()
# %%
plt.figure(figsize=(12, 6))

plt.hist(
    np.array(prior_samples.prior["sd"][0, :]),
    bins=50,
    color="#3498db",
    alpha=0.7,
    edgecolor="white",
    density=True,
    label="σ",
)

mean_sd = np.mean(np.array(prior_samples.prior["sd"][0, :]))
plt.axvline(mean_sd, color="#e74c3c", linestyle="--", label=f"Mean: {mean_sd:.2f}", alpha=0.8)
plt.text(mean_sd * 1.1, mean_sd * 0.9, f"Mean: {mean_sd:.2f}", fontsize=10)

plt.grid(True, alpha=0.3)
plt.title("Distribution of Prior Samples (σ)", fontsize=14, pad=15)
plt.xlabel("σ Values", fontsize=12)
plt.ylabel("Density", fontsize=12)

plt.tight_layout()
plt.show()
# %%
az.plot_trace(posterior_samples);

# %%
az.summary(posterior_samples)

# %%
az.plot_posterior(posterior_samples);

# %% Prediction :
"""
Once the posterior distribution has been estimated, 
it can be used to compute predictions on new samples.
"""

with model:
    posterior_samples.extend(pm.sample_posterior_predictive(posterior_samples, random_seed=rng))

# %%
fig, ax = plt.subplots()
az.plot_ppc(posterior_samples, ax=ax)
ax.axvline(data.mean(), ls="--", color="r", label="True mean")
ax.legend(fontsize=10);

# %% Linear Regression

# True parameter values
alpha, sigma = 2, 1
beta = 3

# Size of dataset
train_size = 200

# Predictor variable
x = rng.normal(size=train_size)

# Simulate outcome variable
y = alpha + beta * x + rng.normal(size=train_size) * sigma
# %% we visualize the relationship between the variables.
plt.scatter(x, y, alpha=0.6)
plt.title("Scatter plot of x vs y", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()
# %%
with pm.Model() as linear_model:
    # Priors for unknown model parameters
    alpha_rv = pm.Normal("alpha", mu=0, sigma=10)
    beta_rv = pm.Normal("beta", mu=0, sigma=10)
    sigma_rv = pm.HalfNormal("sigma", sigma=1)

    x_data = pm.Data("x", x.tolist())

    # Expected value of outcome
    mu = alpha_rv + beta_rv * x_data

    # Likelihood (sampling distribution) of observations
    y_obs = pm.Normal("y", mu=mu, sigma=sigma_rv, observed=y, shape=x_data.shape)

pm.model_to_graphviz(linear_model)
# %%
with linear_model:
    posterior_samples = pm.sample(return_inferecedata=True, random_seed=rng)

posterior_samples
# %%
az.plot_trace(posterior_samples, combined=True)
plt.tight_layout()
plt.show()
# %%
az.summary(posterior_samples)

# %%
test_size = 50

test_x = rng.normal(size=test_size)

with linear_model:
    pm.set_data({"x": test_x.tolist()})
    prediction_trace = pm.sample_posterior_predictive(
        posterior_samples,
        var_names=["y"],
        return_inferencedata=True,
        predictions=True,
        extend_inferencedata=True,
        random_seed=rng,
    )
    predictions_mean = prediction_trace["predictions"]["y"][1].mean(axis=0)
    prediction_intervals = az.hdi(ary=prediction_trace["predictions"], hdi_prob=0.95)["y"]
    prediction_df = pd.DataFrame(
        {
            "x": test_x,
            "y": alpha + beta * test_x + rng.normal(size=test_size) * sigma,
            "prediction_mean": predictions_mean,
            "prediction_hdi_5": prediction_intervals[:, 0],
            "prediction_hdi_95": prediction_intervals[:, 1],
        }
    )

prediction_df.head()
# %%
prediction_df = prediction_df.sort_values("x")

plt.figure()
plt.scatter(prediction_df["x"], prediction_df["y"], label="Observed y")
plt.plot(
    prediction_df["x"],
    prediction_df["prediction_mean"],
    label="Prediction mean",
    color="blue",
)
plt.fill_between(
    prediction_df["x"],
    prediction_df["prediction_hdi_5"],
    prediction_df["prediction_hdi_95"],
    color="red",
    alpha=0.2,
    label="95% HDI",
)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
# %%
