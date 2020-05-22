from uboost import GradientBooster, optimizers
import numpy as np
import matplotlib.pyplot as plt

X = np.random.normal(0, 1, (500, 1))
y = np.random.normal((X ** 2).sum(-1), 0.5)
X_test = np.random.normal(0, 1, (500, 1))
y_test = np.random.normal((X ** 2).sum(-1), 0.5)

gb = GradientBooster()
gb.learning_rate = 0.1
gb.n_learners = 100
gb.loss_optimizer = optimizers.MomentumOptimizer()
gb.fit(X, y)

x_grid = np.linspace(X.min() - X.std(), X.max() + X.std(), 1000).reshape(-1, 1)
pred = gb.predict(x_grid)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(gb.history["loss"])
plt.subplot(122)
plt.plot(x_grid, pred, color="red")
plt.scatter(X, y, alpha=0.2)
