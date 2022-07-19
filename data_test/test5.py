from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt



def make_data(N, f=0.3, rseed=1):
    rand = np.random.RandomState(rseed)
    x = rand.randn(N)
    # x[int(f * N):] += 5
    return x

x = make_data(500)

x_d = np.linspace(-8, 8, 500)

# instantiate and fit the KDE model
kde = KernelDensity(bandwidth=2, kernel='gaussian')
kde.fit(x[:, None])

# score_samples returns the log of the probability density
logprob = kde.score_samples(x_d[:, None])

prob = np.exp(logprob)
print(prob)


plt.fill_between(x_d, np.exp(logprob), alpha=0.5)
plt.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
plt.ylim(-0.02, 0.22)
plt.show()