import numpy as np
import pandas as pd

# randomly sample 1000 numbers from a normal distribution
x = np.random.normal(loc=1.25, scale=0.12, size=10000)
y = np.random.normal(loc=0.4375, scale=0.12, size=10000)
print(np.mean(x), np.var(x))
print(np.mean(y), np.var(y))

# randomly multiple 10% of the numbers by -1
x[np.random.choice(10000, 4000, replace=False)] *= -1
y[np.random.choice(10000, 1000, replace=False)] *= -1

# combine the two arrays
combined = np.concatenate((x, y))
cumulative_mean = np.mean(combined)
print(len(x[x > cumulative_mean]), len(y[y > cumulative_mean]))
print(np.mean(x), np.var(x))
print(np.mean(y), np.var(y))

# simulate a binary choice
y = np.random.choice([0, 1], 10000, p=[0.25, 0.75])
print(np.mean(y), np.std(y))


def binomial_var(p1, p2):
    if p1 < 0 or p1 > 1 or p2 < 0 or p2 > 1:
        raise ValueError('p1 and p2 must be between 0 and 1')
    if p1 + p2 != 1:
        raise ValueError('p1 and p2 must sum to 1')
    return np.sqrt(p1 * p2)

print(binomial_var(0.725, 0.75))

# calculate the variance of the original Iowa Gambling Task
deckA = [100, 100, -50, 100, -200, 100, -100, 100, -150, -250]
deckB = [100, 100, 100, 100, 100, 100, 100, 100, 100, -1150]
deckC = [50, 50, 0, 50, 0, 50, 0, 0, 0, 50]
deckD = [50, 50, 50, 50, 50, 50, 50, 50, 50, -200]
print(f'Deck A: {np.std(deckA)}')
print(f'Deck B: {np.std(deckB)}')
print(f'Deck C: {np.std(deckC)}')
print(f'Deck D: {np.std(deckD)}')