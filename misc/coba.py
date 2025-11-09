import matplotlib.pyplot as plt
import numpy as np

cos = np.ones(100)
for i in range(1, 100):
    cos[i] = np.cos(2*np.pi * i / 100)
cos_noisy = cos + np.random.normal(0, 0.25, 100)

def momentum(data, beta=0.9):
    v = 0
    smoothed = np.zeros_like(data)
    for i in range(len(data)):
        v = beta * v + (1 - beta) * data[i]
        smoothed[i] = v
    return smoothed

smoothed_cos = momentum(cos_noisy, beta=0.9)

# plot original and noisy signals
plt.figure(figsize=(10, 5))
plt.plot(cos, label='Original Cosine', color='blue')
plt.plot(cos_noisy, label='Noisy Cosine', color='orange', alpha=0.7)
plt.plot(smoothed_cos, label='Smoothed Cosine (Momentum)', color='green')
plt.title('Original and Noisy Cosine Signals')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show()