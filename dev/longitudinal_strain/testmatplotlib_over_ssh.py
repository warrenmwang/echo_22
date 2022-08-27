import matplotlib.pyplot as plt
import numpy as np

plt.figure()
plt.plot(np.arange(-1, 1, 0.25), np.arange(-1, 1, 0.25))
plt.savefig("./test.png")