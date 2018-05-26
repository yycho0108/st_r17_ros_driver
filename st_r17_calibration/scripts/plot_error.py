import numpy as np
from matplotlib import pyplot as plt

e = np.loadtxt('/tmp/err.csv')
plt.plot(e)
plt.title('DH Parameter Error Over Time')
plt.grid()
plt.xlabel('Step')
plt.ylabel('Error')
plt.show()

