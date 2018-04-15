import numpy as np
import matplotlib.pyplot as plt

grayhounds = 500
labs = 500

gray_height = 28 + 4 * np.random.randn(grayhounds)
lab_height = 24 + 4 * np.random.randn(labs)

plt.hist([gray_height, lab_height],  stacked=True, color=['r', 'b'])
plt.show()

