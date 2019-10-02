import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.stats import norm

x = np.linspace(-4, 4, 1000)
y1 = np.zeros_like(x)
y2 = -13 * norm(0, 1).pdf(x)

fig, ax = plt.subplots(1, 1, figsize=(5, 3.7))
ax.fill_between(x, y1, y2, color='k')
ax.set(xlim=(-4, 4), ylim=(-6, 0))
ax.add_artist(Circle((1.5, -3), 1, color='w'))
ax.text(1.7, -3, 'i', va='center', ha='center', fontsize=28, style='italic',
        weight='semibold')
ax.text(2.8, -3, 'CCF', va='center', ha='center', fontsize=30, color='#CC3333',
        weight='bold')
ax.axis('off')

# fig.savefig('logo.png', transparent=True)
plt.show()
