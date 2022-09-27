import numpy as np

x = np.linspace(0, 1, 100)
fx1 = x
fx2 = 2 * x / (x+1)

import tensor_operations.plot as o4plot

o4plot.lines(ys=[fx1, fx2], xs=[x, x], labels=["IoU", "F1"], xlabel="True Positives [%]")





