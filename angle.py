import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns

csv_dir = './data/driving_log.csv'

angles = []
with open(csv_dir) as f:
    reader = csv.reader(f)
    for line in list(reader)[1:]:
        angle = float(line[3])
        angles.append(angle)

x = np.array(angles)

sns.distplot(x)
plt.show()
