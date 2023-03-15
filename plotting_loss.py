import numpy as np
import matplotlib.pyplot as plt
import os 

# data = np.load('traces1.npy')
# plt.plot(data[0])
# plt.show()

directory = "log"

names = ["neg_rank", "paraphrase", "sentiment","similarity"]
leg = []

for i, filename in enumerate(os.listdir(directory)):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        for i in range(len(names)):
            if filename.__contains__(names[i]):
                leg.append(names[i])
                data = np.load(f)
                print(data)
                plt.plot(data)

plt.legend(leg)
plt.show()
