import numpy as np
import matplotlib.pyplot as plt
import os 

# data = np.load('traces1.npy')
# plt.plot(data[0])
# plt.show()

directory = "log"
experiments = ["03_14_2023","03_15_2023"]
titles = ["Negative Rank", "Negative Rank with Shared Layer and Scaled Loss"]

names = ["neg_rank", "paraphrase", "sentiment","similarity"]
leg = []
for j, experiment in enumerate(experiments):
    for i, filename in enumerate(os.listdir(directory)):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            for i in range(len(names)):
                if filename.__contains__(names[i]) and filename.__contains__(experiment):
                    leg.append(names[i])
                    data = np.load(f)
                    plt.plot(data)
    
    plt.title(titles[j])
    plt.legend(leg)
    plt.savefig(titles[j]+".jpg")
    plt.figure()
