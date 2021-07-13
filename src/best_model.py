import numpy as np
import os
import pickle


mean_scores = []
std_scores = []
filenames = []

for root, dirs, files in os.walk('./outputs'):
    for f in files:
        if f.split('.')[-1] == 'pkl' and f.split('.')[0][:5] == 'model' and 'feats' in f:
            scores = pickle.load(open(os.path.join(root, f), 'rb'))
            mean_scores.append(np.mean(scores))
            std_scores.append(np.std(scores))
            filenames.append(f)

print(mean_scores, std_scores)
print(np.argmin(mean_scores), filenames[np.argmin(mean_scores)])


