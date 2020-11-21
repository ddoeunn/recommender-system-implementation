import numpy as np
from LogisticMF.Model import LogisticMF
from utils.dataset import movielens
from utils.dataset.split import Split
from utils.evaluation.ranking import MPR
from utils.common.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL
)
HEADER = [DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL]
TRAIN_SIZE = 0.9

# Load Dataset
# path= "C:\\Users\\DoeunKim\\Desktop"
# data = movielens.load_data(header=HEADER, local_cache_path=path, unzip_path=path)
data = movielens.load_data(header=HEADER)

# Split Dataset
split = Split(data, train_size=TRAIN_SIZE)
train, test = split.train_test_split()

# Set Hyper Parameters
alpha = np.sum(train == 0) / data[DEFAULT_RATING_COL].sum()
f = 15          # number of latent factors
lam = 0.6       # regularization parameter
lr = 1.0        # learning rate
epochs = 100    # epoch

# Model
LMF = LogisticMF(train, f, alpha, lam)

# Train
LMF.fit(lr, epochs)

# Recommend
rec_result = LMF.recommend()
print(rec_result['recommend'][:5])
print(rec_result['item_like_prob'][:5])

# Evaluation
print('train MPR = ', MPR(train, LMF.P))
print('test  MPR = ', MPR(test, LMF.P))


