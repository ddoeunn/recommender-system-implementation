import numpy as np
from scipy import sparse
from sklearn.utils.validation import check_random_state
from utils.common.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    SEED
)

class Split():
    def __init__(self, data, train_size, user_col=DEFAULT_USER_COL, item_col=DEFAULT_ITEM_COL,
                 rating_col=DEFAULT_RATING_COL, random_state=SEED, dtype=np.float64):
        self.users = data[user_col].to_numpy() - 1
        self.items = data[item_col].to_numpy() - 1
        self.ratings = data[rating_col].to_numpy()
        self.train_size = train_size
        self.random_state = random_state
        self.dtype = dtype
        self.shape = (np.unique(self.users).shape[0], np.unique(self.items).shape[0])

    def get_train_mask(self):
        random_state = check_random_state(self.random_state)
        n_events = self.users.shape[0]
        train_mask = random_state.rand(n_events) <= self.train_size

        for array in (self.users, self.items):
            present = array[train_mask]
            test_vals = array[~train_mask]
            is_ts_val_in_present = np.in1d(test_vals, present)  # test_vals이 present를 포함하고 있는지 리턴
            ts_val_not_in_present = test_vals[np.where(~is_ts_val_in_present)[0]]  # test_vals 중 present에 포함되지 않는 값들
            missing = np.unique(ts_val_not_in_present)

            if missing.shape[0] == 0:
                continue

            array_mask_missing = np.in1d(array, missing)  # train_mask에 포함되어야 하는 값들
            where_missing = np.where(array_mask_missing)[0]  # train_mask에 포함되어야 하는 값들의 index

            # train_mask에 포함시키기
            added = set()
            for idx, val in zip(where_missing, array[where_missing]):
                if val in added:
                    continue
                train_mask[idx] = True
                added.add(val)

        return train_mask

    def make_sparse_csr(self):
        data, rows, cols = (np.asarray(x) for x in (self.ratings, self.users, self.items))
        sparse_csr = sparse.csr_matrix((data, (rows, cols)), shape=self.shape, dtype=self.dtype)

        return sparse_csr

    def make_sparse_tr_ts(self, train_mask):
        r_train = self.make_sparse_csr()
        r_test = np.zeros(self.shape)
        for ts_u, ts_i, ts_r in zip(self.users[~train_mask], self.items[~train_mask], self.ratings[~train_mask]):
            r_test[ts_u, ts_i] = ts_r
        r_test = sparse.csr_matrix(r_test)

        return r_train, r_test

    def train_test_split(self):
        train_mask = self.get_train_mask()
        train, test = self.make_sparse_tr_ts(train_mask=train_mask)
        print('shape of train= {}, test={}'.format(train.shape, test.shape))
        print('size of train set = ', sum(train_mask) / len(self.users))
        return train.toarray(), test.toarray()

