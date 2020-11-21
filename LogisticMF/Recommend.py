import numpy as np
from itertools import islice
from utils.common.constants import DEFAULT_TOP_K

def recommend_top_k(rating_mat, prob_mat, top_k=DEFAULT_TOP_K):

    recommend_lst = []
    recommended_prob_lst = []
    for ratings, probs in zip(rating_mat, prob_mat):
        already_liked = set(np.nonzero(ratings)[0])
        count = top_k + len(already_liked)

        idx = np.argpartition(probs, -count)[-count:]
        top_idx = np.argsort(probs[idx])[::-1]
        best = idx[top_idx]
        top_n = list(islice((rec for rec in best if rec not in already_liked), top_k))
        recommended_prob_lst.append(probs[top_n].tolist())
        recommend_lst.append(top_n)

    recommend_arr = np.array(recommend_lst)
    recommend_rst = dict()
    recommend_rst['recommend'] = recommend_arr
    recommend_rst['item_like_prob'] = recommended_prob_lst
    return recommend_rst
