from typing import List, Union

import numpy as np
import torch

from blades.clients.client import BladesClient
from .mean import _BaseAggregator



class Dnc(_BaseAggregator):
    r"""A robust aggregator from paper `Manipulating the Byzantine: Optimizing
    Model Poisoning Attacks and Defenses for Federated Learning.

    <https://par.nsf.gov/servlets/purl/10286354>`_.
    """

    def __init__(
        self, num_byzantine, *, sub_dim=10000, num_iters=1, filter_frac=1.0
    ) -> None:
        super(Dnc, self).__init__()

        self.num_byzantine = num_byzantine
        self.sub_dim = sub_dim
        self.num_iters = num_iters
        self.fliter_frac = filter_frac

    def __call__(
        self, inputs: Union[List[BladesClient], List[torch.Tensor], torch.Tensor]
    ):
        updates = self._get_updates(inputs)
        d = len(updates[0])





        benign_ids = []
        

        mu = updates.mean(dim=0)
        centered_update = updates - mu

        rank = 1000
        Omega = torch.rand(centered_update.shape[1], rank)

        Y = centered_update @ Omega
        #for q in range(3):
        #    Y = centered_update @ (centered_update.T @ Y)
        Q, _ = torch.linalg.qr(Y)

        B = Q.T @ centered_update
        u_tilde, s, v = torch.linalg.svd(B, full_matrices=False)
        

        v = v[0, :]
        score = np.array([(torch.dot(b_i, v) ** 2).item() for b_i in B])

        good = score.argsort()[
            : len(updates) - int(self.fliter_frac * self.num_byzantine)]
        benign_ids = list(set(good))
        benign_updates = updates[benign_ids, :].mean(dim=0)
        return benign_updates
