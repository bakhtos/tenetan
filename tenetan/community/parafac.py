from tenetan.networks import SnapshotGraph
import tensorly as tl
import numpy as np

__all__ = ['parafac_community']


def parafac_community(N: SnapshotGraph, R: int):
    """
    Gauvin L, Panisson A, Cattuto C (2014) Detecting the Community Structure and Activity Patterns of Temporal Networks:
    A Non-Negative Tensor Factorization Approach. PLoS ONE 9(1): e86028.
    https://doi.org/10.1371/journal.pone.0086028
    :param N: Temporal network
    :param R: Number of communities
    :return:
    """

    cp_tensor, errors = tl.decomposition.non_negative_parafac_hals(N._tensor, R, return_errors=True)
    in_communities, out_communities, time_activity = cp_tensor.factors
    return in_communities, out_communities, time_activity, errors
