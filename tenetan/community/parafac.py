from tenetan.networks import SnapshotGraph
import tensorly as tl
import numpy as np

__all__ = ['parafac_community']


def parafac_community(network: SnapshotGraph, n_communities: int, **tensorly_kwargs):
    """
    Gauvin L, Panisson A, Cattuto C (2014) Detecting the Community Structure and Activity Patterns of Temporal Networks:
    A Non-Negative Tensor Factorization Approach. PLoS ONE 9(1): e86028.
    https://doi.org/10.1371/journal.pone.0086028
    :param network: Temporal network
    :param n_communities: Number of communities to detect
    :param tensorly_kwargs: kwargs to pass to tensorly.decomposition.non_negative_parafac_hals
    :return:
    """

    cp_tensor, errors = tl.decomposition.non_negative_parafac_hals(network._tensor, n_communities, return_errors=True, **tensorly_kwargs)
    in_communities, out_communities, raw_temporal_activity = cp_tensor.factors
    return in_communities, out_communities, raw_temporal_activity, errors
