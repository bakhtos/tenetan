from tenetan.networks import SnapshotGraph
import tensorly as tl
import numpy as np

import logging

__all__ = ['parafac_community']


def parafac_community(network: SnapshotGraph, n_communities: int, /, *to_return, **tensorly_kwargs):
    """
    Gauvin L, Panisson A, Cattuto C (2014) Detecting the Community Structure and Activity Patterns of Temporal Networks:
    A Non-Negative Tensor Factorization Approach. PLoS ONE 9(1): e86028.
    https://doi.org/10.1371/journal.pone.0086028
    :param network: Temporal network
    :param n_communities: Number of communities to detect
    :param to_return: Data to return, default ('in_communities', 'out_communities', 'raw_temporal_activity')
    :param tensorly_kwargs: kwargs to pass to tensorly.decomposition.non_negative_parafac_hals
    :return:
    """

    if len(to_return) == 0:
        to_return = ('in_communities', 'out_communities', 'raw_temporal_activity')
    tensorly_kwargs['return_errors'] = True

    cp_tensor, errors = tl.decomposition.non_negative_parafac_hals(network._tensor, n_communities, **tensorly_kwargs)
    in_communities, out_communities, raw_temporal_activity = cp_tensor.factors
    return_dict = {}
    for data in to_return:  # TODO Switch to match when upgrading to 9.10
        if data == "in_communities":
            return_dict[data] = in_communities
        elif data == "out_communities":
            return_dict[data] = out_communities
        elif data == "raw_temporal_activity":
            return_dict[data] = raw_temporal_activity
        elif data == "errors":
            return_dict[data] = errors
        else:
            logging.error(f"{__name__}: unknown data requested from parafac_communities - {data}")

    return return_dict
