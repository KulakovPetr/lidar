"""Pair a playback token with one cloud. The stamp is not the identity."""

from __future__ import annotations


def pair_arrival(pending_token, pending_cloud, token=None, cloud=None):
    """Return (pair, pending_token, pending_cloud).

    A pair is (cloud, token). Sequence lives on the token. Two clouds may
    share a stamp; that does not swap their tokens when only one pair is
    in flight.
    """
    if token is not None and cloud is not None:
        return (cloud, token), None, None
    if token is not None:
        if pending_cloud is not None:
            return (pending_cloud, token), None, None
        return None, token, pending_cloud
    if cloud is not None:
        if pending_token is not None:
            return (cloud, pending_token), None, None
        return None, pending_token, cloud
    return None, pending_token, pending_cloud
