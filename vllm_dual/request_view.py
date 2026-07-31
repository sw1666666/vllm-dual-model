"""Read-only request progress views for independent KV allocators."""

from __future__ import annotations


class RequestProgressView:
    def __init__(self, request, num_computed_tokens: int):
        object.__setattr__(self, "_request", request)
        object.__setattr__(self, "num_computed_tokens", num_computed_tokens)

    def __getattr__(self, name):
        return getattr(self._request, name)

    def __setattr__(self, name, value):
        raise AttributeError("RequestProgressView is read-only")


class RequestIdView:
    def __init__(self, request_id: str):
        self.request_id = request_id
