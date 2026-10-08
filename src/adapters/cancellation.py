"""Cooperative cancellation shared by algorithms, adapters and scripts."""
from contextlib import contextmanager
from pathlib import Path
import threading


class WorkflowCancelled(BaseException):
    pass


_state = threading.local()


def check_cancelled():
    callback = getattr(_state, "callback", None)
    stop_file = getattr(_state, "stop_file", None)
    if (stop_file is not None and Path(stop_file).exists()) or (callback is not None and callback()):
        raise WorkflowCancelled("Workflow cancelled by user")


@contextmanager
def cancellation_scope(callback):
    previous = getattr(_state, "callback", None)
    _state.callback = callback
    try:
        yield
    finally:
        _state.callback = previous


@contextmanager
def suspend_cancellation():
    previous = set_stop_file(None)
    try:
        with cancellation_scope(None):
            yield
    finally:
        set_stop_file(previous)


def set_stop_file(path):
    previous = getattr(_state, "stop_file", None)
    _state.stop_file = path
    return previous
