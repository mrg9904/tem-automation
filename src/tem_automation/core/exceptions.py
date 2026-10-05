class MicroscopeError(RuntimeError):
    """Base exception raised by microscope adapters."""


class ConnectionError(MicroscopeError):
    """Raised when an adapter cannot connect to its instrument."""


class AcquisitionError(MicroscopeError):
    """Raised when an image acquisition fails."""


class UnsupportedCapabilityError(MicroscopeError):
    """Raised when an adapter does not provide a requested capability."""

