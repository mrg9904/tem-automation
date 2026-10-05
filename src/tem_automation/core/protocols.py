from __future__ import annotations

from typing import Protocol
from typing import runtime_checkable

from tem_automation.core.models import AcquisitionSettings
from tem_automation.core.models import ImageFrame
from tem_automation.core.models import MicroscopeState


@runtime_checkable
class Microscope(Protocol):
    """Minimum platform-independent API required by HAADF autofocus."""

    def connect(self) -> None:
        ...

    def close(self) -> None:
        ...

    def get_state(self) -> MicroscopeState:
        ...

    def set_defocus(self, defocus_m: float) -> None:
        ...

    def acquire(self, settings: AcquisitionSettings) -> ImageFrame:
        ...

