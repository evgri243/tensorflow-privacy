# Copyright 2021, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PrivacyAccountant abstract base class."""

import abc

from tensorflow_privacy.privacy.dp_event import dp_event
from tensorflow_privacy.privacy.dp_event import dp_event_builder


class PrivacyAccountant(metaclass=abc.ABCMeta):
  """Abstract base class for privacy accountants."""

  def __init__(self):
    self._ledger = dp_event_builder.DpEventBuilder()

  @abc.abstractmethod
  def is_supported(self, event: dp_event.DpEvent) -> bool:
    """Checks whether the `DpEvent` can be processed by this accountant.

    In general this will require recursively checking the structure of the
    `DpEvent`. In particular `ComposedDpEvent` and `SelfComposedDpEvent` should
    be recursively examined.

    Args:
      event: The `DpEvent` to check.

    Returns:
      True iff this accountant supports processing `event`.
    """

  @abc.abstractmethod
  def _compose(self, event: dp_event.DpEvent, count: int = 1):
    """Update internal state to account for application of a `DpEvent`.

    Calls to `get_epsilon` or `get_delta` after calling `_compose` will return
    values that account for this `DpEvent`.

    Args:
      event: A `DpEvent` to process.
      count: The number of times to compose the event.
    """

  def compose(self, event: dp_event.DpEvent, count: int = 1):
    """Update internal state to account for application of a `DpEvent`.

    Calls to `get_epsilon` or `get_delta` after calling `compose` will return
    values that account for this `DpEvent`.

    Args:
      event: A `DpEvent` to process.
      count: The number of times to compose the event.

    Raises:
      TypeError: `event` is not supported by this `PrivacyAccountant`.
    """
    if not self.is_supported(event):
      raise TypeError(f'`DpEvent` {event} is of unsupported type.')
    self._ledger.compose(event, count)
    self._compose(event, count)

  @property
  def ledger(self) -> dp_event.DpEvent:
    """Returns the (composed) `DpEvent` processed so far by this accountant."""
    return self._ledger.build()

  @abc.abstractmethod
  def get_epsilon(self, target_delta: float) -> float:
    """Gets the current epsilon.

    Args:
      target_delta: The target delta.

    Returns:
      The current epsilon, accounting for all composed `DpEvent`s.
    """

  def get_delta(self, target_epsilon: float) -> float:
    """Gets the current delta.

    An implementer of `PrivacyAccountant` may choose not to override this, in
    which case `NotImplementedError` will be raised.

    Args:
      target_epsilon: The target epsilon.

    Returns:
      The current delta, accounting for all composed `DpEvent`s.
    """
    raise NotImplementedError()
