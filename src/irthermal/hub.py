"""Fan-out of captured frames to however many consumers want them.

One thread reads a camera; the MJPEG streams, the metrics exporter and the
telemetry sockets all subscribe here. That is the whole reason this exists: in
v1 every consumer called `read()` itself, so two viewers meant two captures
competing for one device.

Subscribers cannot block the reader. Each one gets a small queue, and when it
fills the oldest frame is discarded rather than queued -- for live video the
newest frame is the only one worth having, and unbounded buffering of a 25 fps
stream is just a slow memory leak.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Self

from irthermal.core.frame import Frame

DEFAULT_QUEUE_SIZE = 1


class Subscription:
    """A consumer's view of the frame stream.

    Async-iterate it to receive frames. Iteration ends when the hub closes.
    """

    def __init__(self, hub: FrameHub, maxsize: int) -> None:
        self._hub = hub
        self._queue: asyncio.Queue[Frame | None] = asyncio.Queue(maxsize=maxsize)
        self._closed = False
        self.dropped = 0
        """Frames discarded because this consumer was not keeping up."""

    def _offer(self, frame: Frame) -> None:
        """Enqueue a frame, evicting the oldest if this consumer is behind."""
        if self._queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
                self.dropped += 1
        self._queue.put_nowait(frame)

    def _stop(self) -> None:
        """Wake any waiting consumer with the end-of-stream sentinel.

        The queue may be full, so make room: an undelivered frame matters less
        than the consumer noticing that the stream has ended.
        """
        if self._closed:
            return

        self._closed = True
        if self._queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
        self._queue.put_nowait(None)

    async def get(self) -> Frame | None:
        """Wait for the next frame. `None` once the hub has closed."""
        return await self._queue.get()

    def close(self) -> None:
        """Unsubscribe. Safe to call more than once."""
        self._hub._remove(self)
        self._stop()

    def __aiter__(self) -> AsyncIterator[Frame]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[Frame]:
        while True:
            frame = await self._queue.get()
            if frame is None:
                return
            yield frame

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class FrameHub:
    """Publishes frames to every current subscriber.

    Construct it inside the running event loop: it captures that loop so the
    capture thread can hand frames over safely.
    """

    def __init__(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._subscribers: list[Subscription] = []
        self._latest: Frame | None = None
        self._closed = False

    @property
    def latest(self) -> Frame | None:
        """The most recent frame, or None before the first capture."""
        return self._latest

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def subscribe(self, maxsize: int = DEFAULT_QUEUE_SIZE) -> Subscription:
        """Start receiving frames.

        A `maxsize` above 1 buffers, for consumers that must not miss frames --
        lock-in integration, for one. Live viewers should stay at 1.
        """
        if self._closed:
            raise RuntimeError("hub is closed")

        subscription = Subscription(self, maxsize)
        self._subscribers.append(subscription)
        return subscription

    def _remove(self, subscription: Subscription) -> None:
        if subscription in self._subscribers:
            self._subscribers.remove(subscription)

    def publish(self, frame: Frame) -> None:
        """Hand a frame to every subscriber. Must run on the event loop."""
        if self._closed:
            return
        self._latest = frame
        for subscription in self._subscribers:
            subscription._offer(frame)

    def publish_from_thread(self, frame: Frame) -> None:
        """Publish from the capture thread."""
        if self._closed:
            return
        # A closed loop means the service is shutting down and the capture
        # thread has not noticed yet; the frame is simply no longer wanted.
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self.publish, frame)

    def close(self) -> None:
        """End every subscription. Further publishes are ignored."""
        self._closed = True
        for subscription in list(self._subscribers):
            subscription._stop()
        self._subscribers.clear()
