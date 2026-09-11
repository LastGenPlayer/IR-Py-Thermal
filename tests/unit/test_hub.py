"""Frame fan-out: delivery, backpressure and shutdown."""

from __future__ import annotations

import asyncio

import pytest

from conftest import make_frame
from irthermal.core.frame import Frame
from irthermal.core.resolutions import ResolutionParams
from irthermal.hub import FrameHub


def frames(params: ResolutionParams, count: int) -> list[Frame]:
    return [make_frame(params, sequence=n) for n in range(1, count + 1)]


async def test_subscriber_receives_published_frames(small_params: ResolutionParams) -> None:
    hub = FrameHub()
    async with hub.subscribe() as subscription:
        published = make_frame(small_params, sequence=7)
        hub.publish(published)

        assert await subscription.get() is published


async def test_latest_tracks_the_most_recent_frame(small_params: ResolutionParams) -> None:
    hub = FrameHub()
    assert hub.latest is None

    for frame in frames(small_params, 3):
        hub.publish(frame)

    assert hub.latest is not None
    assert hub.latest.sequence == 3


async def test_every_subscriber_gets_the_frame(small_params: ResolutionParams) -> None:
    hub = FrameHub()
    async with hub.subscribe() as first, hub.subscribe() as second:
        assert hub.subscriber_count == 2
        frame = make_frame(small_params)
        hub.publish(frame)

        assert await first.get() is frame
        assert await second.get() is frame


async def test_slow_subscriber_gets_the_newest_frame_not_the_oldest(
    small_params: ResolutionParams,
) -> None:
    """Live video wants the current frame; a stale backlog is worse than a gap."""
    hub = FrameHub()
    async with hub.subscribe() as subscription:
        for frame in frames(small_params, 5):
            hub.publish(frame)

        received = await subscription.get()

    assert received is not None
    assert received.sequence == 5
    assert subscription.dropped == 4


async def test_queue_does_not_grow_without_bound(small_params: ResolutionParams) -> None:
    """A subscriber that never reads must not accumulate 25 frames a second."""
    hub = FrameHub()
    async with hub.subscribe() as subscription:
        frame = make_frame(small_params)
        for _ in range(10_000):
            hub.publish(frame)

        assert subscription._queue.qsize() == 1
        assert subscription.dropped == 9_999


async def test_buffered_subscriber_keeps_every_frame_up_to_its_size(
    small_params: ResolutionParams,
) -> None:
    """Lock-in integration needs consecutive frames, so deeper queues are allowed."""
    hub = FrameHub()
    async with hub.subscribe(maxsize=4) as subscription:
        for frame in frames(small_params, 4):
            hub.publish(frame)

        received = [await subscription.get() for _ in range(4)]

    assert [f.sequence for f in received if f] == [1, 2, 3, 4]
    assert subscription.dropped == 0


async def test_iteration_ends_when_the_hub_closes(small_params: ResolutionParams) -> None:
    hub = FrameHub()
    subscription = hub.subscribe()

    async def consume() -> list[int]:
        return [frame.sequence async for frame in subscription]

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)

    hub.publish(make_frame(small_params, sequence=1))
    await asyncio.sleep(0)
    hub.close()

    assert await asyncio.wait_for(task, timeout=1.0) == [1]


async def test_closing_a_subscription_unregisters_it(small_params: ResolutionParams) -> None:
    hub = FrameHub()
    subscription = hub.subscribe()
    assert hub.subscriber_count == 1

    subscription.close()
    assert hub.subscriber_count == 0

    hub.publish(make_frame(small_params))
    assert await subscription.get() is None


async def test_publishing_after_close_is_ignored(small_params: ResolutionParams) -> None:
    hub = FrameHub()
    hub.close()
    hub.publish(make_frame(small_params))

    assert hub.latest is None
    with pytest.raises(RuntimeError, match="closed"):
        hub.subscribe()


async def test_publish_from_thread_reaches_subscribers(small_params: ResolutionParams) -> None:
    """The capture thread hands frames over without touching the loop directly."""
    hub = FrameHub()
    frame = make_frame(small_params, sequence=99)

    async with hub.subscribe() as subscription:
        await asyncio.to_thread(hub.publish_from_thread, frame)
        received = await asyncio.wait_for(subscription.get(), timeout=1.0)

    assert received is not None
    assert received.sequence == 99
