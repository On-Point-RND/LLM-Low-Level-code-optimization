import asyncio
from typing import List


class DevicePool:
    """
    asyncio-safe pool of CUDA device indices.

    acquire() suspends the caller until a device is free.
    release() returns the device immediately.
    Only one worker can hold a given device_id at a time.
    """

    def __init__(self, device_ids: List[int]) -> None:
        self._queue: asyncio.Queue[int] = asyncio.Queue()
        for device_id in device_ids:
            self._queue.put_nowait(device_id)

    async def acquire(self) -> int:
        return await self._queue.get()

    def release(self, device_id: int) -> None:
        self._queue.put_nowait(device_id)
