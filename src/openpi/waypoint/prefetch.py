"""Background-thread batch prefetcher shared by both training scripts.

This lives in the package rather than in ``scripts/`` because it used to exist
twice: ``train_waypoint_joint.py`` was hardened against a dead worker (sentinel
+ timed ``get`` + forwarded traceback) while ``scripts/train_waypoint.py`` kept
the original version, whose ``__next__`` was a bare unbounded ``queue.get()``.
A worker that raised there parked the main thread forever with no traceback
anywhere -- a training job that looks alive and produces nothing.  One copy
means one place to harden.
"""

from __future__ import annotations

import logging
import queue
import threading
import traceback


class PrefetchIter:
    """Background-thread prefetcher for a DataLoader.

    While the main thread runs GPU forward/backward, a daemon thread pre-loads
    the next batch via the DataLoader iterator, hiding most of the CPU
    data-loading latency behind GPU compute.  The loader is restarted on
    ``StopIteration``, so iteration is endless and the caller controls the step
    budget.

    Failure semantics (the reason this class exists):
      * a worker exception is re-raised **in the consumer**, with the worker's
        traceback logged;
      * the worker pushes a sentinel on the way out so a consumer already parked
        in ``get()`` wakes up instead of hanging;
      * ``get`` is polled with a timeout, so even a lost sentinel (full queue)
        surfaces as an error rather than a hang;
      * a *live* worker that never yields a batch (e.g. a dataset that skips
        every episode) is reported with escalating warnings instead of silence.
    """

    _SENTINEL = object()

    def __init__(self, loader, prefetch_count=2):
        self._loader = loader
        self._queue: queue.Queue = queue.Queue(maxsize=prefetch_count)
        self._stop = threading.Event()
        self._exception: BaseException | None = None
        self._traceback: str | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            it = iter(self._loader)
            while not self._stop.is_set():
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(self._loader)
                    batch = next(it)
                self._queue.put(batch)
        except BaseException as exc:  # noqa: BLE001 - forwarded to the main thread
            self._exception = exc
            self._traceback = traceback.format_exc()
            # Wake the consumer.  Without this sentinel the main thread parked
            # forever in an unbounded `queue.get()` while the worker was already
            # dead -- a hung training job with no traceback anywhere.  `put`
            # itself must not block, hence the timeout-and-drop fallback.
            try:
                self._queue.put(self._SENTINEL, timeout=5.0)
            except queue.Full:
                logging.error("prefetch worker died and the queue is full; consumer will poll")

    def __iter__(self):
        return self

    def __next__(self):
        self._raise_if_failed()
        waited = 0.0
        while True:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                self._raise_if_failed()
                if not self._thread.is_alive():
                    raise RuntimeError("prefetch worker exited without an exception")
                # A live worker that never yields (e.g. a dataset whose episodes are
                # all skipped) would otherwise poll here forever with no output at
                # all.  Say so, escalating, so a stall is visible in the log.
                waited += 1.0
                if waited in (30.0, 120.0) or (waited >= 300.0 and waited % 300.0 == 0):
                    logging.warning(
                        "prefetch worker has produced no batch for %.0fs -- the dataset may be "
                        "yielding nothing (check the skip counters in the dataset epoch log)",
                        waited,
                    )
                continue
            if item is self._SENTINEL:
                self._raise_if_failed()
                raise RuntimeError("prefetch worker signalled failure without an exception")
            return item

    def _raise_if_failed(self):
        if self._exception is not None:
            if self._traceback:
                logging.error("prefetch worker traceback:\n%s", self._traceback)
            raise self._exception

    def stop(self):
        self._stop.set()
