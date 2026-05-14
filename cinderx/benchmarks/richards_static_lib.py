# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
based on a Java version:
 Based on original version written in BCPL by Dr Martin Richards
 in 1981 at Cambridge University Computer Laboratory, England
 and a C++ version derived from a Smalltalk version written by
 L Peter Deutsch.
 Java version:  Copyright (C) 1995 Sun Microsystems, Inc.
 Translation from C++, Mario Wolczko
 Outer loop added by Alex Jacoby
"""

from __future__ import annotations

import __static__
from __static__ import box, cbool, inline, int64

from typing import Final, cast

import cinderx.jit
import sys

# Task IDs
I_IDLE: Final[int] = 1
I_WORK: Final[int] = 2
I_HANDLERA: Final[int] = 3
I_HANDLERB: Final[int] = 4
I_DEVA: Final[int] = 5
I_DEVB: Final[int] = 6

# Packet types
K_DEV: Final[int] = 1000
K_WORK: Final[int] = 1001

# Packet

BUFSIZE: Final[int] = 4


class Packet:
    def __init__(self, l: Packet | None, i: int64, k: int64) -> None:
        self.link: Packet | None = l
        self.ident: int64 = i
        self.kind: int64 = k
        self.datum: int = 0
        self.data: list[int] = [0] * BUFSIZE

    def append_to(self, lst: Packet | None) -> Packet:
        self.link = None
        if lst is None:
            return self
        p: Packet = lst
        next: Packet | None = p.link
        while next is not None:
            p = next
            next = p.link
        p.link = self
        return lst


# Task Records


class TaskRec:
    pass


class DeviceTaskRec(TaskRec):
    def __init__(self) -> None:
        self.pending: Packet | None = None


class IdleTaskRec(TaskRec):
    def __init__(self) -> None:
        self.control: int64 = int64(1)
        self.count: int64 = int64(10000)


class HandlerTaskRec(TaskRec):
    def __init__(self) -> None:
        self.work_in: Packet | None = None
        self.device_in: Packet | None = None

    def workInAdd(self, p: Packet) -> Packet:
        self.work_in = work_in = p.append_to(self.work_in)
        return work_in

    def deviceInAdd(self, p: Packet) -> Packet:
        self.device_in = device_in = p.append_to(self.device_in)
        return device_in


class WorkerTaskRec(TaskRec):
    def __init__(self) -> None:
        self.destination: int64 = int64(I_HANDLERA)
        self.count: int64 = int64(0)


# Task


class TaskState:
    def __init__(self) -> None:
        self.packet_pending: cbool = cbool(True)
        self.task_waiting: cbool = cbool(False)
        self.task_holding: cbool = cbool(False)

    def packetPending(self) -> TaskState:
        self.packet_pending: cbool = cbool(True)
        self.task_waiting: cbool = cbool(False)
        self.task_holding: cbool = cbool(False)
        return self

    def waiting(self) -> TaskState:
        self.packet_pending: cbool = cbool(False)
        self.task_waiting: cbool = cbool(True)
        self.task_holding: cbool = cbool(False)
        return self

    def running(self) -> TaskState:
        self.packet_pending: cbool = cbool(False)
        self.task_waiting: cbool = cbool(False)
        self.task_holding: cbool = cbool(False)
        return self

    def waitingWithPacket(self) -> TaskState:
        self.packet_pending: cbool = cbool(True)
        self.task_waiting: cbool = cbool(True)
        self.task_holding: cbool = cbool(False)
        return self

    def isPacketPending(self) -> cbool:
        return self.packet_pending

    def isTaskWaiting(self) -> cbool:
        return self.task_waiting

    def isTaskHolding(self) -> cbool:
        return self.task_holding

    def isTaskHoldingOrWaiting(self) -> cbool:
        if self.task_holding:
            return cbool(True)
        if not self.packet_pending:
            if self.task_waiting:
                return cbool(True)
        return cbool(False)

    def isWaitingWithPacket(self) -> cbool:
        if self.packet_pending:
            if self.task_waiting:
                if not self.task_holding:
                    return cbool(True)
        return cbool(False)


tracing: bool = False
layout: int = 0


def trace(a: object) -> None:
    global layout
    layout -= 1
    if layout <= 0:
        print()
        layout = 50
    print(a, end="")


TASKTABSIZE: Final[int] = 10


class TaskWorkArea:
    def __init__(self) -> None:
        none_task = cast(Task | None, None)
        self.taskTab: list[Task | None] = [none_task] * TASKTABSIZE

        self.taskList: Task | None = None

        self.holdCount: int64 = int64(0)
        self.qpktCount: int64 = int64(0)


class Task(TaskState):
    def __init__(
        self,
        i: int64,
        p: int64,
        w: Packet | None,
        initialState: TaskState,
        r: TaskRec,
    ) -> None:
        wa: TaskWorkArea = taskWorkArea
        self.link: Task | None = wa.taskList
        self.ident: int64 = i
        self.priority: int64 = p
        self.input: Packet | None = w

        self.packet_pending: cbool = initialState.isPacketPending()
        self.task_waiting: cbool = initialState.isTaskWaiting()
        self.task_holding: cbool = initialState.isTaskHolding()

        self.handle = r

        wa.taskList = self
        wa.taskTab[i] = self

    def fn(self, pkt: Packet | None, r: TaskRec) -> Task | None:
        raise NotImplementedError

    def addPacket(self, p: Packet, old: Task) -> Task:
        if self.input is None:
            self.input = p
            self.packet_pending: cbool= cbool(True)
            if self.priority > old.priority:
                return self
        else:
            p.append_to(self.input)
        return old

    def runTask(self) -> Task | None:
        if TaskState.isWaitingWithPacket(self):
            msg: Packet | None = self.input
            if msg is not None:
                self.input = msg.link
                if self.input is None:
                    self.running()
                else:
                    self.packetPending()
        else:
            msg = None

        return self.fn(msg, self.handle)

    def waitTask(self) -> Task:
        self.task_waiting: cbool = cbool(True)
        return self

    def hold(self) -> Task | None:
        taskWorkArea.holdCount += 1
        self.task_holding: cbool = cbool(True)
        return self.link

    def release(self, i: int) -> Task:
        t: Task = self.findtcb(i)
        t.task_holding = cbool(False)
        if t.priority > self.priority:
            return t
        else:
            return self

    def qpkt(self, pkt: Packet) -> Task:
        t: Task = self.findtcb(box(pkt.ident))
        taskWorkArea.qpktCount += 1
        pkt.link = None
        pkt.ident = self.ident
        return t.addPacket(pkt, self)

    def findtcb(self, id: int) -> Task:
        t = taskWorkArea.taskTab[id]
        return cast(Task, t)


taskWorkArea: TaskWorkArea = TaskWorkArea()


# DeviceTask


class DeviceTask(Task):
    def __init__(
        self, i: int64, p: int64, w: Packet | None, s: TaskState, r: DeviceTaskRec
    ) -> None:
        Task.__init__(self, i, p, w, s, r)

    def fn(self, pkt: Packet | None, r: TaskRec) -> Task:
        d: DeviceTaskRec = cast(DeviceTaskRec, r)
        if pkt is None:
            pkt = d.pending
            if pkt is None:
                return self.waitTask()
            else:
                d.pending = None
                return self.qpkt(pkt)
        else:
            d.pending = pkt
            if tracing:
                trace(pkt.datum)
            return cast(Task, self.hold())


class HandlerTask(Task):
    def __init__(
        self, i: int64, p: int64, w: Packet, s: TaskState, r: HandlerTaskRec
    ) -> None:
        Task.__init__(self, i, p, w, s, r)

    def fn(self, pkt: Packet | None, r: TaskRec) -> Task:
        h: HandlerTaskRec = cast(HandlerTaskRec, r)
        if pkt is not None:
            if pkt.kind == int64(K_WORK):
                h.workInAdd(pkt)
            else:
                h.deviceInAdd(pkt)
        work: Packet | None = h.work_in
        if work is None:
            return self.waitTask()
        count: int = work.datum
        if count >= BUFSIZE:
            h.work_in = work.link
            return self.qpkt(work)

        dev: Packet | None = h.device_in
        if dev is None:
            return self.waitTask()

        h.device_in = dev.link
        dev.datum = work.data[count]
        work.datum = count + 1
        return self.qpkt(dev)


# IdleTask


class IdleTask(Task):
    def __init__(
        self, i: int64, p: int64, w: int, s: TaskState, r: IdleTaskRec
    ) -> None:
        Task.__init__(self, i, int64(0), None, s, r)

    def fn(self, pkt: Packet | None, r: TaskRec) -> Task | None:
        i: IdleTaskRec = cast(IdleTaskRec, r)
        i.count -= 1
        if i.count == 0:
            return self.hold()
        elif i.control & 1 == 0:
            i.control //= 2
            return self.release(I_DEVA)
        else:
            i.control = i.control // 2 ^ int64(0xD008)
            return self.release(I_DEVB)


# WorkTask


A: Final[int] = 65  # ord('A')


class WorkTask(Task):
    def __init__(
        self, i: int64, p: int64, w: Packet, s: TaskState, r: WorkerTaskRec
    ) -> None:
        Task.__init__(self, i, p, w, s, r)

    def fn(self, pkt: Packet | None, r: TaskRec) -> Task:
        w: WorkerTaskRec = cast(WorkerTaskRec, r)
        if pkt is None:
            return self.waitTask()

        if w.destination == int64(I_HANDLERA):
            dest: int64 = int64(I_HANDLERB)
        else:
            dest = int64(I_HANDLERA)

        w.destination = dest
        pkt.ident = dest
        pkt.datum = 0

        i = 0
        while i < BUFSIZE:
            x: int64 = w.count + 1
            w.count = x
            if w.count > int64(26):
                w.count = int64(1)
            pkt.data[i] = A + box(w.count) - 1
            i = i + 1

        return self.qpkt(pkt)


def schedule() -> None:
    t: Task | None = taskWorkArea.taskList
    while t is not None:
        if tracing:
            print("tcb =", box(t.ident))

        if TaskState.isTaskHoldingOrWaiting(t):
            t = t.link
        else:
            if tracing:
                trace(chr(ord("0") + box(t.ident)))
            t = t.runTask()


class Richards:
    def run(self, iterations: int) -> bool:
        for i in range(iterations):
            taskWorkArea.holdCount = int64(0)
            taskWorkArea.qpktCount = int64(0)

            IdleTask(int64(I_IDLE), int64(1), 10000, TaskState().running(), IdleTaskRec())

            wkq: Packet | None = Packet(None, int64(0), int64(K_WORK))
            wkq = Packet(wkq, int64(0), int64(K_WORK))
            WorkTask(
                int64(I_WORK),
                int64(1000),
                wkq,
                TaskState().waitingWithPacket(),
                WorkerTaskRec(),
            )

            wkq = Packet(None, int64(I_DEVA), int64(K_DEV))
            wkq = Packet(wkq, int64(I_DEVA), int64(K_DEV))
            wkq = Packet(wkq, int64(I_DEVA), int64(K_DEV))
            HandlerTask(
                int64(I_HANDLERA),
                int64(2000),
                wkq,
                TaskState().waitingWithPacket(),
                HandlerTaskRec(),
            )

            wkq = Packet(None, int64(I_DEVB), int64(K_DEV))
            wkq = Packet(wkq, int64(I_DEVB), int64(K_DEV))
            wkq = Packet(wkq, int64(I_DEVB), int64(K_DEV))
            HandlerTask(
                int64(I_HANDLERB),
                int64(3000),
                wkq,
                TaskState().waitingWithPacket(),
                HandlerTaskRec(),
            )

            wkq = None
            DeviceTask(int64(I_DEVA), int64(4000), wkq, TaskState().waiting(), DeviceTaskRec())
            DeviceTask(int64(I_DEVB), int64(5000), wkq, TaskState().waiting(), DeviceTaskRec())

            schedule()

            if taskWorkArea.holdCount == 9297 and taskWorkArea.qpktCount == 23246:
                pass
            else:
                print("err")
                return False
        return True

if __name__ == "__main__":
    cinderx.jit.auto()

    import time
    s = time.perf_counter()
    num_iterations = 100
    if len(sys.argv) > 1:
        num_iterations = int(sys.argv[1])
    Richards().run(num_iterations)
    e = time.perf_counter()
    print(e-s)
