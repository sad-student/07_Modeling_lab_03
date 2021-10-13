import queue

import numpy
import matplotlib.pyplot as plt
from functools import reduce


class Generator:
    p = None
    k = None
    b = None

    _instance = None
    initialized = False

    def __init__(self, k: int, b: int, p: int = 13):
        self.k = k
        self.b = b
        self.p = p

    @classmethod
    def get_instance(cls) -> 'Generator':
        return cls._instance

    @classmethod
    def initialize(cls, k: int, b: int, p: int = 13):
        cls._instance = Generator(k, b, p)
        cls.initialized = True

    def next(self, a: int = None, b: int = None):
        if a is None:
            a = 0
        else:
            a = int(a)
        if b is None:
            b = self.b
        else:
            b = int(b)
        self.p = a + (self.p * self.k) % (b - a)
        return self.p/(b - a)

    def uniform(self, a: float, b: float):
        return a + (b - a) * self.next()

    def normal(self, m: float, sigma: float, n: int = 6):
        n = int(n)
        _sum = sum([self.next() for _ in range(n)])
        return m + sigma * int(numpy.sqrt((12 + n - 1) / n)) * (_sum - int(n / 2))

    def exponential(self, alpha: float):
        return -(1 / alpha) * numpy.log(self.next())

    def gamma(self, alpha: float, nu: int):
        return -(1 / alpha) * numpy.log(reduce(lambda x, y: x * y, [self.next() for _ in range(nu)], 1))

    def triangle(self, a: float, b: float, inv: bool = False):
        _func = numpy.amax if inv else numpy.amin
        return a + (b - a) * _func([self.next(), self.next()])

    def simpson(self, a: float, b: float):
        args = [self.uniform(a / 2, b / 2) for _ in range(2)]
        return sum(args)


class QueueSystem:
    class Token:
        pass

    class NodeBase:
        source = None
        consumers = None

        tick_counter = None
        busy_counter = None
        consumed = None

        def __init__(self, src=None):
            if isinstance(src, QueueSystem.NodeBase):
                self.source = src
            self.consumers = []
            self.tick_counter = 0
            self.busy_counter = 0
            self.consumed = False

        def subscribe(self, dst=None):
            if isinstance(dst, QueueSystem.NodeBase):
                self.consumers.append(dst)

        def feed(self, token=None):
            if token is not None:
                for consumer in self.consumers:
                    consumer.feed(token)
            pass

        def consume(self):
            self.consumed = True
            pass

        def tick(self):
            self.tick_counter += 1
            self.consumed = False
            pass

    class Source(NodeBase):
        static = None
        blocking = None
        gen_prob = None
        gen_rate = None
        gen_counter = None
        blocked = None

        generated_value = None
        retired_counter = None

        def __init__(self, static: bool = False, blocking: bool = False, **kwargs):
            super().__init__(None)
            self.static = static
            if static:
                self.gen_rate = kwargs.get('gen_rate')
                if type(self.gen_rate) is not int:
                    raise ValueError
                self.gen_counter = 0
            else:
                self.gen_prob = kwargs.get('gen_prob')
            self.blocking = blocking
            if not self.blocking:
                self.retired_counter = 0
            self.blocked = False

        def tick(self):
            if not self.consumed:
                if self.blocking:
                    self.blocked = True
                else:
                    self.retired_counter += 1
            if not self.blocked:
                self.busy_counter += 1
                if self.static:
                    self.gen_counter += 1
                    if self.gen_counter >= self.gen_rate:
                        # TODO: generate new Token
                        self.generated_value = QueueSystem.Token()
                        self.gen_counter = 0
                else:
                    if Generator.get_instance().next() < self.gen_prob:
                        # TODO: generate new Token
                        self.generated_value = QueueSystem.Token()
                        pass
            super().tick()
            pass

        def consume(self):
            super().consume()
            return self.generated_value

    class Queue(NodeBase):
        capacity = None

        _queue = None
        overall_counter = None
        full_counter = None

        def __init__(self, src: 'QueueSystem.NodeBase' = None, capacity: int = 2):
            super().__init__(src)
            self.capacity = capacity
            self.overall_counter = 0
            self.full_counter = 0
            self._queue = queue.Queue(capacity)

        def tick(self):
            self.overall_counter += self._queue.qsize()
            if not self._queue.empty():
                self.busy_counter += 1
                if self._queue.full():
                    self.full_counter += 1
            super().tick()

        def consume(self):
            if not self._queue.empty():
                return self._queue.get(False)
            return None

    pass


class Modeler:

    pass


def lab():
    pass


if __name__ == '__main__':
    while True:
        lab()
        print(f"Quit? [y]/n")
        if input() != "n":
            break
