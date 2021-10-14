import queue
from abc import ABC

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
        _sources = None
        _consumers = None

        _tick_counter = None
        _busy_counter = None

        # limit consume() call to prevent loops in graph
        _consume_counter = None

        # limit tick() call to prevent loops in graph
        _tick_call_counter = None

        def __init__(self, src=None):
            self._sources = []
            if isinstance(src, QueueSystem.NodeBase):
                self._sources.append(src)
                src.subscribe(self)
            elif type(src) is list:
                for item in src:
                    self._sources.append(item)
                    item.subscribe(self)
            self._consumers = []
            self._tick_counter = 0
            self._busy_counter = 0
            self._consume_counter = 0
            self._tick_call_counter = 0

        def subscribe(self, dst=None):
            if isinstance(dst, QueueSystem.NodeBase):
                self._consumers.append(dst)

        def consume(self):
            if self._consume_counter >= len(self._consumers):   # <= 0:
                raise Exception("Too many consume call for node")
            self._consume_counter += 1  # -=1

        def tick(self):
            if self._tick_call_counter == 0:
                self.run()
                for parent in self._sources:
                    parent.tick()
                self._consume_counter = 0   # len(self._consumers)
                self._tick_counter += 1
            self._tick_call_counter += 1
            if self._tick_call_counter == max(len(self._consumers), 1):
                self._tick_call_counter = 0

        def run(self):
            raise NotImplementedError
        
        def get_busy_rate(self):
            return self._busy_counter / self._tick_counter
        
        def get_tick_counter(self):
            return self._tick_counter

    class Source(NodeBase):
        _static = None
        _blocking = None
        _gen_prob = None
        _gen_rate = None
        _gen_counter = None
        _blocked = None

        _generated_value = None
        _generated_counter = None
        _discarded_counter = None

        def __init__(self, static: bool = False, blocking: bool = False, **kwargs):
            super().__init__(None)
            self._static = static
            if static:
                self._gen_rate = kwargs.get('gen_rate')
                if type(self._gen_rate) is not int:
                    raise ValueError
                self._gen_counter = 0
            else:
                self._gen_prob = kwargs.get('gen_prob')
            self._blocking = blocking
            self._discarded_counter = 0
            self._generated_counter = 0
            self._blocked = False

        def run(self):
            if self._generated_value is not None:
                if self._blocking:
                    self._blocked = True
                else:
                    self._discarded_counter += 1
            if not self._blocked:
                self._busy_counter += 1
                if self._static:
                    self._gen_counter += 1
                    if self._gen_counter >= self._gen_rate:
                        # TODO: generate new Token
                        self._generated_value = QueueSystem.Token()
                        self._generated_counter += 1
                        self._gen_counter = 0
                else:
                    if Generator.get_instance().next() < self._gen_prob:
                        # TODO: generate new Token
                        self._generated_value = QueueSystem.Token()
                        self._generated_counter += 1

        def consume(self):
            super().consume()
            ret_value = self._generated_value
            self._generated_value = None
            return ret_value
        
        def get_discard_rate(self):
            return self._discarded_counter / self._generated_counter

        def get_generation_rate(self):
            return self._generated_counter / self._busy_counter

    class Queue(NodeBase):
        _capacity = None

        _queue = None
        _overall_counter = None
        _full_counter = None

        def __init__(self, src, capacity: int = 2):
            super().__init__(src)
            self._capacity = capacity
            self._overall_counter = 0
            self._full_counter = 0
            self._queue = queue.Queue(capacity)
            self.consume_limit = 0

        def run(self):
            for parent in self._sources:
                if not self._queue.full():
                    temp = parent.consume()
                    if temp is not None:
                        self._queue.put(temp, False)
                else:
                    break

            self._overall_counter += self._queue.qsize()
            if not self._queue.empty():
                self._busy_counter += 1
                if self._queue.full():
                    self._full_counter += 1

        def consume(self):
            super().consume()
            if not self._queue.empty():
                return self._queue.get(False)
            return None

        def get_full_rate(self):
            return self._full_counter / self._busy_counter

        def get_overall_wait(self):
            return self._overall_counter / self._busy_counter

    class Server(NodeBase):
        _blocking = None
        _serve_prob = None
        _blocked = None

        _served_value = None
        _served_counter = None
        _discarded_b_counter = None
        _discarded_s_counter = None

        def __init__(self, src, serve_prob, blocking=False):
            super().__init__(src)
            self._blocking = blocking
            self._serve_prob = serve_prob
            self._blocked = False
            self._discarded_b_counter = 0
            self._discarded_s_counter = 0
            self._served_counter = 0

        def run(self):
            if self._served_value is not None:
                if self._blocking:
                    self._blocked = True
                else:
                    self._discarded_b_counter += 1
            if not self._blocked:
                self._busy_counter += 1
                for parent in self._sources:
                    temp = parent.consume()
                    if temp is not None:
                        if Generator.get_instance().next() < self._serve_prob:
                            self._served_value = temp
                            self._served_counter += 1
                            break
                        else:
                            self._discarded_s_counter += 1

        def consume(self):
            super().consume()
            ret_value = self._served_value
            self._served_value = None
            return ret_value

        def get_serve_rate(self):
            return self._served_counter / self._busy_counter

        def get_service_discard_rate(self):
            return self._discarded_s_counter / self._busy_counter

        def get_block_discard_rate(self):
            return self._discarded_b_counter / self._busy_counter


class Modeler:

    pass


def lab():
    Generator.initialize(102191, 203563, 131)

    params = [2, 1, 0.55, 0.5]
    _labels = [f"Generator rate: ", f"Queue capacity: ",
               f"Server 1 discard probability: ", f"Server 2 discard probability: "]
    _types = [int, int, float, float]
    for i in range(len(params)):
        try:
            temp = _types[i](input(_labels[i]))
            params[i] = temp if temp != 0 else params[i]
        except ValueError:
            continue

    nodes = []
    nodes.append(QueueSystem.Source(True, True, gen_rate=params[0]))
    nodes.append(QueueSystem.Queue(nodes[0], params[1]))
    nodes.append(QueueSystem.Server(nodes[1], 1 - params[2], False))
    nodes.append(QueueSystem.Server(nodes[2], 1 - params[3]))

    for _ in range(int(10e4)):
        nodes[3].tick()

    print(f"\n\tQueue system statistics: ")
    print(f"S1 busy rate: {nodes[0].get_busy_rate()}; \tS1 generation rate: {nodes[0].get_generation_rate()}")
    print(f"Q1 busy rate: {nodes[1].get_busy_rate()}; \tQ1 overall wait duration: {nodes[1].get_overall_wait()}; "
          f"\tQ1 full queue rate: {nodes[1].get_full_rate()}")
    print(f"C1 busy rate: {nodes[2].get_busy_rate()}; \tC1 serve rate: {nodes[2].get_serve_rate()}; "
          f"\tC1 block discard rate: {nodes[2].get_block_discard_rate()}; "
          f"\tC1 service discard rate: {nodes[2].get_service_discard_rate()}")
    print(f"C2 busy rate: {nodes[3].get_busy_rate()}; \tC2 serve rate: {nodes[3].get_serve_rate()}; "
          f"\tC2 block discard rate: {nodes[3].get_block_discard_rate()}; "
          f"\tC2 service discard rate: {nodes[3].get_service_discard_rate()}")
    print(f"Total queue system clock count: {nodes[3].get_tick_counter()}, {nodes[0].get_tick_counter()}")


if __name__ == '__main__':
    while True:
        lab()
        print(f"Quit? [y]/n")
        if input() != "n":
            break
