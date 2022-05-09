from multiprocessing import Process, Queue
import traceback


class WrappedException:
    def __init__(self, e, trace):
        self.e = e
        self.trace = trace


# Executes the given func but aborts if it runs for longer than timeout seconds.
def exec_timeout(timeout, func, *args, **kwargs):
    retQueue = Queue()

    def wrap_call():
        try:
            retQueue.put(func(*args, **kwargs))
        except Exception as e:
            retQueue.put(WrappedException(e, traceback.format_exc()))

    p = Process(target=wrap_call)
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join(1)
        if p.is_alive():
            p.kill()
            p.join()
        raise TimeoutError("Function did not complete within the timeout time")

    ret = retQueue.get()
    if isinstance(ret, WrappedException):
        print(ret.trace)
        raise ret.e
    return ret
