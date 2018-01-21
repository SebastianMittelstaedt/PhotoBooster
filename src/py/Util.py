import pyopencl as cl
import numpy as np
import time

mf = cl.mem_flags


class Stopwatch:
    def __init__(self):
        self.t1 = 0.0
        self.t2 = 0.0

    def start(self):
        self.t1 = time.time()
        self.t2 = self.t1

    def check(self, text):
        t = time.time()
        dt = t - self.t2
        print("\t %s took: %s" % (text, str(dt)))
        self.t2 = time.time()

    def end(self):
        t = time.time()
        dt = t - self.t1
        print("\t The whole process took: %s" % (str(dt)))


class GPU_Buffer:
    buffers = []

    def __init__(self, context, cpu_buffer):
        self.cpu_buffer = cpu_buffer
        self.gpu_buffer = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=cpu_buffer)
        GPU_Buffer.buffers.append(self.gpu_buffer)

    def copy_buffer_from_gpu(self, queue,  wait):
        cl.enqueue_copy(queue, self.cpu_buffer, self.gpu_buffer, wait_for=wait).wait()
        return self.cpu_buffer

    @staticmethod
    def release_all():
        for b in GPU_Buffer.buffers:
            b.release()
        GPU_Buffer.buffers = []

    @staticmethod
    def create_buffers(context, shape):
        cpu = np.zeros(shape, np.float32)
        return GPU_Buffer(context, cpu)


class GPU_Image(GPU_Buffer):
    def __init__(self, context, cpu_buffer, shape):
        super().__init__(context, cpu_buffer)
        self.shape = shape
        self.shape_gpu = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.int32([shape[0],shape[1],shape[2]]))
        GPU_Buffer.buffers.append(self.shape_gpu)

    @staticmethod
    def create_image(context, shape):
        cpu = np.zeros(shape, np.float32)
        return GPU_Image(context, cpu, shape)