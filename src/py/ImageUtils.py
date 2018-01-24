import pyopencl as cl

class ImageUtils():
    def __init__(self, context, path="src/opencl/ImageUtils.cl"):
        file = open(path,"r")
        code = file.read()
        self.prg = cl.Program(context, code).build()
        self.white_balance_kernel = self.prg.white_balance
        self.sample_image_kernel = self.prg.sample_image

    def sample_image(self, queue, image, sampled,wait_for=None):
        self.sample_image_kernel.set_args(image.gpu_buffer, sampled.gpu_buffer, image.shape_gpu, sampled.shape_gpu)
        return cl.enqueue_nd_range_kernel(queue, self.sample_image_kernel, (sampled.shape[0],sampled.shape[1]), None, wait_for=wait_for)

    def white_balance(self, queue, lms, result, factors, white,wait_for=None):
        self.white_balance_kernel.set_args(lms.gpu_buffer, result.gpu_buffer, lms.shape_gpu, factors.gpu_buffer, white.gpu_buffer)
        return cl.enqueue_nd_range_kernel(queue, self.white_balance_kernel, (lms.shape[0],lms.shape[1]), None, wait_for=wait_for)

