import pyopencl as cl


class ColorConversion():
    def __init__(self, context, path="src/opencl/ColorConversion.cl"):
        file = open(path,"r")
        code = file.read()
        self.prg = cl.Program(context, code).build()
        self.rgb2lms_kernel = self.prg.rgb2lms
        self.lms2rgb_kernel = self.prg.lms2rgb
        self.rgb2hsi_kernel = self.prg.rgb2hsi
        self.hsi2rgb_kernel = self.prg.hsi2rgb

    def rgb2lms(self, queue, rgb, lms,wait_for=None):
        self.rgb2lms_kernel.set_args(rgb.gpu_buffer, lms.gpu_buffer, rgb.shape_gpu)
        return cl.enqueue_nd_range_kernel(queue, self.rgb2lms_kernel, (rgb.shape[0],rgb.shape[1]), None, wait_for=wait_for)

    def lms2rgb(self, queue, lms, rgb, wait_for=None):
        self.lms2rgb_kernel.set_args(lms.gpu_buffer, rgb.gpu_buffer, rgb.shape_gpu)
        return cl.enqueue_nd_range_kernel(queue, self.lms2rgb_kernel, (rgb.shape[0],rgb.shape[1]), None, wait_for=wait_for)

    def rgb2hsi(self, queue, rgb, hsi,wait_for=None):
        self.rgb2hsi_kernel.set_args(rgb.gpu_buffer, hsi.gpu_buffer, rgb.shape_gpu)
        return cl.enqueue_nd_range_kernel(queue, self.rgb2hsi_kernel, (rgb.shape[0],rgb.shape[1]), None, wait_for=wait_for)

    def hsi2rgb(self, queue, hsi, rgb,wait_for=None):
        self.hsi2rgb_kernel.set_args(hsi.gpu_buffer, rgb.gpu_buffer, rgb.shape_gpu)
        return cl.enqueue_nd_range_kernel(queue, self.hsi2rgb_kernel, (rgb.shape[0],rgb.shape[1]), None, wait_for=wait_for)