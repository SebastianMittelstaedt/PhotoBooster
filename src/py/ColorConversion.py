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
        self.rgb2lmshsi_kernel = self.prg.rgb2lmshsi
        self.lmshsi2rgb_kernel = self.prg.lmshsi2rgb
        self.white_balance_kernel = self.prg.white_balance
        self.white_balance_half_point_kernel = self.prg.white_balance_half_point
        self.sample_image_kernel = self.prg.sample_image

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

    def sample_image(self, queue, image, sampled,wait_for=None):
        self.sample_image_kernel.set_args(image.gpu_buffer, sampled.gpu_buffer, image.shape_gpu, sampled.shape_gpu)
        return cl.enqueue_nd_range_kernel(queue, self.sample_image_kernel, (sampled.shape[0],sampled.shape[1]), None, wait_for=wait_for)

    def white_balance(self, queue, lms, result, factors, white,wait_for=None):
        self.white_balance_kernel.set_args(lms.gpu_buffer, result.gpu_buffer, lms.shape_gpu, factors.gpu_buffer, white.gpu_buffer)
        return cl.enqueue_nd_range_kernel(queue, self.white_balance_kernel, (lms.shape[0],lms.shape[1]), None, wait_for=wait_for)

    # def rgb2lmshsi(self, queue, rgb, hsi, shape, height, channels, white, wait_for=None):
    #     self.rgb2lmshsi_kernel.set_args(rgb, hsi, rgb.shape[1], rgb.shape[2], white)
    #     return cl.enqueue_nd_range_kernel(queue, self.rgb2lmshsi_kernel, (shape[0], shape[1]), None, wait_for=wait_for)
    #
    # def lmshsi2rgb(self, queue, hsi, rgb, shape, height, channels, white, wait_for=None):
    #     self.lmshsi2rgb_kernel.set_args(hsi, rgb, rgb.shape[1], rgb.shape[2], white)
    #     return cl.enqueue_nd_range_kernel(queue, self.lmshsi2rgb_kernel, (shape[0], shape[1]), None, wait_for=wait_for)









    def white_balance_half_point(self, queue, lms, result, shape, height, channels, factors, white,wait_for=None):
        self.white_balance_half_point_kernel.set_args(lms, result, height, channels, factors, white)
        return cl.enqueue_nd_range_kernel(queue, self.white_balance_half_point_kernel, (shape[0], shape[1]), None, wait_for=wait_for)




