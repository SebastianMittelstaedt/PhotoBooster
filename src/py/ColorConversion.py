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

    @staticmethod
    def rgb2lms_cpu(rgb):
        r = rgb[0] / 255.0
        g = rgb[1] / 255.0
        b = rgb[2] / 255.0

        if r > 0.04045:
            r= pow(((r + 0.055) / 1.055), 2.4)
        else:
            r = r / 12.92

        if g > 0.04045:
            g= pow(((g + 0.055) / 1.055), 2.4)
        else:
            g = g / 12.92

        if b > 0.04045:
            b = pow(((b + 0.055) / 1.055), 2.4)
        else:
            b = b / 12.92

        r *= 100.0
        g *= 100.0
        b *= 100.0

        # sRGB, Illuminant = D65
        x = (r * 0.4124564) + (g * 0.3575761) + (b * 0.1804375)
        y = (r * 0.2126729) + (g * 0.7151522) + (b * 0.0721750)
        z = (r * 0.0193339) + (g * 0.1191920) + (b * 0.9503041)

        l = 0.7328* x + 0.4296 * y - 0.1624 * z
        m = -0.7036* x + 1.6975* y + 0.0061* z
        s = 0.0030* x + 0.0136* y + 0.9834* z

        return [l,m,s]

    @staticmethod
    def rgb2xyz_cpu(rgb):
        r = rgb[0] / 255.0
        g = rgb[1] / 255.0
        b = rgb[2] / 255.0

        if r > 0.04045:
            r = pow(((r + 0.055) / 1.055), 2.4)
        else:
            r = r / 12.92

        if g > 0.04045:
            g = pow(((g + 0.055) / 1.055), 2.4)
        else:
            g = g / 12.92

        if b > 0.04045:
            b = pow(((b + 0.055) / 1.055), 2.4)
        else:
            b = b / 12.92

        r *= 100.0
        g *= 100.0
        b *= 100.0

        # sRGB, Illuminant = D65
        x = (r * 0.4124564) + (g * 0.3575761) + (b * 0.1804375)
        y = (r * 0.2126729) + (g * 0.7151522) + (b * 0.0721750)
        z = (r * 0.0193339) + (g * 0.1191920) + (b * 0.9503041)

        return [x,y,z]

    @staticmethod
    def xyz2rgb_cpu(xyz):

        x = xyz[0] / 100.0
        y = xyz[1] / 100.0
        z = xyz[2] / 100.0

        # sRGB, Illuminant = D65
        r = (x * 3.2404542) + (y * -1.5371385) + (z * -0.4985314)
        g = (x * -0.9692660) + (y * 1.8760108) + (z * 0.0415560)
        b = (x * 0.0556434) + (y * -0.2040259) + (z * 1.0572252)

        if r > 0.0031308:
            r = 1.055 * (pow(r, (1.0 / 2.4))) - 0.055
        else:
            r = 12.92 * r

        if g > 0.0031308:
            g = 1.055 * (pow(g, (1.0 / 2.4))) - 0.055
        else:
            g = 12.92 * g

        if b > 0.0031308:
            b = 1.055 * (pow(b, (1.0 / 2.4))) - 0.055
        else:
            b = 12.92 * b

        if r > 1.0:
            r = 255.0
        elif r < 0.0:
            r = 0.0
        else:
            r = r * 255.0

        if g > 1.0:
            g = 255.0
        elif g < 0.0:
            g = 0.0
        else:
            g = g * 255.0

        if b > 1.0:
            b = 255.0
        elif b < 0.0:
            b = 0.0
        else:
            b = b * 255.0

        return [r, g, b]

    @staticmethod
    def lms2rgb_cpu(lms):
        l = lms[0]
        m = lms[1]
        s = lms[2]

        x = 1.096124* l - 0.278869* m + 0.182745* s
        y = 0.454369* l + 0.473533* m + 0.072098* s
        z = -0.009628* l - 0.005698* m + 1.015326* s

        x = x / 100.0
        y = y / 100.0
        z = z / 100.0

        # sRGB, Illuminant = D65
        r = (x * 3.2404542) + (y * -1.5371385) + (z * -0.4985314)
        g = (x * -0.9692660) + (y * 1.8760108) + (z * 0.0415560)
        b = (x * 0.0556434) + (y * -0.2040259) + (z * 1.0572252)

        if r > 0.0031308:
            r = 1.055 * (pow(r, (1.0 / 2.4))) - 0.055
        else:
            r = 12.92*r

        if g > 0.0031308:
            g = 1.055 * (pow(g, (1.0 / 2.4))) - 0.055
        else:
            g = 12.92*g

        if b > 0.0031308:
            b = 1.055 * (pow(b, (1.0 / 2.4))) - 0.055
        else:
            b = 12.92*b

        if r > 1.0:
            r = 255.0
        elif r < 0.0:
            r = 0.0
        else:
            r = r*255.0

        if g > 1.0:
            g = 255.0
        elif g < 0.0:
            g = 0.0
        else:
            g = g*255.0

        if b > 1.0:
            b = 255.0
        elif b < 0.0:
            b = 0.0
        else:
            b = b*255.0

        return [r,g,b]

if __name__ == "__main__":
    # Whitepoints from https://de.mathworks.com/help/images/ref/whitepoint.html
    # Conversion to CAT02 for chromatic adaption
    xyz_E = [98.07, 100.00, 118.22] # Cold
    xyz_D65 = [95.04, 100.00, 108.88] # Normal - cold
    xyz_D55 = [95.68, 100.00, 92.14] # Normal - warm
    xyz_50 = [96.42, 100.00, 82.51] # Warm
    xyz_A = [109.85, 100.00, 35.58] # Extra warm

    rgb = ColorConversion.xyz2rgb_cpu(xyz_D55)
    lms = ColorConversion.rgb2lms_cpu(rgb)
    print(rgb)
    print(lms)
