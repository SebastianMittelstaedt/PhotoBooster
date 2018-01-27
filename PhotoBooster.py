import pyopencl as cl
import imageio
import numpy as np
from src.py.ColorConversion import ColorConversion
from src.py.ImageUtils import ImageUtils
from src.py.Util import Stopwatch
from src.py.GPU_Buffers import GPU_Buffer, GPU_Image
import os
from scipy import signal
import argparse


def get_white_balance_factors(lms, f1, f2):
    l = np.sort(lms[:,:,0].flatten())
    m = np.sort(lms[:, :, 1].flatten())
    s = np.sort(lms[:, :, 2].flatten())

    l1 = l[np.int(0.5 + f1 * (l.shape[0] - 1))]
    m1 = m[np.int(0.5 + f1 * (l.shape[0] - 1))]
    s1 = s[np.int(0.5 + f1 * (l.shape[0] - 1))]

    l2 = l[np.int(0.5 + f2 * (l.shape[0] - 1))]
    m2 = m[np.int(0.5 + f2 * (l.shape[0] - 1))]
    s2 = s[np.int(0.5 + f2 * (l.shape[0] - 1))]

    return np.asarray([l1,l2,m1,m2,s1,s2])


def white_balance(context, queue, rgb_gpu, sampling=8.0, f1=0.01, f2=0.99, wait=None):
    """

    :param context:
    :param queue:
    :param image:
    :return:
    """
    color_conversion = ColorConversion(context)
    image_utils = ImageUtils(context)
    sampled_shape = (np.int(rgb_gpu.shape[0] / sampling), np.int(rgb_gpu.shape[1] / sampling), rgb_gpu.shape[2])
    sampled = np.zeros(sampled_shape, np.float32).flatten()
    tmp1_gpu = GPU_Image(context, np.empty_like(rgb_gpu.cpu_buffer), rgb_gpu.shape)
    tmp2_gpu = GPU_Image(context, np.empty_like(rgb_gpu.cpu_buffer), rgb_gpu.shape)
    #tmp3_gpu = GPU_Image(context, np.empty_like(rgb_gpu.cpu_buffer), rgb_gpu.shape)
    sampled_gpu = GPU_Image(context, sampled, sampled_shape)

    # Reference white D65 in CAT02
    white_gpu = GPU_Buffer(context, np.float32([94.92728, 103.53711, 108.73741]))

    # Convert RGB to CAT02
    e1 = color_conversion.rgb2lms(queue, rgb_gpu, tmp1_gpu, wait_for=wait)

    tmp3_gpu, e3 = remove_pixel_errors(context, queue, tmp1_gpu, [e1])


    # Sample CAT02 image down to estimate white balance factors on a much smaller image
    e2 = image_utils.sample_image(queue, tmp3_gpu, sampled_gpu, [e3])
    lms = sampled_gpu.copy_buffer_from_gpu(queue, [e2]).reshape(sampled_shape)
    white_balance_factor = get_white_balance_factors(lms, f1, f2)
    white_balance_factor_gpu = GPU_Buffer(context, white_balance_factor)

    # Perform white balance
    e1 = image_utils.white_balance(queue, tmp3_gpu, tmp2_gpu, white_balance_factor_gpu, white_gpu, [e2])

    # Convert CAT02 to RGB
    e2 = color_conversion.lms2rgb(queue, tmp2_gpu, tmp1_gpu, [e1])

    return tmp1_gpu, e2


def stretch_saturation(context, queue, rgb_gpu, no_limit_saturation_boost=True, sampling=8.0, f1=0.0, f2=0.98, wait=None):

    color_conversion = ColorConversion(context)
    image_utils = ImageUtils(context)
    sampled_shape = (np.int(rgb_gpu.shape[0] / sampling), np.int(rgb_gpu.shape[1] / sampling), rgb_gpu.shape[2])
    sampled = np.zeros(sampled_shape, np.float32).flatten()
    tmp1_gpu = GPU_Image(context, np.empty_like(rgb_gpu.cpu_buffer), rgb_gpu.shape)
    tmp2_gpu = GPU_Image(context, np.empty_like(rgb_gpu.cpu_buffer), rgb_gpu.shape)
    #tmp3_gpu = GPU_Image(context, np.empty_like(rgb_gpu.cpu_buffer), rgb_gpu.shape)
    sampled_gpu = GPU_Image(context, sampled, sampled_shape)

    if no_limit_saturation_boost:
        sat_gpu = GPU_Buffer(context, np.float32([1.0, 1.3, 1.0]))
    else:
        sat_gpu = GPU_Buffer(context, np.float32([1.0, 1.0, 1.0]))

    e1 = color_conversion.rgb2hsi(queue, rgb_gpu, tmp1_gpu, wait_for=wait)
    e2 = image_utils.sample_image(queue, tmp1_gpu, sampled_gpu, [e1])

    hsi = sampled_gpu.copy_buffer_from_gpu(queue, [e2]).reshape(sampled_shape)
    sat_balance = get_white_balance_factors(hsi, f1, f2)
    sat_balance[0] = 0.0
    sat_balance[1] = 1.0
    sat_balance[4] = 0.0
    sat_balance[5] = 1.0
    sat_balance_gpu = GPU_Buffer(context, sat_balance)

    e1 = image_utils.white_balance(queue, tmp1_gpu, tmp2_gpu, sat_balance_gpu, sat_gpu)

    tmp3_gpu, e3 = sharpen_image(context, queue, tmp2_gpu, [e1])

    e2 = color_conversion.hsi2rgb(queue, tmp3_gpu, tmp1_gpu, [e3])

    return tmp1_gpu, e2

def remove_pixel_errors(context, queue, lms_gpu, wait):
    image_utils = ImageUtils(context)
    tmp1_gpu = GPU_Image(context, np.empty_like(lms_gpu.cpu_buffer), lms_gpu.shape)
    tmp2_gpu = GPU_Image(context, np.empty_like(lms_gpu.cpu_buffer), lms_gpu.shape)

    gauss = generate_gauss_kernel(1)
    gauss_cpu = GPU_Buffer(context, gauss)

    e1 = image_utils.low_pass_x(queue, lms_gpu, tmp1_gpu, gauss_cpu, wait_for=wait)
    e2 = image_utils.low_pass_y(queue, tmp1_gpu, tmp2_gpu, gauss_cpu, wait_for=[e1])

    return tmp2_gpu, e2


def sharpen_image(context, queue, hsi_gpu, wait):
    image_utils = ImageUtils(context)
    tmp1_gpu = GPU_Image(context, np.empty_like(hsi_gpu.cpu_buffer), hsi_gpu.shape)
    tmp2_gpu = GPU_Image(context, np.empty_like(hsi_gpu.cpu_buffer), hsi_gpu.shape)

    gauss = generate_gauss_kernel(2)
    gauss_cpu = GPU_Buffer(context, gauss)

    e1 = image_utils.low_pass_x(queue, hsi_gpu, tmp1_gpu, gauss_cpu, wait_for=wait)
    e2 = image_utils.low_pass_y(queue, tmp1_gpu, tmp2_gpu, gauss_cpu, wait_for=[e1])
    e3 = image_utils.high_pass(queue,hsi_gpu,tmp2_gpu,tmp1_gpu,wait_for=[e2])

    return tmp1_gpu, e3


def bootstrap_all_files_in_folder2(context, queue, input_folder, output_folder, no_saturation_boost, no_limit_saturation_boost, white_balance_factor, saturation_boost_factor):
    sw = Stopwatch()

    with os.scandir(input_folder) as listOfEntries:
        for entry in listOfEntries:
            if entry.is_file():
                file = entry.name
                if file.endswith((".png", ".jpg", ".JPG", ".jpeg", ".bmp")):
                    print("Processing: %s" % (file))

                    sw.start()
                    image = np.asarray(imageio.imread(os.path.join(input_folder, file)).astype(np.float32))

                    # JPG cannot save alpha -> remove alpha channel....
                    if image.shape[2] == 4:
                        image = image[:,:,0:3]

                    image_shape = image.shape
                    image = image.flatten()
                    rgb_gpu = GPU_Image(context, image, image_shape)
                    sw.check("Loading")

                    hsi_gpu = GPU_Image(context, np.empty_like(rgb_gpu.cpu_buffer), rgb_gpu.shape)
                    tmp1_gpu = GPU_Image(context, np.empty_like(rgb_gpu.cpu_buffer), rgb_gpu.shape)
                    tmp2_gpu = GPU_Image(context, np.empty_like(rgb_gpu.cpu_buffer), rgb_gpu.shape)

                    image_utils = ImageUtils(context)
                    color_conversion = ColorConversion(context)

                    e0 = color_conversion.rgb2hsi(queue,rgb_gpu,hsi_gpu)

                    e1 = image_utils.low_pass_x(queue,hsi_gpu,tmp1_gpu, [e0])
                    e2 = image_utils.low_pass_y(queue, tmp1_gpu, tmp2_gpu, wait_for=[e1])
                    e3 = image_utils.high_pass(queue, hsi_gpu, tmp2_gpu, tmp1_gpu, wait_for=[e2])

                    e4 = color_conversion.hsi2rgb(queue, tmp1_gpu, tmp2_gpu, wait_for=[e3])

                    res = tmp2_gpu.copy_buffer_from_gpu(queue, [e4])
                    res = res.reshape(image_shape).astype(np.uint8)
                    sw.check("Loading from GPU")

                    # jpg is much faster than png, however, now images with alpha cannot be saved ....
                    imageio.imwrite(os.path.join(output_folder, file + ".JPG"), res, format="JPG")
                    sw.check("Writing file")

                    GPU_Buffer.release_all()
                    sw.end()




def bootstrap_all_files_in_folder(context, queue, input_folder, output_folder, no_saturation_boost, no_limit_saturation_boost, white_balance_factor, saturation_boost_factor):
    sw = Stopwatch()

    with os.scandir(input_folder) as listOfEntries:
        for entry in listOfEntries:
            if entry.is_file():
                file = entry.name
                if file.endswith((".png", ".jpg", ".JPG", ".jpeg", ".bmp")):
                    print("Processing: %s" % (file))

                    sw.start()
                    image = np.asarray(imageio.imread(os.path.join(input_folder, file)).astype(np.float32))

                    # JPG cannot save alpha -> remove alpha channel....
                    if image.shape[2] == 4:
                        image = image[:,:,0:3]

                    image_shape = image.shape
                    image = image.flatten()
                    rgb_gpu = GPU_Image(context, image, image_shape)
                    sw.check("Loading")

                    white_balanced_gpu, e1 = white_balance(context, queue, rgb_gpu, f1=1.0-white_balance_factor, f2=white_balance_factor)
                    sw.check("White balance")

                    if no_saturation_boost:
                        res = white_balanced_gpu.copy_buffer_from_gpu(queue, [e1])
                    else:
                        sat_stretched_gpu, e2 = stretch_saturation(context, queue, white_balanced_gpu, no_limit_saturation_boost, f1=0.0, f2=saturation_boost_factor, wait=[e1])
                        sw.check("stretch_saturation")
                        res = sat_stretched_gpu.copy_buffer_from_gpu(queue, [e2])

                    res = res.reshape(image_shape).astype(np.uint8)
                    sw.check("Loading from GPU")

                    # jpg is much faster than png, however, now images with alpha cannot be saved ....
                    imageio.imwrite(os.path.join(output_folder, file+".JPG"), res, format="JPG")
                    sw.check("Writing file")

                    GPU_Buffer.release_all()
                    sw.end()


def get_input_folder_from_args():
    parser = argparse.ArgumentParser(description='python cli')
    parser.add_argument("-i", "--input_folder", help="Input folder which is processed", required=True)
    parser.add_argument("-w", "--white_balance", help="White balance factor", required=False)
    parser.add_argument("-s", "--saturation_boost", help="Saturation boost factor", required=False)

    # parse input arguments
    args = parser.parse_args()

    input_folder = args.input_folder

    if not args.white_balance:
        args.white_balance = 0.99
    if not args.saturation_boost:
        args.saturation_boost = 0.98

    white_balance_factor = float(args.white_balance)
    saturation_boost_factor = float(args.saturation_boost)


    return input_folder, white_balance_factor, saturation_boost_factor


def generate_gauss_kernel(std):
    w = 4*std+1 # Covers >95% of kernel
    g = signal.gaussian(w, std=std)
    return g / g.sum()


if __name__ == "__main__":
    # Use the line below for development
    # input_folder = "D:\Test"
    input_folder, white_balance_factor, saturation_boost_factor = get_input_folder_from_args()

    ctx = cl.create_some_context(interactive=True)
    queue = cl.CommandQueue(ctx)

    output_folder = os.path.join(input_folder, "PhotoBooster")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("Do saturation boost?")
    no_saturation_boost = input("n/y") == "n"

    no_limit_saturation_boost = True
    if not no_saturation_boost:
        print("Limit saturation boost?")
        no_limit_saturation_boost = input("n/y") == "n"

    bootstrap_all_files_in_folder(ctx, queue, input_folder, output_folder, no_saturation_boost, no_limit_saturation_boost, white_balance_factor, saturation_boost_factor)



