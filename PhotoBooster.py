import pyopencl as cl
import imageio
import numpy as np
from src.py.ColorConversion import ColorConversion
from src.py.Util import GPU_Buffer, GPU_Image, Stopwatch
import os
import sys, getopt



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
    sampled_shape = (np.int(rgb_gpu.shape[0] / sampling), np.int(rgb_gpu.shape[1] / sampling), rgb_gpu.shape[2])
    sampled = np.zeros(sampled_shape, np.float32).flatten()
    tmp1_gpu = GPU_Image(context, np.empty_like(rgb_gpu.cpu_buffer), rgb_gpu.shape)
    tmp2_gpu = GPU_Image(context, np.empty_like(rgb_gpu.cpu_buffer), rgb_gpu.shape)
    sampled_gpu = GPU_Image(context, sampled, sampled_shape)

    # Reference white D65 in CAT02
    white_gpu = GPU_Buffer(context, np.float32([94.92728, 103.53711, 108.73741]))

    # Convert RGB to CAT02
    e1 = color_conversion.rgb2lms(queue, rgb_gpu, tmp1_gpu, wait_for=wait)

    # Sample CAT02 image down to estimate white balance factors on a much smaller image
    e2 = color_conversion.sample_image(queue, tmp1_gpu, sampled_gpu, [e1])
    lms = sampled_gpu.copy_buffer_from_gpu(queue, [e2]).reshape(sampled_shape)
    white_balance_factor = get_white_balance_factors(lms, f1, f2)
    white_balance_factor_gpu = GPU_Buffer(context, white_balance_factor)

    # Perform white balance
    e1 = color_conversion.white_balance(queue, tmp1_gpu, tmp2_gpu, white_balance_factor_gpu, white_gpu)

    # Convert CAT02 to RGB
    e2 = color_conversion.lms2rgb(queue, tmp2_gpu, tmp1_gpu, [e1])

    return tmp1_gpu, e2

def stretch_saturation(context, queue, rgb_gpu, no_limit_saturation_boost=True, sampling=8.0, f1=0.0, f2=0.98, wait=None):

    color_conversion = ColorConversion(context)
    sampled_shape = (np.int(rgb_gpu.shape[0] / sampling), np.int(rgb_gpu.shape[1] / sampling), rgb_gpu.shape[2])
    sampled = np.zeros(sampled_shape, np.float32).flatten()
    tmp1_gpu = GPU_Image(context, np.empty_like(rgb_gpu.cpu_buffer), rgb_gpu.shape)
    tmp2_gpu = GPU_Image(context, np.empty_like(rgb_gpu.cpu_buffer), rgb_gpu.shape)
    sampled_gpu = GPU_Image(context, sampled, sampled_shape)

    if no_limit_saturation_boost:
        sat_gpu = GPU_Buffer(context, np.float32([1.0, 1.3, 1.0]))
    else:
        sat_gpu = GPU_Buffer(context, np.float32([1.0, 1.0, 1.0]))

    e1 = color_conversion.rgb2hsi(queue, rgb_gpu, tmp1_gpu, wait_for=wait)
    e2 = color_conversion.sample_image(queue, tmp1_gpu, sampled_gpu, [e1])

    hsi = sampled_gpu.copy_buffer_from_gpu(queue, [e2]).reshape(sampled_shape)
    sat_balance = get_white_balance_factors(hsi, f1, f2)
    sat_balance[0] = 0.0
    sat_balance[1] = 1.0
    sat_balance[4] = 0.0
    sat_balance[5] = 1.0
    sat_balance_gpu = GPU_Buffer(context, sat_balance)

    e1 = color_conversion.white_balance(queue, tmp1_gpu, tmp2_gpu, sat_balance_gpu, sat_gpu)

    e2 = color_conversion.hsi2rgb(queue, tmp2_gpu, tmp1_gpu, [e1])

    return tmp1_gpu, e2

def bootstrap_all_files_in_folder(context, queue, input_folder, output_folder, no_saturation_boost=False, no_limit_saturation_boost=True):
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

                    white_balanced_gpu, e1 = white_balance(context, queue, rgb_gpu)
                    sw.check("White balance")

                    if no_saturation_boost:
                        res = white_balanced_gpu.copy_buffer_from_gpu(queue, [e1])
                    else:
                        sat_stretched_gpu, e2 = stretch_saturation(context, queue, white_balanced_gpu, no_limit_saturation_boost, wait=[e1])
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
    try:
        myopts, args = getopt.getopt(sys.argv[1:], "i:")
    except getopt.GetoptError as e:
        print(str(e))
        print("Usage: %s -i input_folder " % sys.argv[0])
        sys.exit(2)

    for o, a in myopts:
        if o == '-i':
            input_folder = a

    return input_folder


if __name__ == "__main__":

    # Use the line below for development
    # input_folder = "D:\Test"
    input_folder = get_input_folder_from_args()

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

    bootstrap_all_files_in_folder(ctx, queue, input_folder, output_folder, no_saturation_boost, no_limit_saturation_boost)



