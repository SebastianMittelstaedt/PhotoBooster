
__kernel void white_balance(__global const float* lms, __global float* result, __global const int* image_shape, __global const float* factors, __global const float* white)
{
    float r, g, b, x, y, z;
    int i = get_global_id(0);
    int j = get_global_id(1);
    int h = image_shape[1];
    int c = image_shape[2];
    int idx = i * h * c  + j * c;

	result[idx] = (lms[idx] - factors[0])/(factors[1]- factors[0]) * white[0];
	result[idx+1] = (lms[idx+1] - factors[2])/(factors[3]- factors[2]) * white[1];
	result[idx+2] = (lms[idx+2] - factors[4])/(factors[5]- factors[4]) * white[2];
}

__kernel void sample_image(__global const float* image, __global float* sampledImage, __global const int* image_shape, __global const int* sampled_shape)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int h = sampled_shape[1];
    int c = sampled_shape[2];
    int idx = x * h * c  + y * c;

    int originalX = (int)(0.5f+(float)x*(float)(image_shape[0]-1)/(float)(sampled_shape[0]-1));
    int originalY = (int)(0.5f+(float)y*(float)(image_shape[1]-1)/(float)(sampled_shape[1]-1));

    int idx_original = originalX * image_shape[1] * c + originalY * c;

    sampledImage[idx] = image[idx_original];
    sampledImage[idx+1] = image[idx_original+1];
    sampledImage[idx+2] = image[idx_original+2];
}
