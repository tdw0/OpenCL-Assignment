kernel void filter_r(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	int image_size = get_global_size(0)/3; //each image consists of 3 colour channels
	int colour_channel = id / image_size; // 0 - red, 1 - green, 2 - blue

	//this is just a copy operation, modify to filter out the individual colour channels
	B[id] = A[id];
}

//simple ND identity kernel
kernel void identityND(global const uchar* A, global uchar* B) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	B[id] = A[id];
}

//2D averaging filter
kernel void avg_filterND(global const uchar* A, global uchar* B) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	uint result = 0;

	//simple boundary handling - just copy the original pixel
	if ((x == 0) || (x == width-1) || (y == 0) || (y == height-1)) {
		result = A[id];	
	} else {
		for (int i = (x-1); i <= (x+1); i++)
		for (int j = (y-1); j <= (y+1); j++) 
			result += A[i + j*width + c*image_size];

		result /= 9;
	}

	B[id] = (uchar)result;
}

//2D 3x3 convolution kernel
kernel void convolutionND(global const uchar* A, global uchar* B, constant float* mask) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	float result = 0;

	//simple boundary handling - just copy the original pixel
	if ((x == 0) || (x == width-1) || (y == 0) || (y == height-1)) {
		result = A[id];	
	} else {
		for (int i = (x-1); i <= (x+1); i++)
		for (int j = (y-1); j <= (y+1); j++) 
			result += A[i + j*width + c*image_size]*mask[i-(x-1) + j-(y-1)];
	}

	B[id] = (uchar)result;
}

//a very simple histogram implementation
kernel void hist_simple(global const int* A, global int* H) { 
	int id = get_global_id(0);

	//assumes that H has been initialised to 0
	int bin_index = A[id];//take value as a bin index

	atomic_inc(&H[bin_index]);//serial operation, not very efficient!
}


//a very simple histogram implementation
kernel void ghetto_hist(global const uchar* image, global int* histogram, const int bin_size) {
	int id = get_global_id(0);

	//assumes that H has been initialised to 0
	int bin_index = image[id]/bin_size;//take pixel value as a bin index

	atomic_inc(&histogram[bin_index]);//serial operation, not very efficient!
}


//simple exclusive serial scan based on atomic operations - sufficient for small number of elements
kernel void cum_hist(global int* C, global int* H, const int bin_size) {
	int id = get_global_id(0);
	int N = get_global_size(0);

	//for (int i = id + 1; i < N && id < N; i++)
	//	atomic_add(&C[id], H[i] );
	for (int i = 1; i < id && id < N; i++)
		atomic_add( &C[id/bin_size], H[i] );
}

//- Input: CH, Output : LUT
//- Your kernel should modify CH so that the maximum value is 255 (multiply each bin of CH by 255 / total_pixels).
//- Check : copy results back to the host and check the values.Is the last entry = 255 ?
kernel void lookup_table(global int* cum_hist, global int* LUT, const int num_pixels, const int bin_size) {
	int id = get_global_id(0);
	LUT[id] = cum_hist[id] * 255 / num_pixels / bin_size;
}

//- Input: image, LUT, Output: output_image.
//- Your kernel should assign for each pixel the output value from the LUT : output_pixel_value = LUT[input_pixel_value].
kernel void re_projection(global const uchar* in_image, global uchar* out_image, global int* LUT, const int bin_size) {
	int id = get_global_id(0);
	out_image[id] = LUT[ in_image[id] / bin_size ];
}
