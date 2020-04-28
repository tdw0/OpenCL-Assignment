//a very simple histogram implementation
kernel void ghetto_hist(global const uchar* image, global int* histogram, const int bin_size, const int num_bins, const int pixels) {
	int id = get_global_id(0);

	//assumes that H has been initialised to 0
	int bin_index = image[id] / bin_size; //take pixel value as a bin index

	atomic_inc(&histogram[bin_index]); //serial operation, not very efficient!
}

//a parallelised histogram implementation
kernel void faster_hist(global const uchar* image, global int* histogram, const int bin_size, const int num_bins, const int pixels) {
	__local int localHist[256];
	int l_id = get_local_id(0);
	int g_id = get_global_id(0);

	// init local hist to 0
	for (int i = l_id; i < num_bins; i += get_local_size(0)) {
		localHist[i] = 0;
	}

	// wait until all work-items in the work group have completed stores
	//barrier(CLK_LOCAL_MEM_FENCE);

	// compute local histogram
	for (int i = g_id; i < pixels; i += get_global_size(0)) {
		atomic_add(&localHist[image[i]], 1);
	}

	//barrier(CLK_LOCAL_MEM_FENCE);

	int bin_index = image[g_id] / bin_size; //take pixel value as a bin index
	
	for (int i = l_id; i < 256; i += get_local_size(0)) {
		atomic_add(&histogram[bin_index], localHist[i]);
	}
	
	//assumes that H has been initialised to 0
	//int bin_index = image[id] / bin_size; //take pixel value as a bin index

	//atomic_inc(&histogram[bin_index]); //serial operation, not very efficient!
}

//simple exclusive serial scan based on atomic operations - sufficient for small number of elements
kernel void cum_hist(global int* C, global int* H, const int bin_size) {
	int id = get_global_id(0);
	int N = get_global_size(0);

	for (int i = 1; i < id && id < N; i++)
		atomic_add( &C[id], H[i] );
}

//- Input: CH, Output : LUT
//- Your kernel should modify CH so that the maximum value is 255 (multiply each bin of CH by 255 / total_pixels).
//- Check : copy results back to the host and check the values.Is the last entry = 255 ?
kernel void lookup_table(global int* cum_hist, global int* LUT, const int num_pixels, const int bin_size) {
	int id = get_global_id(0);
//	LUT[id] = cum_hist[id] * 255 / num_pixels / bin_size;
//	LUT[id] = ((double) cum_hist[id]) * 255 / cum_hist[255];
	LUT[id] = ((double)cum_hist[id]) * ((double)255 / num_pixels) / bin_size;
}

//- Input: image, LUT, Output: output_image.
//- Your kernel should assign for each pixel the output value from the LUT : output_pixel_value = LUT[input_pixel_value].
kernel void re_projection(global const uchar* in_image, global uchar* out_image, global int* LUT, const int bin_size) {
	int id = get_global_id(0);
	out_image[id] = LUT[ in_image[id] / bin_size ];
}
