//a very simple histogram implementation
kernel void ghetto_hist(global const uchar* image, global int* histogram, const int bin_size) {
	int id = get_global_id(0);

	//assumes that H has been initialised to 0
	int bin_index = image[id] / bin_size; //take pixel value as a bin index

	atomic_inc(&histogram[bin_index]); //serial operation, not very efficient!
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
