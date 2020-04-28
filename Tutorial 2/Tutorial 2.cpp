#include <iostream>
#include <vector>
#include <iomanip>

#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test.pgm";

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		CImg<unsigned char> image_input(image_filename.c_str());
		CImgDisplay disp_input(image_input, "input");
		int total_pixels = image_input.size();

		//Part 3 - host operations
		//3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//3.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//Part 4 - device operations

		//a 256 bin histogram
		const int max_img_val = 256;
		const int bin_size = 2;
		const int num_bins = max_img_val / bin_size;
		std::vector<int> hist(num_bins);
		
		//cumulative histogram
		std::vector<int> cum_hist(num_bins);

		//Look-up table
		std::vector<int> LUT(num_bins);

		//device - buffers
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size()); //should be the same as input image
		cl::Buffer dev_hist(context, CL_MEM_READ_WRITE, hist.size()*sizeof(int));
		cl::Buffer dev_cum_hist(context, CL_MEM_READ_WRITE, cum_hist.size()*sizeof(int));
		cl::Buffer dev_lookup_table(context, CL_MEM_READ_WRITE, LUT.size()*sizeof(int));

		//4.1 Copy images to device memory
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);
		queue.enqueueFillBuffer(dev_hist, (int)0, 0, hist.size()*sizeof(int)); //zero the H buffer
		//queue.enqueueWriteBuffer(dev_hist, CL_TRUE, 0, hist.size()*sizeof(int), &hist[0]);

		//4.2 Setup and execute the kernel (i.e. device code)
		cl::Kernel hist_kernel = cl::Kernel(program, "faster_hist");
		cl::Event kernel_event;
		cl::Event mem_event;
		hist_kernel.setArg(0, dev_image_input);
		//kernel.setArg(1, dev_image_output);
		hist_kernel.setArg(1, dev_hist);
		hist_kernel.setArg(2, bin_size);
		hist_kernel.setArg(3, num_bins);
		int px = image_input.size();
		hist_kernel.setArg(4, px);
		queue.enqueueNDRangeKernel(hist_kernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &kernel_event);

		vector<unsigned char> output_buffer(image_input.size());

		//4.3 Copy the result from device to host
		//queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);
		queue.enqueueReadBuffer(dev_hist, CL_TRUE, 0, sizeof(int)*hist.size(), hist.data(), NULL, &mem_event);

		//output kernel exec time
		std::cout << "Kernel execution time [ns]:" <<
			kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;		//output mem xfer time
		std::cout << "Memory transfer time [ns]:" <<
			mem_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			mem_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		//Setup kernel for Cum_histogram
		cl::Kernel cum_hist_kernel = cl::Kernel(program, "cum_hist");
		cum_hist_kernel.setArg(0, dev_cum_hist);
		cum_hist_kernel.setArg(1, dev_hist);
		cum_hist_kernel.setArg(2, bin_size);
		queue.enqueueNDRangeKernel(cum_hist_kernel, cl::NullRange, cl::NDRange(hist.size()), cl::NullRange);

		//copy the result from device to host
		queue.enqueueReadBuffer(dev_cum_hist, CL_TRUE, 0, sizeof(int)*cum_hist.size(), cum_hist.data());

		//Setup kernel for LUT
		cl::Kernel LUT_kernel = cl::Kernel(program, "lookup_table");
		LUT_kernel.setArg(0, dev_cum_hist);
		LUT_kernel.setArg(1, dev_lookup_table);
		LUT_kernel.setArg(2, total_pixels);
		LUT_kernel.setArg(3, bin_size);
		queue.enqueueNDRangeKernel(LUT_kernel, cl::NullRange, cl::NDRange(cum_hist.size()), cl::NullRange);

		//copy the result from device to host
		queue.enqueueReadBuffer(dev_lookup_table, CL_TRUE, 0, sizeof(int)*LUT.size(), LUT.data());

		// print LUT
		cout << "------- LUT -------" << endl;
		for (auto i : LUT) {
			cout << i << endl;
		}


		//setup kernel for re-projection
		cl::Kernel re_proj = cl::Kernel(program, "re_projection");
		re_proj.setArg(0, dev_image_input);
		re_proj.setArg(1, dev_image_output);
		re_proj.setArg(2, dev_lookup_table);
		re_proj.setArg(3, bin_size);
		queue.enqueueNDRangeKernel(re_proj, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange);

		//copy the result from device to host
		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);

		if (LUT.size() != max_img_val / bin_size) {
			cout << "LUT.size() != max_img_val / bin_size!!!" << endl;
			cout << "LUT.size()  = " << LUT.size() << endl;
			cout << "max_img_val = " << max_img_val << endl;
			cout << "bin_size    = " << bin_size << endl;
		}
		// draw histogram
		cout << "------- Histogram -------" << endl;
		for (int bin = 0; bin < hist.size(); bin++) {
			cout << setw(7) << bin << " | ";
			const auto val = hist.at(bin);
			for( int i = 0; i < (val / 100.0)/bin_size; i++ )
				cout << "#" ;

			cout << endl;
		}

		////draw cumulative histogram
		cout << "------- Cumulative Histogram -------" << endl;
		for (int bin = 0; bin < cum_hist.size(); bin++) {
			cout << setw(7) << bin << " | ";
			const auto val = cum_hist.at(bin);
			cout << val;
			//for( int i = 0; i < (val / 100.0)/30; i++ )
			//	cout << "#" ;

			cout << endl;
		}



		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		CImgDisplay disp_output(output_image, "output");

		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
			disp_input.wait(1);
			disp_output.wait(1);
		}

	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}