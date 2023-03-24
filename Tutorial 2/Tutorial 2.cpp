#include <iostream>
#include <vector>

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
	string image_filename = "test_large(1).pgm";

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
		CImgDisplay disp_input(image_input,"input");

		//a 3x3 convolution mask implementing an averaging filter
		std::vector<float> convolution_mask = { 1.f / 9, 1.f / 9, 1.f / 9,
												1.f / 9, 1.f / 9, 1.f / 9,
												1.f / 9, 1.f / 9, 1.f / 9 };

		//Part 3 - host operations
		//3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context);

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

		typedef int mytype;
		std::vector<mytype> H(256);
		size_t histsize = H.size() * sizeof(mytype);


		//device - buffers
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size()); //should be the same as input image
		cl::Buffer dev_histogram_output(context, CL_MEM_READ_WRITE, histsize);
		cl::Buffer dev_cumulative_histogram_output(context, CL_MEM_READ_WRITE, histsize);
		cl::Buffer dev_LUT_output(context, CL_MEM_READ_WRITE, histsize);
//		cl::Buffer dev_convolution_mask(context, CL_MEM_READ_ONLY, convolution_mask.size()*sizeof(float));

		//4.1 Copy images to device memory
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);
//		queue.enqueueWriteBuffer(dev_convolution_mask, CL_TRUE, 0, convolution_mask.size()*sizeof(float), &convolution_mask[0]);

		//4.2 Setup and execute the kernel (i.e. device code)
		cl::Kernel kernel = cl::Kernel(program, "hist_simple");
		kernel.setArg(0, dev_image_input); 
		kernel.setArg(1, dev_histogram_output); 
//		kernel.setArg(2, dev_convolution_mask);

		cl::Event prof_event;

		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &prof_event);

		vector<unsigned char> output_buffer(image_input.size());
		//4.3 Copy the result from device to host
		//queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);
		queue.enqueueReadBuffer(dev_histogram_output, CL_TRUE, 0, histsize, &H[0]);
		std::cout << "Histogram: " << H << std::endl;

		std::vector<mytype> CH(256);

		queue.enqueueFillBuffer(dev_cumulative_histogram_output, 0, 0, histsize);

		//The second kernel call plots a cumulative histogram of the total pixels in the picture across pixel values 0-255, so by 255, all pixels have been counted
		cl::Kernel kernel_hist_cum = cl::Kernel(program, "hist_cum");
		kernel_hist_cum.setArg(0, dev_histogram_output);
		kernel_hist_cum.setArg(1, dev_cumulative_histogram_output);

		cl::Event prof_CH;

		queue.enqueueNDRangeKernel(kernel_hist_cum, cl::NullRange, cl::NDRange(histsize), cl::NullRange, NULL, &prof_CH);
		queue.enqueueReadBuffer(dev_cumulative_histogram_output, CL_TRUE, 0, histsize, &CH[0]);
		std::cout << "cumulative Histogram: " << CH << std::endl;

		std::vector<mytype> LUT(256);

		queue.enqueueFillBuffer(dev_LUT_output, 0, 0, histsize);

		//The third kernel call creates a new histogram that will serve as a look up table of the new pixel vales. It does this by normalising the cumulative histogram, essentially decreasing the value of the pixels to increase the contrast
		cl::Kernel kernel_LUT = cl::Kernel(program, "normalise");
		kernel_LUT.setArg(0, dev_cumulative_histogram_output);
		kernel_LUT.setArg(1, dev_LUT_output);

		cl::Event prof_event3;

		queue.enqueueNDRangeKernel(kernel_LUT, cl::NullRange, cl::NDRange(histsize), cl::NullRange, NULL, &prof_event3);
		queue.enqueueReadBuffer(dev_LUT_output, CL_TRUE, 0, histsize, &LUT[0]);
		std::cout << "LUT: " << LUT << std::endl;

		cl::Kernel kernel_ReProject = cl::Kernel(program, "back_proj");
		kernel_ReProject.setArg(0, dev_image_input);
		kernel_ReProject.setArg(1, dev_LUT_output);
		kernel_ReProject.setArg(2, dev_image_output);

		cl::Event prof_event4;

		//The values from each histogram are printed, along with the kernel execution times and memory transfer of each kernel.

		queue.enqueueNDRangeKernel(kernel_ReProject, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &prof_event4);
		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);

		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		CImgDisplay disp_output(output_image,"output");

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
