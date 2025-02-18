#include "CL/cl.hpp"
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "utility.h"
// #include "error_handling.hpp"
using namespace aocl_utils;
using namespace std;
using namespace std::chrono;

#define NUM_CMD_QUEUES 16

void createKernel(string kernel_name)
{
    // error code returned from api calls
    cl_int err;
    // create kernel and check for errors
    kernels[kernel_name] = clCreateKernel(program, kernel_name.c_str(), &err);
    checkErr(err, __LINE__);
}

void initializeOpenCLParameters()
{
    cl_int err;
    if(!setCwdToExeDir())
	{
		exit(1);
    }

    // Get the OpenCL platform.
    platform = findPlatform("Intel(R) FPGA");
    if(platform == NULL)
	{
		std::cerr << "ERROR: FPGA platform not found." << std::endl;
		exit(1);
    }

    // Query the available OpenCL devices.
    scoped_array<cl_device_id> devices;
    cl_uint num_devices;

    devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));

    // We'll just use the first device.
    device = devices[0];

    // Display some device information.
    // display_device_info(device);

    // Create the context.
    context = clCreateContext(NULL, 1, &device, &oclContextCallback, NULL, &err);
    checkErr(err, "Failed to create context");

    // Create the command queue.
    queue1 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    checkError(err, "Failed to create command queue");
    queue2 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);

    // Create the program.
    std::string binary_file = getBoardBinaryFile("gnn", device);
    printf("Using AOCX: %s\n", binary_file.c_str());
    program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

    // Build the program that was just created.
    err = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    checkErr(err, "Failed to build program");

    createKernel("GEMM");
    createKernel("linear");
    createKernel("transpose");
    createKernel("interaction_cat");
    createKernel("aggregate_cat");
}

void cleanup()
{
	for(const auto & kernel : kernels)
	{
		if(kernel.second != 0) clReleaseKernel(kernel.second);
	}
	if(program) clReleaseProgram(program);
	if(queue1) clReleaseCommandQueue(queue1);
	if(queue2) clReleaseCommandQueue(queue2);
	if(context) clReleaseContext(context);
}
