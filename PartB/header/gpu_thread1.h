// Create other necessary functions here

__global__ void convolution(int *input,
							int *kernel,
							long long unsigned int *output,
							int output_row,
							int output_col,
							int kernel_row,
							int kernel_col,
							int input_row,
							int input_col)
{
	int output_i = blockIdx.y * blockDim.y + threadIdx.y;
	int output_j = blockIdx.x * blockDim.x + threadIdx.x;
	if (output_i < output_row && output_j < output_col) {

		int kernel_idx = blockIdx.z * blockDim.z;

		int kernel_i = kernel_idx/kernel_col;
		int kernel_j = kernel_idx%kernel_col;

		int input_i = (output_i + 2 * kernel_i) % input_row;
		int input_j = (output_j + 2 * kernel_j) % input_col;

		atomicAdd(&output[output_i * output_col + output_j], input[input_i * input_col + input_j] * kernel[kernel_idx]);
	}
}

// Fill in this function
void gpuThread(int input_row,
			   int input_col,
			   int *input,
			   int kernel_row,
			   int kernel_col,
			   int *kernel,
			   int output_row,
			   int output_col,
			   long long unsigned int *output)
{

	int *d_input, *d_kernel;
	long long unsigned int *d_output;
	int input_bytes = sizeof(int) * input_row * input_col;
	int output_bytes = sizeof(long long unsigned int) * output_row * output_col;
	int kernel_bytes = sizeof(int) * kernel_row * kernel_col;
	// Allocate memory in GPU
	cudaError_t cuda_err = cudaMalloc(&d_input, input_bytes);
	if (cuda_err != cudaSuccess)
	{
		cout << "CUDA ERROR: " << cudaGetErrorString(cuda_err) << endl;
		exit(EXIT_FAILURE);
	}
	cudaMalloc(&d_output, output_bytes);
	cudaMalloc(&d_kernel, kernel_bytes);

	// Copy data into GPU
	cudaMemcpy(d_input, input, input_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_output, output, output_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_kernel, kernel, kernel_bytes, cudaMemcpyHostToDevice);

	// max threads per grid is 32x32 = 1024 in CUDA programming model
	int thread_grid_size = 32;
	dim3 threadsPerBlock(thread_grid_size, thread_grid_size, 1);
	dim3 blocksPerGrid(ceil(double(output_row) / float(thread_grid_size)), ceil(double(output_col) / float(thread_grid_size)), kernel_row*kernel_col);
	// std::cout << "blocks per grid: " << blocksPerGrid.x << " " << blocksPerGrid.y << " " << blocksPerGrid.z << std::endl;
	convolution<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_kernel, d_output, output_row, output_col, kernel_row, kernel_col, input_row, input_col);
	cuda_err = cudaGetLastError();
	if (cuda_err != cudaSuccess)
	{
		std::cout << "CUDA Kernel Execution ERROR: " << cudaGetErrorString(cuda_err) << std::endl;
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(output, d_output, output_bytes, cudaMemcpyDeviceToHost);

}
