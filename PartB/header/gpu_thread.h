// Create other necessary functions here
//
__global__ void convolution(int *input, 
		int *kernel, 
		long long unsigned int *output, 
		int output_row, 
		int output_col, 
		int kernel_row, 
		int kernel_col) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < output_row*output_col) {
		for (int k_i=0; k_i < kernel_row; k_i++) {
			for (int k_j=0; k_j < kernel_col; k_j++) {
				output[idx] = input[idx];
			}
		}	
	} 
}

// Fill in this function
void gpuThread( int input_row, 
                int input_col,
                int *input, 
                int kernel_row, 
                int kernel_col, 
                int *kernel,
                int output_row, 
                int output_col, 
                long long unsigned int *output ) 
{
    
	int* d_input, *d_kernel;
	long long unsigned int* d_output;
	int input_bytes = sizeof(int)*input_row*input_col;
	int output_bytes = sizeof(long long unsigned int)*output_row*output_col;
	int kernel_bytes = sizeof(int)*kernel_row*kernel_col;
	// Allocate memory in GPU
	cudaError_t cuda_err = cudaMalloc(&d_input, input_bytes);
	if (cuda_err != cudaSuccess) {
		cout << "CUDA ERROR: " << cudaGetErrorString(cuda_err) << endl;
		exit(EXIT_FAILURE);
	}	
	cudaMalloc(&d_output, output_bytes);
	cudaMalloc(&d_kernel, kernel_bytes);

	// Copy data into GPU
	cudaMemcpy(d_input, input, input_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_output, output, output_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_kernel, kernel, kernel_bytes, cudaMemcpyHostToDevice);

	int thread_per_block = 256;
	int block_in_grid = 256;

	convolution<<< block_in_grid, thread_per_block >>> (d_input, d_kernel, d_output, output_row, output_col, kernel_row, kernel_col);

	cudaMemcpy(output, d_output, output_bytes, cudaMemcpyDeviceToHost);
	
}
