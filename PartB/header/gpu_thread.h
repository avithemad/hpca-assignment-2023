// Create other necessary functions here

__global__ void convolution(int *input, 
		int *kernel, 
		long long unsigned int *output, 
		int output_row, 
		int output_col, 
		int kernel_row, 
		int kernel_col,
		int input_row,
		int input_col) {
	int output_i = blockIdx.y*blockDim.y + threadIdx.y;
	int output_j = blockIdx.x*blockDim.x + threadIdx.x;

	long long unsigned int temp = 0;
	if (output_i*output_col < output_row && output_j < output_col); {
	for (int kernel_i=0; kernel_i < kernel_row; kernel_i++) {
		for (int kernel_j=0; kernel_j < kernel_col; kernel_j++) {
			int input_i = (output_i + 2*kernel_i) % input_row;		
			int input_j = (output_j + 2*kernel_j) % input_col;
			temp += input[input_i*input_col + input_j]*kernel[kernel_i*kernel_col + kernel_j];
		}
	}	
	output[output_i*output_col + output_j] = temp;}
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

	dim3 threadsPerBlock(16, 16);
    	dim3 blocksPerGrid(ceil(double(output_row)/16.0f), ceil(double(output_col)/16.0f));
	cout << "Threads per block, x: " << threadsPerBlock.x << " y: " << threadsPerBlock.y << endl;
	cout << "Blocks per grid, x: " << blocksPerGrid.x << " y: " << blocksPerGrid.y << endl;
        /*if (output_row*output_col > 512){
            threadsPerBlock.x = 512;
            threadsPerBlock.y = 512;
            blocksPerGrid.x = ceil(double(output_row)/double(threadsPerBlock.x));
            blocksPerGrid.y = ceil(double(output_col)/double(threadsPerBlock.y));
        }*/

	convolution<<< blocksPerGrid, threadsPerBlock >>> (d_input, d_kernel, d_output, output_row, output_col, kernel_row, kernel_col, input_row, input_col);
cudaMemcpy(output, d_output, output_bytes, cudaMemcpyDeviceToHost);
/*
	for (int i=0; i<output_row; i++) {
		for(int j=0; j<output_col; j++) {
			cout << output[i*output_col + j] << "\t";
		}
		cout << endl;
	}
	cout << endl;
*/	
}
