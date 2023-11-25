#include <pthread.h>

// Define the number of threads
#define NUM_THREADS 4 // You can adjust this based on the number of CPU cores

// Define a structure to hold parameters for each thread
typedef struct
{
    int thread_id;
    int input_row;
    int input_col;
    int *input;
    int kernel_row;
    int kernel_col;
    int *kernel;
    int output_row;
    int output_col;
    long long unsigned int *output;
} ThreadParams;

// Create other necessary functions here

// Function to perform convolution for a portion of the output
void *convolution_thread(void *arg)
{
    ThreadParams *params = (ThreadParams *)arg;
    int thread_id = params->thread_id;
    int input_row = params->input_row;
    int input_col = params->input_col;
    int *input = params->input;
    int kernel_row = params->kernel_row;
    int kernel_col = params->kernel_col;
    int *kernel = params->kernel;
    int output_row = params->output_row;
    int output_col = params->output_col;
    long long unsigned int *output = params->output;

    // Get the number of rows and cols that can be calculated contiguously
    int normal_start_rows = output_row - kernel_row + 1;
    int normal_start_cols = output_col - kernel_col + 1;
    normal_start_rows = normal_start_rows - (normal_start_rows % 8);
    normal_start_cols = normal_start_cols - (normal_start_cols % 8);

    // Debug
    // cout << normal_start_rows << " X " << normal_start_cols << endl;
    int *temp = new int[output_row * output_col];
    for (int kernel_i = 0; kernel_i < kernel_row; kernel_i++)
    {
        for (int kernel_j = 0; kernel_j < kernel_col; kernel_j++)
        {
            __m256i k_vec = _mm256_set1_epi32(kernel[kernel_i * kernel_col + kernel_j]);
            for (int output_i = thread_id; output_i < output_row; output_i += NUM_THREADS)
            {
                int input_i = (output_i + 2 * kernel_i) % input_row;
                int input_idx = input_i * input_col;
                int output_idx = output_i * output_col;
                for (int output_j = 0; output_j < normal_start_cols; output_j += 8)
                {
                    int input_j = (output_j + 2 * kernel_j);
                    __m256i o_vec = _mm256_loadu_si256((__m256i *)(input + input_idx + input_j));

                    __m256i r_vec = _mm256_mullo_epi32(k_vec, o_vec);
                    // int *r = (int *)&r_vec;
                    r_vec = _mm256_add_epi32(r_vec, _mm256_loadu_si256((__m256i *)(temp + output_i * output_col + output_j)));
                    _mm256_storeu_si256((__m256i *)(temp + output_i * output_col + output_j), r_vec);
                }
            }
        }
    }

    for (int i = thread_id; i < output_row; i += NUM_THREADS)
    {
        int idx = i * output_col;
        for (int j = 0; j < normal_start_cols; j++)
        {
            output[idx] = temp[idx];
            idx++;
        }
    }
    for (int output_i = thread_id; output_i < output_row; output_i += NUM_THREADS)
    {
        for (int output_j = normal_start_cols; output_j < output_col; output_j++)
        {
            for (int kernel_i = 0; kernel_i < kernel_row; kernel_i++)
            {
                for (int kernel_j = 0; kernel_j < kernel_col; kernel_j++)
                {
                    int input_i = (output_i + 2 * kernel_i) % input_row;
                    int input_j = (output_j + 2 * kernel_j) % input_col;
                    output[output_i * output_col + output_j] += input[input_i * input_col + input_j] * kernel[kernel_i * kernel_col + kernel_j];
                }
            }
        }
    }
    pthread_exit(NULL);
}

// Fill in this function
void multiThread(int input_row,
                 int input_col,
                 int *input,
                 int kernel_row,
                 int kernel_col,
                 int *kernel,
                 int output_row,
                 int output_col,
                 long long unsigned int *output)
{
    // Create threads
    pthread_t threads[NUM_THREADS];
    ThreadParams thread_params[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++)
    {
        thread_params[i].thread_id = i;
        thread_params[i].input_row = input_row;
        thread_params[i].input_col = input_col;
        thread_params[i].input = input;
        thread_params[i].kernel_row = kernel_row;
        thread_params[i].kernel_col = kernel_col;
        thread_params[i].kernel = kernel;
        thread_params[i].output_row = output_row;
        thread_params[i].output_col = output_col;
        thread_params[i].output = output;

        pthread_create(&threads[i], NULL, convolution_thread, &thread_params[i]);
    }

    // Wait for threads to complete
    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }
}
