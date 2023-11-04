#include <pthread.h>
#include <emmintrin.h>

// Define the number of threads
#define NUM_THREADS 16 // You can adjust this based on the number of CPU cores

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

    // Calculate the portion of the output matrix to compute for this thread
    int output_rows_per_thread = output_row / NUM_THREADS;
    int start_row = thread_id * output_rows_per_thread;
    int end_row = (thread_id == NUM_THREADS - 1) ? output_row : start_row + output_rows_per_thread;

    // for (int output_i = thread_id; output_i < output_row; output_i += NUM_THREADS)
    // {
    //     for (int output_j = 0; output_j < output_col; output_j++)
    //     {
    //         long long unsigned int temp = 0;
    //         for (int kernel_i = 0; kernel_i < kernel_row; kernel_i++)
    //         {
    //             for (int kernel_j = 0; kernel_j < kernel_col; kernel_j++)
    //             {
    //                 int input_i = (output_i + 2 * kernel_i) % input_row;
    //                 int input_j = (output_j + 2 * kernel_j) % input_col;
    //                 temp += input[input_i * input_col + input_j] * kernel[kernel_i * kernel_col + kernel_j];
    //             }
    //         }
    //         output[output_i * output_col + output_j] = temp;
    //     }
    // }

    // for (int output_i = thread_id; output_i < end_row; output_i += 2 * NUM_THREADS)
    // {
    //     for (int output_j = 0; output_j < output_col; output_j += 2)
    //     {
    //         long long unsigned int temp = 0;
    //         long long unsigned int temp2 = 0;
    //         long long unsigned int temp3 = 0;
    //         long long unsigned int temp4 = 0;
    //         for (int kernel_i = 0; kernel_i < kernel_row; kernel_i++)
    //         {
    //             for (int kernel_j = 0; kernel_j < kernel_col; kernel_j++)
    //             {
    //                 int input_i = (output_i + 2 * kernel_i) % input_row;
    //                 int input_j = (output_j + 2 * kernel_j) % input_col;
    //                 temp += input[input_i * input_col + input_j] * kernel[kernel_i * kernel_col + kernel_j];
    //                 temp2 += input[input_i * input_col + input_j + 1] * kernel[kernel_i * kernel_col + kernel_j];
    //                 temp3 += input[(input_i + 1) * input_col + input_j] * kernel[kernel_i * kernel_col + kernel_j];
    //                 temp4 += input[(input_i + 1) * input_col + input_j + 1] * kernel[kernel_i * kernel_col + kernel_j];
    //             }
    //         }
    //         output[output_i * output_col + output_j] = temp;
    //         output[output_i * output_col + output_j + 1] = temp2;
    //         output[(output_i + 1) * output_col + output_j] = temp3;
    //         output[(output_i + 1) * output_col + output_j + 1] = temp4;
    //     }
    // }
    for (int output_i = thread_id; output_i < end_row; output_i += 2 * NUM_THREADS)
    {
        for (int output_j = 0; output_j < output_col; output_j += 2)
        {
            __m128i temp1 = _mm_setzero_si128();
            __m128i temp2 = _mm_setzero_si128();
            __m128i temp3 = _mm_setzero_si128();
            __m128i temp4 = _mm_setzero_si128();

            for (int kernel_i = 0; kernel_i < kernel_row; kernel_i++)
            {
                for (int kernel_j = 0; kernel_j < kernel_col; kernel_j++)
                {
                    int input_i = (output_i + 2 * kernel_i) % input_row;
                    int input_j = (output_j + 2 * kernel_j) % input_col;

                    __m128i input_values = _mm_set_epi16(input[input_i * input_col + input_j + 1],
                                                         input[input_i * input_col + input_j],
                                                         input[(input_i + 1) * input_col + input_j + 1],
                                                         input[(input_i + 1) * input_col + input_j],
                                                         0, 0, 0, 0);

                    __m128i kernel_values = _mm_set1_epi16(kernel[kernel_i * kernel_col + kernel_j]);

                    temp1 = _mm_add_epi32(temp1, _mm_mullo_epi16(input_values, kernel_values));

                    // Similar calculations for temp2, temp3, and temp4
                }
            }

            long long unsigned int result[4];
            _mm_storeu_si128((__m128i *)result, temp1);
            output[output_i * output_col + output_j] = result[0];
            output[output_i * output_col + output_j + 1] = result[1];
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