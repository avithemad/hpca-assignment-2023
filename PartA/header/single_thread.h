#include <immintrin.h>

// Optimize this function
void singleThread(int input_row,
                  int input_col,
                  int *input,
                  int kernel_row,
                  int kernel_col,
                  int *kernel,
                  int output_row,
                  int output_col,
                  long long unsigned int *output)
{

    for (int i = 0; i < output_row * output_col; ++i)
        output[i] = 0;
    // Get the number of rows and cols that can be calculated contiguously
    int normal_start_rows = output_row - kernel_row + 1;
    int normal_start_cols = output_col - kernel_col + 1;
    normal_start_rows = normal_start_rows - (normal_start_rows % 8);
    normal_start_cols = normal_start_cols - (normal_start_cols % 8);

    // Debug
    // cout << normal_start_rows << " X " << normal_start_cols << endl;
    int* temp = (int*)malloc(sizeof(int)*output_row*output_col);
    for (int kernel_i = 0; kernel_i < kernel_row; kernel_i++)
    {
        for (int kernel_j = 0; kernel_j < kernel_col; kernel_j++)
        {
            __m512i k_vec = _mm512_set1_epi32(kernel[kernel_i * kernel_col + kernel_j]);
            for (int output_i = 0; output_i < output_row; output_i++)
            {
                int input_i = (output_i + 2 * kernel_i) % input_row;
                int input_idx = input_i * input_col;
                int output_idx = output_i * output_col;
                for (int output_j = 0; output_j < normal_start_cols; output_j += 8)
                {
                    int input_j = (output_j + 2 * kernel_j);
                    __m512i o_vec = _mm512_loadu_si512((__m512i *)(input + input_idx + input_j));

                    __m512i r_vec = _mm512_mullo_epi32(k_vec, o_vec);
                    // int *r = (int *)&r_vec;
                    r_vec = _mm512_add_epi32(r_vec, _mm512_loadu_si512((__m512i*)(temp + output_i*output_col + output_j)));
                    _mm512_storeu_si512((__m512i*)(temp + output_i*output_col + output_j), r_vec);
                }
            }
        }
    }

    for (int i=0;i<output_row*output_col;i++) {
        output[i] = temp[i];
    }

    for (int output_i = 0; output_i < output_row; output_i++)
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
}
