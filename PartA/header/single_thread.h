#include <emmintrin.h>

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

    // Optimization 1, use a temporary variable to store the output
    // for (int output_i = 0; output_i < output_row; output_i++)
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

    for (int output_i = 0; output_i < output_row; output_i += 2)
    {
        for (int output_j = 0; output_j < output_col; output_j += 2)
        {
            long long unsigned int temp = 0;
            long long unsigned int temp2 = 0;
            long long unsigned int temp3 = 0;
            long long unsigned int temp4 = 0;
            for (int kernel_i = 0; kernel_i < kernel_row; kernel_i++)
            {
                for (int kernel_j = 0; kernel_j < kernel_col; kernel_j++)
                {
                    int input_i = (output_i + 2 * kernel_i) % input_row;
                    int input_j = (output_j + 2 * kernel_j) % input_col;
                    temp += input[input_i * input_col + input_j] * kernel[kernel_i * kernel_col + kernel_j];
                    temp2 += input[input_i * input_col + input_j + 1] * kernel[kernel_i * kernel_col + kernel_j];
                    temp3 += input[(input_i + 1) * input_col + input_j] * kernel[kernel_i * kernel_col + kernel_j];
                    temp4 += input[(input_i + 1) * input_col + input_j + 1] * kernel[kernel_i * kernel_col + kernel_j];
                }
            }
            output[output_i * output_col + output_j] = temp;
            output[output_i * output_col + output_j + 1] = temp2;
            output[(output_i + 1) * output_col + output_j] = temp3;
            output[(output_i + 1) * output_col + output_j + 1] = temp4;
        }
    }

    // for (int output_i = 0; output_i < output_row; output_i += 2)
    // {
    //     for (int output_j = 0; output_j < output_col; output_j += 2)
    //     {
    //         __m128i temp1 = _mm_setzero_si128();
    //         __m128i temp2 = _mm_setzero_si128();
    //         __m128i temp3 = _mm_setzero_si128();
    //         __m128i temp4 = _mm_setzero_si128();

    //         for (int kernel_i = 0; kernel_i < kernel_row; kernel_i++)
    //         {
    //             for (int kernel_j = 0; kernel_j < kernel_col; kernel_j++)
    //             {
    //                 int input_i = (output_i + 2 * kernel_i) % input_row;
    //                 int input_j = (output_j + 2 * kernel_j) % input_col;

    //                 __m128i input_values = _mm_set_epi16(input[input_i * input_col + input_j + 1],
    //                                                      input[input_i * input_col + input_j],
    //                                                      input[(input_i + 1) * input_col + input_j + 1],
    //                                                      input[(input_i + 1) * input_col + input_j],
    //                                                      0, 0, 0, 0);

    //                 __m128i kernel_values = _mm_set1_epi16(kernel[kernel_i * kernel_col + kernel_j]);

    //                 temp1 = _mm_add_epi32(temp1, _mm_mullo_epi16(input_values, kernel_values));

    //                 // Similar calculations for temp2, temp3, and temp4
    //             }
    //         }

    //         long long unsigned int result[4];
    //         _mm_storeu_si128((__m128i *)result, temp1);
    //         output[output_i * output_col + output_j] = result[0];
    //         output[output_i * output_col + output_j + 1] = result[1];

    //         // Similar store operations for temp2, temp3, and temp4
    //     }
    // }

    // for (int output_i = 0; output_i < output_row; output_i += 2)
    // {
    //     for (int output_j = 0; output_j < output_col; output_j += 2)
    //     {
    //         for (int kernel_i = 0; kernel_i < kernel_row; kernel_i++)
    //         {
    //             for (int kernel_j = 0; kernel_j < kernel_col; kernel_j++)
    //             {
    //                 int input_i = (output_i + 2 * kernel_i) % input_row;
    //                 int input_j = (output_j + 2 * kernel_j) % input_col;
    //                 int input_ip = (output_i + 1 + 2 * kernel_i) % input_row;
    //                 int input_jp = (output_j + 1 + 2 * kernel_j) % input_col;

    //                 output[output_i * output_col + output_j] += input[input_i * input_col + input_j] * kernel[kernel_i * kernel_col + kernel_j];
    //                 // if (output_j + 1 < output_col)
    //                 // {

    //                 output[output_i * output_col + output_j + 1] += input[input_i * input_col + input_jp] * kernel[kernel_i * kernel_col + kernel_j];
    //                 // }
    //                 // if (output_i + 1 < output_row)
    //                 // {

    //                 output[(output_i + 1) * output_col + output_j] += input[(input_ip)*input_col + input_j] * kernel[kernel_i * kernel_col + kernel_j];
    //                 // }
    //                 // if (output_j + 1 < output_col && output_i + 1 < output_row)
    //                 // {

    //                 output[(output_i + 1) * output_col + output_j + 1] += input[(input_ip)*input_col + input_jp] * kernel[kernel_i * kernel_col + kernel_j];
    //                 // }
    //             }
    //         }
    //     }
    // }

    // Optimization 3
    // for (int kernel_i = 0; kernel_i < kernel_row; kernel_i++)
    // {
    //     for (int kernel_j = 0; kernel_j < kernel_col; kernel_j++)
    //     {
    //         for (int output_i = 0; output_i < output_row; output_i++)
    //         {
    //             for (int output_j = 0; output_j < output_col; output_j++)
    //             {
    //                 int input_i = (output_i + 2 * kernel_i) % input_row;
    //                 int input_j = (output_j + 2 * kernel_j) % input_col;
    //                 output[output_i * output_col + output_j] += input[input_i * input_col + input_j] * kernel[kernel_i * kernel_col + kernel_j];
    //             }
    //         }
    //     }
    // }
}
