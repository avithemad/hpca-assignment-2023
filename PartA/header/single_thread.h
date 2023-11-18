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

    // Reduce index calculations
    // for (int kernel_i = 0; kernel_i < kernel_row; kernel_i++)
    // {
    //     int ki_off = 2 * kernel_i;
    //     for (int kernel_j = 0; kernel_j < kernel_col; kernel_j++)
    //     {
    //         int k_ind = kernel_i * kernel_col + kernel_j;
    //         int kj_off = 2 * kernel_j;
    //         for (int output_i = 0; output_i < output_row; output_i++)
    //         {
    //             int out_off = output_i * output_col;
    //             int input_i = (output_i + ki_off) % input_row;
    //             int in_off = input_i * input_col;

    //             for (int output_j = 0; output_j < output_col; output_j++)
    //             {
    //                 int input_j = (output_j + kj_off) % input_col;
    //                 output[out_off + output_j] += input[in_off + input_j] * kernel[k_ind];
    //             }
    //         }
    //     }
    // }

    // Optimization 4, compute in blocks for 4 to exploit locality, use modulus sparingly, added reductions in index calculations
    // for (int output_i = 0; output_i < output_row; output_i += 2)
    // {
    //     for (int output_j = 0; output_j < output_col; output_j += 2)
    //     {
    //         long long unsigned int temp = 0;
    //         long long unsigned int temp2 = 0;
    //         long long unsigned int temp3 = 0;
    //         long long unsigned int temp4 = 0;
    //         for (int kernel_i = 0; kernel_i < kernel_row; kernel_i++)
    //         {
    //             int input_i = (output_i + 2 * kernel_i) % input_row;
    //             int in_off = input_i * input_col;
    //             int in1_off = (input_i + 1) * input_col;
    //             int k_off = kernel_i * kernel_col;
    //             for (int kernel_j = 0; kernel_j < kernel_col; kernel_j++)
    //             {
    //                 int k_ind = k_off + kernel_j;
    //                 int input_j = (output_j + 2 * kernel_j) % input_col;
    //                 int x = in_off + input_j, y = in1_off + input_j;
    //                 temp += input[x] * kernel[k_ind];
    //                 temp2 += input[x + 1] * kernel[k_ind];
    //                 temp3 += input[y] * kernel[k_ind];
    //                 temp4 += input[y + 1] * kernel[k_ind];
    //             }
    //         }
    //         int out_off = output_i * output_col;
    //         int out_off1 = (output_i + 1) * output_col;
    //         output[out_off + output_j] = temp;
    //         output[out_off + output_j + 1] = temp2;
    //         output[out_off1 + output_j] = temp3;
    //         output[out_off1 + output_j + 1] = temp4;
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
                int input_i = (output_i + 2 * kernel_i) % input_row;
                int in_off = input_i * input_col;
                int k_off = kernel_i * kernel_col;
                int in1_off = in_off + input_col;
                for (int kernel_j = 0; kernel_j < kernel_col; kernel_j++)
                {
                    int k_ind = k_off + kernel_j;
                    int input_j = (output_j + 2 * kernel_j) % input_col;
                    int x = in_off + input_j, y = in1_off + input_j;
                    temp += input[x] * kernel[k_ind];
                    temp2 += input[x + 1] * kernel[k_ind];
                    temp3 += input[y] * kernel[k_ind];
                    temp4 += input[y + 1] * kernel[k_ind];
                }
            }

            int out_off = output_i * output_col;
            int out_off1 = out_off + output_col;
            output[out_off + output_j] = temp;
            output[out_off + output_j + 1] = temp2;
            output[out_off1 + output_j] = temp3;
            output[out_off1 + output_j + 1] = temp4;
        }
    }

    // for (int output_i = 0; output_i < output_row; output_i += 2)
    // {
    //     for (int output_j = 0; output_j < output_col; output_j += 4)
    //     {
    //         long long unsigned int temp_0_0 = 0;
    //         long long unsigned int temp_0_1 = 0;
    //         long long unsigned int temp_0_2 = 0;
    //         long long unsigned int temp_0_3 = 0;
    //         long long unsigned int temp_1_0 = 0;
    //         long long unsigned int temp_1_1 = 0;
    //         long long unsigned int temp_1_2 = 0;
    //         long long unsigned int temp_1_3 = 0;
    //         for (int kernel_i = 0; kernel_i < kernel_row; kernel_i++)
    //         {
    //             for (int kernel_j = 0; kernel_j < kernel_col; kernel_j++)
    //             {
    //                 int kind = kernel_i * kernel_col + kernel_j;
    //                 int input_i = (output_i + 2 * kernel_i)%input_row, input_j = (output_j + 2 * kernel_j)%input_col;

    //                 temp_0_0 += input[input_i * input_col + input_j] * kernel[kind];
    //                 temp_0_1 += input[input_i * input_col + (input_j + 1)%input_col] * kernel[kind];
    //                 temp_0_2 += input[input_i * input_col + (input_j + 2)%input_col] * kernel[kind];
    //                 temp_0_3 += input[input_i * input_col + (input_j + 3)%input_col] * kernel[kind];
    //                 temp_1_0 += input[(input_i + 1) * input_col + input_j] * kernel[kind];
    //                 temp_1_1 += input[(input_i + 1) * input_col + (input_j + 1)%input_col] * kernel[kind];
    //                 temp_1_2 += input[(input_i + 1) * input_col + (input_j + 2)%input_col] * kernel[kind];
    //                 temp_1_3 += input[(input_i + 1) * input_col + (input_j + 3)%input_col] * kernel[kind];
    //             }
    //         }
    //         output[output_i * output_col + output_j] = temp_0_0;
    //         output[output_i * output_col + output_j + 1] = temp_0_1;
    //         output[output_i * output_col + output_j + 2] = temp_0_2;
    //         output[output_i * output_col + output_j + 3] = temp_0_3;
    //         output[(output_i + 1) * output_col + output_j] = temp_1_0;
    //         output[(output_i + 1) * output_col + output_j + 1] = temp_1_1;
    //         output[(output_i + 1) * output_col + output_j + 2] = temp_1_2;
    //         output[(output_i + 1) * output_col + output_j + 3] = temp_1_3;
    //     }
    // }
}
