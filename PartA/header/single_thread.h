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

    // for (int i = 0; i < output_row * output_col; ++i)
    //     output[i] = 0;

    // Optimization 4, compute in blocks for 4 to exploit locality, use modulus sparingly
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
                    int i = output_i + 2 * kernel_i, j = output_j + 2 * kernel_j;
                    int input_i = i >= input_row ? i % input_row : i;
                    int input_j = j >= input_col ? j % input_col : j;

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
}
