## Base implementation multi thread 1

```

    int output_rows_per_thread = output_row / NUM_THREADS;
    int start_row = thread_id * output_rows_per_thread;
    int end_row = (thread_id == NUM_THREADS - 1) ? output_row : start_row + output_rows_per_thread;

    for (int output_i = thread_id; output_i < output_row; output_i += NUM_THREADS)
    {
        for (int output_j = 0; output_j < output_col; output_j++)
        {
            long long unsigned int temp = 0;
            for (int kernel_i = 0; kernel_i < kernel_row; kernel_i++)
            {
                for (int kernel_j = 0; kernel_j < kernel_col; kernel_j++)
                {
                    int input_i = (output_i + 2 * kernel_i) % input_row;
                    int input_j = (output_j + 2 * kernel_j) % input_col;
                    temp += input[input_i * input_col + input_j] * kernel[kernel_i * kernel_col + kernel_j];
                }
            }
            output[output_i * output_col + output_j] = temp;
        }
    }
```

## Optimization 2
```

    for (int output_i = 2 * thread_id; output_i < output_row; output_i += (NUM_THREADS))
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
```

## Optimization 3
```
for (int output_i = thread_id; output_i < output_row; output_i += NUM_THREADS)
    {
        for (int output_j = 0; output_j < output_col; output_j += 2)
        {
            long long unsigned int temp = 0;
            long long unsigned int temp2 = 0;
            for (int kernel_i = 0; kernel_i < kernel_row; kernel_i++)
            {
                for (int kernel_j = 0; kernel_j < kernel_col; kernel_j++)
                {
                    int input_i = (output_i + 2 * kernel_i) % input_row;
                    int input_j = (output_j + 2 * kernel_j) % input_col;
                    temp += input[input_i * input_col + input_j] * kernel[kernel_i * kernel_col + kernel_j];
                    temp2 += input[input_i * input_col + input_j + 1] * kernel[kernel_i * kernel_col + kernel_j];
                }
            }
            output[output_i * output_col + output_j] = temp;
            output[output_i * output_col + output_j + 1] = temp2;
        }
    }
```