using System;
using utils;

namespace ml {
    public static class exercise3 {
        public static void test_rasmussen() {
            const double lambda = 0.1d;
            var file_path = "./data/ex3data1.txt";
            double[][] train_data;
            double[] result_data, label_result_data, gradients, initial_theta;
            double cost;
            int i, labels, offset;

            var parse_result = file_utils.parse_file(file_path, out train_data, out result_data);

            if (parse_result.has_errors()) {
                Console.WriteLine(parse_result.get_errors());
                return;
            }

            Console.WriteLine($"rows: {train_data.Length}, columns: {train_data[0].Length}");

            int label_train_count = result_data.Length / 10; // how many training examples we have for each label (numbers from 0 to 9)

            for (labels = 0; labels < 10; labels++) {
                label_result_data = new double[train_data.Length];
                for (i = 0; i < label_train_count; i++)
                    label_result_data[label_train_count * labels + i] = 1;

                initial_theta = new double[train_data[0].Length + 1];
                var result = ml_funcs.rasmussen(train_data, label_result_data , initial_theta, lambda, 1);

                if (result.has_errors()) {
                    Console.WriteLine(result.get_errors());
                    return;
                }
            }
        }
    }
}