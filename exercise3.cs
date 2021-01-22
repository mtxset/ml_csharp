using System;
using utils;

namespace ml {
    public static class exercise3 {
        public static void test_cost_logistic_regression_regularized() {
            var file_path = "./data/ex3data1.txt";
            double[][] train_data;
            double[] result_data;

            var parse_result = file_utils.parse_file(file_path, out train_data, out result_data);

            if (parse_result.has_errors()) {
                Console.WriteLine(parse_result.get_errors());
                return;
            }
        }
    }
}