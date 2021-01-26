using System;
using utils;

namespace ml {
    public static class exercise3 {
        public static void test_cost_logistic_regression_regularized() {
            const double lambda = 0.1d;
            var file_path = "./data/ex3data1.txt";
            double[][] train_data;
            double[] result_data, gradients, initial_theta;
            double cost;

            var parse_result = file_utils.parse_file(file_path, out train_data, out result_data);

            if (parse_result.has_errors()) {
                Console.WriteLine(parse_result.get_errors());
                return;
            }
            Console.WriteLine($"rows: {train_data.Length}, columns: {train_data[0].Length}");

            initial_theta = new double[train_data[0].Length];
            ml_funcs.cost_logistic_regression_regularized(train_data, result_data, initial_theta, lambda, out cost, out gradients);

            Console.Write("cost_logistic_regression_regularized() cost"); 
            if (Math.Round(cost, 5) == 0.69315)
                Console.WriteLine(" .. OK");
            else
                Console.WriteLine(" .. FAILED"); 
        }
    }
}