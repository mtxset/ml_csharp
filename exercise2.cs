using System;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using utils;

namespace ml {
    public static class exercise2 {
        public static void test_cost_logistic_regression_regularized() {
            var file_path = "./data/ex2data2.txt";
            const double lambda = 1;

            double[][] train_data;
            double[] result_data, initial_theta;

            var parse_result = file_utils.parse_file(file_path, out train_data, out result_data);

            if (parse_result.has_errors()) {
                Console.WriteLine(parse_result.get_errors());
                return;
            }

            initial_theta = new double[train_data[0].Length + 1];
            double cost; // 0.69315
            double[] gradient;  // 0.008474576 0.018788093 0.000077771
            var result = ml_funcs.cost_logistic_regression_regularized(train_data, result_data, initial_theta, lambda, out cost, out gradient);

            if (result.has_errors()) {
                Console.WriteLine(result.get_errors());
                return;
            }

            Console.Write("cost_logistic_regression_regularized cost");
            if (Math.Round(cost, 5) == 0.69315) 
                Console.WriteLine(" .. OK");
            else
                Console.WriteLine(" .. FAILED");

            var test_results = new double[] { 0.008474576, 0.018788093, 0.000077771 };
            Console.Write("cost_logistic_regression_regularized gradient");
            var correct = 0;
            for (var i = 0; i < test_results.Length; i++)
                if (Math.Round(gradient[i], 9) == test_results[i]) correct++;

            if (correct == test_results.Length)
                Console.WriteLine(" .. OK");
            else
                Console.WriteLine(" .. FAILED");
        }

        public static void test_log_regression_cost() {
            var file_path = "./data/ex2data1.txt";
            double[][] train_data;
            double[] result_data;

            var parse_result = file_utils.parse_file(file_path, out train_data, out result_data);

            if (parse_result.has_errors()) {
                Console.WriteLine(parse_result.get_errors());
                return;
            }

            double cost; // 0.69315
            double[] gradient; // -0.1, -12.00922, -11.26284
            ml_funcs.cost_logistic_regression(train_data, result_data, new double[] { 0, 0, 0 }, out cost, out gradient);

            Console.Write("cost_logistic_regression cost");  
            if (Math.Round(cost, 5) == 0.69315) 
                Console.WriteLine(" .. OK");
            else
                Console.WriteLine(" .. FAILED"); 

            double[] test_results = new double[] { -0.1,  -12.00922, -11.26284 };

            Console.Write("cost_logistic_regression gradient");
            var correct = 0;
            for (var i = 0; i < test_results.Length; i++)
                if (Math.Round(gradient[i], 5) == test_results[i]) correct++;

            if (correct == test_results.Length)
                Console.WriteLine(" .. OK");
            else
                Console.WriteLine(" .. FAILED");
        }
    }
}