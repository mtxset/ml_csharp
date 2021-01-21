using System;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using utils;

namespace ml {
    public static class exercise2 {
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
            double[] gradient; // -0.1,  -12.00922, -11.26284
            var theta = ml_funcs.cost_logistic_regression(train_data, result_data, new double[] { 0, 0, 0 }, out cost, out gradient);

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