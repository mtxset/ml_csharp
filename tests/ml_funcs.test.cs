using System;
using System.Diagnostics;

namespace ml {
    public static class ml_func_tests {
        
        public static void test_matrix_functions() {
            var x = ml_funcs.matrix_create(1, 3);
            var y = ml_funcs.matrix_create(3, 1);
            double[][] output; 

            x[0] = new double[] { 1, 2, 3 };
            y[0][0] = 4; y[1][0] = 5; y[2][0] = 6;
            var result = ml_funcs.matrix_multiply(x, y, out output);

            Console.Write("matrix_multiply");  
            if (!result.has_errors() && output[0][0] == 32) 
                Console.WriteLine(" .. OK");
            else
                Console.WriteLine(" .. FAILED"); 
        }

        public static void run_all_tests() {
            Stopwatch s = Stopwatch.StartNew();
            
            test_matrix_functions();
            ml.exercise2.test_log_regression_cost();
            ml.exercise2.test_cost_logistic_regression_regularized();

            Console.WriteLine($"Testing time: {s.Elapsed}");
        }
    }
}