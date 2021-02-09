using System;
using System.Diagnostics;
using System.IO;

namespace ml {
    public static class ml_func_tests {

        public static void test_sigmoid() {
            Console.Write("sigmoid");
            if (Math.Round(ml_funcs.sigmoid(0), 1) == 0.5)
                Console.WriteLine(" .. OK");
            else
                Console.WriteLine(" .. FAILED");
        }

        public static void test_matrix_apply_function() {
            Console.Write("test_"+System.Reflection.MethodBase.GetCurrentMethod().Name);
            
            var x = ml_funcs.matrix_create(2, 3);
            x[0] = new double[] { 0, 1, 0.5 }; 
            x[1] = new double[] { 0, -1, -0.5 };

            x = ml_funcs.matrix_apply_function(x, ml_funcs.sigmoid_gradient); 
            if (Math.Round(x[0][0], 4) == 0.2500 &&
                Math.Round(x[0][1], 4) == 0.1966 &&
                Math.Round(x[0][2], 4) == 0.2350 &&
                Math.Round(x[1][0], 4) == 0.2500 &&
                Math.Round(x[1][1], 4) == 0.1966 &&
                Math.Round(x[1][2], 4) == 0.2350)
                Console.WriteLine(" .. OK");
            else
                Console.WriteLine(" .. FAILED");
        }

        public static void test_sigmoid_gradient() {
            // 1        -0.5        0       0.5      -1
            // 0.1966   0.2350   0.2500   0.2350   0.1966
            Console.Write("sigmoid_gradient");
            if (Math.Round(ml_funcs.sigmoid_gradient(1), 4) == 0.1966 &&
                Math.Round(ml_funcs.sigmoid_gradient(-0.5), 4) == 0.2350 &&
                Math.Round(ml_funcs.sigmoid_gradient(0), 4) == 0.2500 &&
                Math.Round(ml_funcs.sigmoid_gradient(-0.5), 4) == 0.2350 &&
                Math.Round(ml_funcs.sigmoid_gradient(-1), 4) == 0.1966)
                Console.WriteLine(" .. OK");
            else
                Console.WriteLine(" .. FAILED");
        }

        public static void test_matrix_insert_column() {
            double insert_value = 0.999d;
            var x = ml_funcs.matrix_create(2, 3);
            x[0] = new double[] { 1.123, 2.12, 3.123 }; 
            x[1] = new double[] { 4.123, 5.123, 6.123 };

            var y = ml_funcs.matrix_insert_column(x, 0, insert_value);
            var z = ml_funcs.matrix_insert_column(x, 3, insert_value);
            Console.Write("test_matrix_insert_column");
            if (y[0].Length == 4 && y[0][0] == insert_value && y[0][1] == x[0][0] &&
                z[0].Length == 4 && z[0][3] == insert_value && z[0][0] == x[0][0])              
                Console.WriteLine(" .. OK");
            else
                Console.WriteLine(" .. FAILED");
        }

        public static void test_matrix_to_csv_and_back() {
            var result = new result_state();
            var x = ml_funcs.matrix_create(2, 3);
            var file_name = "test_matrix_to_csv_and_back.txt";
            double[][] y;

            x[0] = new double[] { 1.123, 2.12, 3.123 }; 
            x[1] = new double[] { 4.123, 5.123, 6.123 };

            result = ml_funcs.matrix_to_csv(file_name, x);

            Console.Write("matrix_to_csv_and_back");
            if (result.has_errors()) {
                Console.WriteLine(" .. FAILED");
                Console.WriteLine(result.all_errors_to_string());
                return;
            }

            result = ml_funcs.matrix_from_csv(file_name, out y);

            if (result.has_errors()) {
                Console.WriteLine(" .. FAILED");
                Console.WriteLine(result.all_errors_to_string());
                return;
            }

            if (ml_funcs.matrix_compare_deep(x, y))                
                Console.WriteLine(" .. OK");
            else
                Console.WriteLine(" .. FAILED");

            File.Delete(file_name);
        } 

        public static void test_matrix_compare_deep() {
            var x = ml_funcs.matrix_create(2, 3);
            var y = ml_funcs.matrix_create(2, 3);
            var z = ml_funcs.matrix_create(3, 3);
            var q = ml_funcs.matrix_create(2, 3, 0);

            x[0] = new double[] { 1, 2, 3 }; 
            x[1] = new double[] { 4, 5, 6 };

            y[0] = new double[] { 1, 2, 3 }; 
            y[1] = new double[] { 4, 5, 6 };

            z[0] = new double[] { 1, 2, 3 }; 
            z[1] = new double[] { 4, 5, 6 };
            z[2] = new double[] { 4, 5, 6 };

            Console.Write("matrix_compare_deep");
            if (ml_funcs.matrix_compare_deep(x, y) && !ml_funcs.matrix_compare_deep(x, z) && !ml_funcs.matrix_compare_deep(x, q))
                Console.WriteLine(" .. OK");
            else
                Console.WriteLine(" .. FAILED");
        }

        public static void test_transpose() {
            var x = ml_funcs.matrix_create(2, 3);

            x[0] = new double[] { 1, 2, 3 }; 
            x[1] = new double[] { 4, 5, 6 }; 
            var y = ml_funcs.matrix_transpose(x);
            Console.Write("matrix_transpose");
            if (y.Length == 3 && y[0].Length == 2 && 
                y[0][0] == 1 && y[0][1] == 4 &&
                y[1][0] == 2 && y[1][1] == 5 &&
                y[2][0] == 3 && y[2][1] == 6)
                Console.WriteLine(" .. OK");
            else
                Console.WriteLine(" .. FAILED");
        }
        
        public static void test_max_and_max_value() {
            var array = new double[] { -2, -1, -3, 1, 2 };
            double max_value;
            int max_index;
            
            ml_funcs.max_value(array, out max_value, out max_index);
            Console.Write("max_value");
            if (max_value == 2 && max_index == 4)
                Console.WriteLine(" .. OK");
            else
                Console.WriteLine(" .. FAILED");

            ml_funcs.min_value(array, out max_value, out max_index);
            Console.Write("min_value");
            if (max_value == -3 && max_index == 2)
                Console.WriteLine(" .. OK");
            else
                Console.WriteLine(" .. FAILED");
        }

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
            
            test_matrix_insert_column();
            test_matrix_to_csv_and_back();
            test_matrix_compare_deep();
            test_transpose();
            test_max_and_max_value();
            test_matrix_functions();
            ml.exercise2.test_log_regression_cost();
            ml.exercise2.test_cost_logistic_regression_regularized();
            ml.exercise3.test_rasmussen();

            Console.WriteLine($"Testing time: {s.Elapsed}");
        }
    }
}