using System;
using System.Diagnostics;

namespace ml {
    public static class ml_func_tests {
        public static void run_all_tests() {
            Stopwatch s = Stopwatch.StartNew();
            
            ml.exercise2.test_log_regression_cost();

            Console.WriteLine($"Testing time: {s.Elapsed}");
        }
    }
}