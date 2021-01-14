using System;

namespace ml {
    public static class ml_funcs {
        
        // cost function for logistic regression
        // J = sum((X*theta - y).^2) / (2*m);
        public static double compute_cost(double[] x, double[] y, double[] theta) {
            var result = 0.0d;
            if (x.Length != y.Length) {
                Console.WriteLine("Arrays x and y should be equal");
                return 0;
            }

            var temp = 0d;
            for (var i = 0; i < x.Length; i++) {
                temp = (theta[0] + theta[1] * x[i]) - y[i];
                temp *= temp;
                result += temp;
            }

            result /= 2*x.Length;

            return result;
        }
    }
}