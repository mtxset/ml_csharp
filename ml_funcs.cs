using System;

namespace ml {
    public static class ml_funcs {
        
        // cost function for logistic regression
        // J = sum((X*theta - y).^2) / (2*m);
        public static double compute_cost(double[] x, double[] y, double[] theta) {
            var result = 0.0d;
            if (x.Length != y.Length) {
                Console.WriteLine("Arrays x and y should be equal");
                return -1;  
            }

            var temp = 0d;
            for (var i = 0; i < x.Length; i++) {
                temp = (theta[0] + theta[1] * x[i]) - y[i];
                temp *= temp;
                result += temp;
            }

            result /= 2*y.Length;

            return result;
        }

        public static void feature_normalize(in double[,] train_data, out double[] mean, out double[] sigma) {
            mean = new double[1];
            sigma = new double[1];       
        }

        public static double mean(in double[] array) {
            var sum = 0d;

            for (var i = 0; i < array.Length; ++i)
                sum += array[i];

            return sum/array.Length; 
        }

        public static double stadard_deviation(in double[] array) {
            var sum = 0d;
            var mu = mean(array);

            for (int i = 0; i < array.Length; i++)
                sum += Math.Pow(array[i] - mu, 2);

            return Math.Sqrt(sum);
        }

        // cost function for logistic regression
        // J = sum((X*theta - y).^2) / (2*m);
        public static double compute_cost_multi(double[,] train_data, double[] result_data, double[] theta) {
            var result = 0.0d;
            if (train_data.GetLength(0) != result_data.Length) {
                Console.WriteLine("Arrays train_data and y should be equal");
                return -1;
            }

            if (train_data.GetLength(1) != theta.Length-1) {
                Console.WriteLine("train_data should have one less size of theta");
                return -1;
            }

            var temp = 0d;
            for (var i = 0; i < train_data.GetLength(0); i++) {
                temp = theta[0];
                for (var t = 1; t < train_data.GetLength(1); t++)
                    temp += theta[t] * train_data[i, t-1];
                    
                temp -= result_data[i];
                temp *= temp;
                result += temp;
            }

            result /= 2*result_data.Length;
            return result;
        }

        public static void gradient_decent(double[] x, double[] y, double[] theta, double alpha, int iterations) {

        }
    }
}