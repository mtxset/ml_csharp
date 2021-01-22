using System;

namespace ml {
    public static class ml_funcs {

        private enum operation { add, subtract, multiply, divide }; 

        public static double[][] matrix_create(int rows, int columns, double prefill = -1d) {
            double[][] result = new double[rows][];

            for (int r = 0; r < rows; r++) {
                result[r] = new double[columns];

                if (prefill != -1d) {
                    for (int c = 0; c < columns; c++)
                        result[r][c] = prefill;
                }
            }
            
            return result;
        }

        public static double[][] matrix_transpose(in double[][] matrix) {
            var result = matrix_create(matrix[0].Length, matrix.Length);
            int row, col;

            for (row = 0; row < matrix.Length; row++) {
                for (col = 0; col < matrix[0].Length; col++) {
                    result[col][row] = matrix[row][col];
                }
            }
            return result;
        }

        public static double[][] matrix_subtract(in double[][] x, in double[][] y) {
            var result = matrix_create(x.Length, x[0].Length);
            int row, col;

            for (row = 0; row < x.Length; row++) {
                for (col = 0; col < x[0].Length; col++) {
                    result[row][col] = x[row][col] - y[row][col]; 
                }
            }
            return result;
        }

        public static double[][] matrix_add(in double[][] x, in double[][] y) {
            var result = matrix_create(x.Length, x[0].Length);
            int row, col;

            for (row = 0; row < x.Length; row++) {
                for (col = 0; col < x[0].Length; col++) {
                    result[row][col] = x[row][col] + y[row][col]; 
                }
            }
            return result;
        }

        private static double[][] matrix_op_vector(in double[][] x, in double[] y, operation op) {
            var result = matrix_create(x.Length, x[0].Length);
            int row, col;

            for (row = 0; row < x.Length; row++) {
                for (col = 0; col < x[0].Length; col++) {
                    switch (op) {
                    case operation.add:
                        result[row][col] = x[row][col] + y[col];
                        break;
                    case operation.subtract:
                        result[row][col] = x[row][col] - y[col];
                        break;
                    case operation.divide:
                        result[row][col] = x[row][col] / y[col];
                        break;
                    case operation.multiply:
                        result[row][col] = x[row][col] * y[col];
                        break;
                    }
                }
            }

            return result;
        } 

        public static double[][] matrix_divide_vector(in double[][] x, in double[] y) {
            return matrix_op_vector(in x, in y, operation.divide);
        }

        public static double[][] matrix_multiply_vector(in double[][] x, in double[] y) {
            return matrix_op_vector(in x, in y, operation.multiply);
        }

        public static double[][] matrix_add_vector(in double[][] x, in double[] y) {
            return matrix_op_vector(in x, in y, operation.add);
        }
        public static double[][] matrix_subtract_vector(in double[][] x, in double[] y) {
            return matrix_op_vector(in x, in y, operation.subtract);
        }

        public static double[][] matrix_multiply(in double[][] x, in double[][] y) {
            var result = matrix_create(x.Length, x[0].Length);
            int row, col, i;
            double sum;

            for (row = 0; row < x.Length; row++) {
                for (col = 0; col < y[0].Length; col++) {
                    sum = 0d;
                    for (i = 0; i < x[0].Length; i++)
                        sum += x[row][i] * y[i][col];

                    x[row][col] = sum; 
                }
            }
            return result;
        }

        public static double sigmoid(in double value) {
            return 1.0d / (1.0d + Math.Exp(-value));
        }
        
        // sigma - standard deviation
        // X_norm = (X - mean) ./ sigma;
        public static void feature_normalize(in double[][] train_data, out double[] mean, out double[] sigma, out double[][] norm_train_data) {
            mean = new double[train_data[0].Length];
            sigma = new double[train_data[0].Length];
            double[][] temp;
            norm_train_data = null;

            var transposed_train_data = matrix_transpose(train_data);
            for (var i = 0; i < train_data[0].Length; ++i) {
                mean[i] = ml_funcs.mean(transposed_train_data[i]);
                sigma[i] = ml_funcs.stadard_deviation(transposed_train_data[i]);
            }

            // (X - mean)
            temp = matrix_subtract_vector(train_data, mean);
            // (X - mean) ./ sigma
            norm_train_data = matrix_divide_vector(temp, sigma);
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

            return Math.Sqrt(sum / array.Length);
        }

        // fmincg
        public static result_state rasmussen(in double[][] train_data, in int max_iterations) {
            var result = new result_state();
            int i = 0;
            const double RHO = 0.01, SIG = 0.5, INT = 0.1, EXT = 3.0, MAX = 20, RATION = 100; 
            double[][] temp_train_data;

            while (++i < max_iterations) {

            }

            return result;
        }

        // theta - parameters
        // X - features
        // lambda - cooeficient for regularization
        // h = sigmoid(X * theta) logistic regression hypothesis
        // J = (1/m .* (sum(-y .* log(h) .- (1 .- y) .* log(1 - h)))) + ((lambda/(2*m)) .* (sum(theta([2:size(theta)]) .^ 2)));
        public static result_state cost_logistic_regression_regularized(in double[][] train_data, in double[] result_data, in double[] theta, in double lambda, out double cost, out double[] gradient) {
            var result = new result_state();
            double temp = 0d, hypothesis = 0d, regularization = 0d;
            int i, t, train_data_count = train_data[0].Length;
            cost = 0d;
            gradient = new double[theta.Length];

            if (train_data.Length != result_data.Length)
                result.add_error("Arrays train_data and result_data should have same amount of entries");

            if (train_data_count != theta.Length-1)
                result.add_error("train_data should have one less size of theta entries");

            if (result.has_errors()) 
                return result;

            for (i = 0; i < train_data.Length; i++) {
                // X * theta
                temp = theta[0];
                for (t = 1; t < train_data_count + 1; t++)
                    temp += theta[t] * train_data[i][t - 1]; // skipping x(1)

                // (sum(theta([2:size(theta)]) .^ 2)));
                for (t = 1; t < theta.Length; t++)
                    regularization += Math.Pow(theta[t], 2);

                // (lambda/(2*m)) * sum(..)
                regularization = lambda/(2*train_data_count) * regularization;
                
                // h = sigmoid(X * theta);
                hypothesis = sigmoid(temp);
                // sum(-y .* log(h) .- (1 .- y) .* log(1 - h))
                cost += -result_data[i] * Math.Log(hypothesis) - (1 - result_data[i]) * Math.Log(1 - hypothesis);

                if (i > 0)  // we ignore bias unit (1) of theta when we regularize 
                    cost += regularization;

                // grad = (1/m .* sum((h .- y) .* X ))' + ((lambda/m) * theta);
                gradient[0] += hypothesis - result_data[i];
                for (t = 1; t < theta.Length; t++) { // skipping x(1)
                    gradient[t] += (hypothesis - result_data[i]) * train_data[i][t - 1];
                    if (i > 0) // we ignore bias unit (1) of theta when we regularize 
                        gradient[t] += (lambda/train_data_count) * theta[t];
                }
            }

            return result;
        }

        // h = sigmoid(X * theta);
        // J = 1/m .* (sum(-y .* log(h) .- (1 .- y) .* log(1 - h)));
        // grad = 1/m .* sum((h .- y) .* X );
        public static result_state cost_logistic_regression(in double[][] train_data, in double[] result_data, in double[] theta, out double cost, out double[] gradient) {
            var result = new result_state();
            double temp = 0d, hypothesis = 0d;
            cost = 0d;
            gradient = new double[theta.Length];
            int i, t;

            if (train_data.Length != result_data.Length)
                result.add_error("Arrays train_data and result_data should have same amount of entries");

            if (train_data[0].Length != theta.Length-1)
                result.add_error("train_data should have one less size of theta entries");

            if (result.has_errors()) 
                return result;

            for (i = 0; i < train_data.Length; ++i) {
                
                // X * theta
                temp = theta[0];
                for (t = 1; t < train_data[0].Length + 1; t++)
                    temp += theta[t] * train_data[i][t - 1]; // skipping x(1)
                
                // h = sigmoid(X * theta);
                hypothesis = sigmoid(temp);
                // sum(-y .* log(h) .- (1 .- y) .* log(1 - h))
                cost += -result_data[i] * Math.Log(hypothesis) - (1 - result_data[i]) * Math.Log(1 - hypothesis);
                
                // sum((h .- y) .* X; // grad = 1/m .* sum((h .- y) .* X );
                gradient[0] += hypothesis - result_data[i];
                for (t = 1; t < theta.Length; t++) // skipping x(1)
                    gradient[t] += (hypothesis - result_data[i]) * train_data[i][t - 1];
            }
            // 1/m .* sum(..)
            cost /= train_data.Length;

            // grad = 1/m .* sum..
            for (i = 0; i < gradient.Length; i++)
                gradient[i] /= train_data.Length;

            return result;
        }

        // m = training samples
        // cost function for linear regression J(theta) = 1/2*m * sum( hypothesis(x) - y)^2
        // hypothesis = theta'x = theta(0) + theta(x) x
        // J = sum((X*theta - y).^2) / (2*m);
        // 
        public static result_state cost_linear_regression(in double[][] train_data, in double[] result_data, in double[] theta, out double cost) {
            var result = new result_state();
            var temp = 0d;
            cost = 0;

            if (train_data.Length != result_data.Length)
                result.add_error("Arrays train_data and result_data should have same amount of entries");

            if (train_data[0].Length != theta.Length-1)
                result.add_error("train_data should have one less size of theta entries");

            if (result.has_errors()) 
                return result;

            for (var i = 0; i < train_data.Length; i++) {
                temp = theta[0];
                for (var t = 1; t < train_data[0].Length + 1; t++)
                    temp += theta[t] * train_data[i][t - 1]; // skipping x(1)
                    
                temp -= result_data[i];
                temp *= temp;
                cost += temp;
            }

            cost /= 2 * result_data.Length;
            return result;
        }

        // m number of training examples
        // ratio = alpha/m
        // t1 = theta(1) - ratio * sum( (X*theta - y) .* X(:,1) )
        // theta = (theta' - ((alpha/m) * sum( (X*theta - y) .* X )))';
        public static result_state gradient_descent(in double[][] train_data, in double[] result_data, in double[] theta, in double alpha, in int iterations, out double[] after_theta, out double[] costs) {
            var result = new result_state();
            after_theta = theta;
            costs = new double[iterations];
            var sub_temp = matrix_create(train_data.Length, theta.Length);
            var temp = 0d;
            int t, i, iter;
            double ratio = alpha / train_data.Length;

            if (train_data.Length != result_data.Length)
                result.add_error("Arrays train_data and result_data should have same amount of entries");

            if (train_data[0].Length != theta.Length-1)
                result.add_error("train_data should have one less size of theta entries");

            if (result.has_errors()) 
                return result;
            
            for (iter = 0; iter < iterations; iter++) {
                for (i = 0; i < train_data.Length; i++) {
                    temp = theta[0];

                    // X*theta
                    for (t = 1; t < train_data[0].Length + 1; t++)
                        temp += theta[t] * train_data[i][t - 1]; // skipping intercept term x(1) 
                    
                    // X*theta - y
                    temp -= result_data[i];
                    
                    // (X*theta - y) .* X
                    sub_temp[i][0] = temp;
                    for (t = 1; t < theta.Length; t++)
                        sub_temp[i][t] = train_data[i][t - 1] * temp;

                    // temp -= result_data[i];
                }

                // (alpha/m) * sum( (X*theta - y) .* X )
                for (i = 0; i < theta.Length; i++) {
                    temp = 0d;
                    for (t = 0; t < train_data.Length; t++)
                        temp += sub_temp[t][i];

                    after_theta[i] = after_theta[i] - ratio * temp;
                }
                
                cost_linear_regression(train_data, result_data, after_theta, out costs[iter]);
            }

            return result;
        }
    }
}
