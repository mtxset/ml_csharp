using System;
using System.IO;

namespace ml {
    public static class ml_funcs {

        private enum operation { add, subtract, multiply, divide };

        public static bool matrix_compare_deep(double[][] x, double[][] y) {
            if (x.Length != y.Length || x[0].Length != y[0].Length)
                return false;

            int row, col;

            for (row = 0; row < x.Length; row++)
                for (col = 0; col < y[0].Length; col++)
                    if (x[row][col] != y[row][col])
                        return false;

            return true;
        }

        public static void min_value(double[] array, out double max_value, out int max_index) {
            double previous_value = array[0];
            max_index = 0;
            max_value = array[0];

            for (int i = 0; i < array.Length; ++i) {
                max_value = Math.Min(max_value, array[i]);
                
                if (previous_value != max_value)
                    max_index = i;

                previous_value = max_value;
            }
        } 

        public static void max_value(double[] array, out double max_value, out int max_index) {
            double previous_value = array[0];
            max_index = 0;
            max_value = array[0];

            for (int i = 0; i < array.Length; ++i) {
                max_value = Math.Max(max_value, array[i]);
                
                if (previous_value != max_value)
                    max_index = i;

                previous_value = max_value;
            }
        }

        public static result_state matrix_from_csv(string file_path, out double[][] matrix) {
            var result = new result_state();
            matrix = new double[0][];
            int row, col, column_size, row_size;
            string file_content;
            string[] buffer, inputs;
            bool parse_result;

            result = utils.file_utils.read_file(file_path, out file_content);

            if (result.has_errors())
                return result;

            buffer = file_content.Split("\r\n");
            row_size = buffer.Length - 1; // split adds empty last entry
            column_size = buffer[0].Split(",").Length;
            matrix = matrix_create(row_size, column_size);

            for (row = 0; row < row_size; row++) {
                inputs = buffer[row].Split(",");

                if (inputs.Length != column_size) {
                    result.add_error("Some of rows have more columns than expected. Column size assumed from first row.");
                    return result;
                }

                for (col = 0; col < column_size; col++) {
                    parse_result = double.TryParse(inputs[col], out matrix[row][col]);

                    if (!parse_result) {
                        result.add_error($"Could not parse: {inputs[col]}");
                        return result;
                    }
                }
            }
            
            return result;
        }

        public static result_state vector_to_csv(string file_path, double[] vector, bool overwrite = true) {
            var result = new result_state();

            if (!overwrite) {
                if (File.Exists(file_path)) {
                    result.add_error($"{file_path} already exists. Pass true for overwrite");
                    return result;
                }
            }

            string buffer = ""; 

            for (int i = 0; i < vector.Length; i++)
                buffer += vector[i].ToString() + ",";

            buffer = buffer.Remove(buffer.Length - 1); // removing last comma

            File.WriteAllText(file_path, buffer);

            return result;
        }

        public static result_state matrix_to_csv(string file_path, double[][] matrix, bool overwrite = true) {
            var result = new result_state();

            if (!overwrite) {
                if (File.Exists(file_path)) {
                    result.add_error($"{file_path} already exists. Pass true for overwrite");
                    return result;
                }
            }

            var buffer = new string[matrix.Length];

            for (int row = 0; row < matrix.Length; row++) {
                buffer[row] = "";

                for (int col = 0; col < matrix[0].Length; col++)
                    buffer[row] += matrix[row][col].ToString() + ",";

                buffer[row] = buffer[row].Remove(buffer[row].Length - 1); // removing last comma
            }

            File.WriteAllLines(file_path, buffer);

            return result;
        }

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

        public static double[][] matrix_transpose(double[][] matrix) {
            var result = matrix_create(matrix[0].Length, matrix.Length);
            int row, col;

            for (row = 0; row < matrix.Length; row++) {
                for (col = 0; col < matrix[0].Length; col++) {
                    result[col][row] = matrix[row][col];
                }
            }
            return result;
        }

        public static double[][] matrix_subtract(double[][] x, double[][] y) {
            var result = matrix_create(x.Length, x[0].Length);
            int row, col;

            for (row = 0; row < x.Length; row++) {
                for (col = 0; col < x[0].Length; col++) {
                    result[row][col] = x[row][col] - y[row][col]; 
                }
            }
            return result;
        }

        public static double[][] matrix_add(double[][] x, double[][] y) {
            var result = matrix_create(x.Length, x[0].Length);
            int row, col;

            for (row = 0; row < x.Length; row++) {
                for (col = 0; col < x[0].Length; col++) {
                    result[row][col] = x[row][col] + y[row][col]; 
                }
            }
            return result;
        }

        private static double[][] matrix_op_vector(double[][] x, double[] y, operation op) {
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

        public static double[][] matrix_divide_vector(double[][] x, double[] y) {
            return matrix_op_vector(x, y, operation.divide);
        }

        public static double[][] matrix_multiply_vector(double[][] x, double[] y) {
            return matrix_op_vector(x, y, operation.multiply);
        }

        public static double[][] matrix_add_vector(double[][] x, double[] y) {
            return matrix_op_vector(x, y, operation.add);
        }

        public static double[][] matrix_subtract_vector(double[][] x, double[] y) {
            return matrix_op_vector(x, y, operation.subtract);
        }

        public static double[] matrix_column_to_vector(double[][] matrix, int column) { 
            var result = new double[matrix.Length];

            for (var i = 0; i < matrix.Length; i++)
                result[i] = matrix[i][column];
    
            return result;
        }

        public static double[] matrix_row_to_vector(double[][] matrix, int row) {
            var result = new double[matrix[0].Length];

            for (var i = 0; i < matrix.Length; i++)
                result[i] = matrix[row][i];

            return result;
        }

        public static result_state matrix_multiply(double[][] x, double[][] y, out double[][] output) {
            var result = new result_state();
            output = matrix_create(x.Length, y[0].Length);
            
            if (x[0].Length != y.Length) {
                result.add_error("The number of columns of the x matrix must equal the number of rows of the y matrix.");
                return result;
            }

            int row, col, i;
            double sum;

            for (row = 0; row < x.Length; row++) {
                for (col = 0; col < y[0].Length; col++) {
                    sum = 0d;
                    for (i = 0; i < x[0].Length; i++)
                        sum += x[row][i] * y[i][col];

                    output[row][col] = sum; 
                }
            }
            return result;
        }

        public static double sigmoid(double value) {
            return 1.0d / (1.0d + Math.Exp(-value));
        }
        
        // sigma - standard deviation
        // X_norm = (X - mean) ./ sigma;
        public static void feature_normalize(double[][] train_data, out double[] mean, out double[] sigma, out double[][] norm_train_data) {
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

        public static double mean(double[] array) {
            var sum = 0d;

            for (var i = 0; i < array.Length; ++i)
                sum += array[i];

            return sum/array.Length; 
        }

        public static double stadard_deviation(double[] array) {
            var sum = 0d;
            var mu = mean(array);

           for (int i = 0; i < array.Length; i++)
                sum += Math.Pow(array[i] - mu, 2);

            return Math.Sqrt(sum / array.Length);
        }



        // Feature mapping function to polynomial features
        // Returns a new feature array with more features, comprising of 
        //   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
        public static result_state map_feature(double[] train_data_x, double[] train_data_y, out double[][] train_data, int degree = 6) {
            var result = new result_state();
            int column_size = (degree + 1) * (degree + 2) / 2, x, y, i, d;
            train_data = matrix_create(train_data_x.Length, column_size - 1, 1);

            if (train_data_x.Length != train_data_y.Length) {
                result.add_error("both arrays should be equal");
                return result;
            }

            d = 0;
            for (x = 1; x <= degree; x++) { // we don't need intercept term
                for (y = 0; y <= x; y++) {
                    for (i = 0; i < train_data_x.Length; i++) {
                        // (X1.^(i-j)).*(X2.^j)
                        train_data[i][d] = Math.Pow(train_data_x[i], x-y) * Math.Pow(train_data_y[i], y);
                    }
                    d++;
                }
            }

            return result;
        } 

        // [p_max, p_index] = max(sigmoid((X * all_theta')),[], 2);
        public static result_state predict_one_vs_all(double[][] trained_theta, double[][] train_data, out int[] predict_indices) {
            var result = new result_state();

            int row, col;
            double predict_value;
            var temp_matrix = matrix_create(train_data.Length, trained_theta.Length);
            var train_data_with_term = matrix_create(train_data.Length, train_data[0].Length + 1);
            predict_indices = new int[train_data.Length];

            // adding term to training data
            for (row = 0; row < train_data.Length; row++) {
                train_data_with_term[row][0] = 1; 
                for (col = 1; col < train_data[0].Length; col++) {
                    train_data_with_term[row][col] = train_data[row][col - 1]; // offsetting col = 1
                }
            }

            // X * all_theta'
            result = matrix_multiply(train_data_with_term, matrix_transpose(trained_theta), out temp_matrix);

            if (result.has_errors()) 
                 return result;

            // sigmoid(X * all_theta')
            for (row = 0; row < temp_matrix.Length; row++) {
                for (col = 0; col < temp_matrix[0].Length; col++) {
                    temp_matrix[row][col] = sigmoid(temp_matrix[row][col]);
                }
                // max(..)
                max_value(temp_matrix[row], out predict_value, out predict_indices[row]);
            }

            return result;
        }

        // (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen
        // Ported to C#
        // Minimize a continuous differentialble multivariate function
        // Conjugate gradient implementation https://en.wikipedia.org/wiki/Conjugate_gradient_method
        // fmincg
        public static result_state rasmussen(double[][] train_data, double[] result_data, double[] initial_theta, double lambda, int max_iterations, out double[] cost_progress, out double[] new_theta) {
            var result = new result_state();
            cost_progress = new double[max_iterations];
            bool ls_failed = false, success;
            int i, c, feature_count = initial_theta.Length;
            const double RHO = 0.01, SIG = 0.5, INT = 0.1, EXT = 3.0, MAX = 20, RATIO = 100;
            double f1, f2, f3, d1 = 0d, d2, d3, z1 = 0d, z2, z3, f0, limit, A, B, C, temp, M;
            double[] df1, df0 = new double[feature_count], df2, s, X0 = new double[feature_count], X = new double[feature_count];
            new_theta = X0;

            // value(cost) and gradient
            result = cost_logistic_regression_regularized(train_data, result_data, X, lambda, out f1, out df1);

            if (result.has_errors())
                return result;

            s = new double[df1.Length];
            for (i = 0; i < df1.Length; i++) {
                s[i] = -df1[i]; // search direction is steepest
                d1 += -df1[i] * df1[i]; // slope
            }

            z1 = 1 / (1 - d1);

            for (c = 0; c < feature_count; c++)
                X[c] = initial_theta[c];

            i = 0;
            while (i < max_iterations) {
                for (c = 0; c < feature_count; c++) {
                    X0[c] = X[c];
                    df0[c] = df1[c];
                }
                f0 = f1;

                // begin line search
                for (c = 0; c < X.Length; c++)
                    X[c] += z1 * s[c];

                result = cost_logistic_regression_regularized(train_data, result_data, X, lambda, out f2, out df2);

                if (result.has_errors())
                    return result;

                d2 = 0d;
                for (c = 0; c < feature_count; c++)
                    d2 += df2[c] * s[c];

                // initialize point 3 equal to point 1
                f3 = f1;
                d3 = d1;
                z3 = -z1;

                M = MAX;
                success = false;
                limit = -1;
                z2 = 0d; 
                A = B = C = 0;
 
                while (true) {
                    while ((f2 > f1 + z1 * RHO * d1 || d2 > -SIG * d1) && M > 0) {
                        limit = z1;
                        if (f2 > f1) // quadratic fit
                            z2 = z3 - (0.5 * d3 * z3 * z3) / (d3 * z3 + f2 - f3);
                        else { 
                            // cubic fit
                            A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3);
                            B = 3 * (f3 - f2) -z3* (d3 + 2 * d2);
                            z2 = (Math.Sqrt(B * B - A * d2 * z3 *z3 ) - B) / A; 
                        }

                        if (double.IsNaN(z2) || double.IsInfinity(z2))
                            z2 = z3 / 2; // if we had a numerical problem then bisect

                        z2 = Math.Max(Math.Min(z2, INT * z3), (1 - INT) * z3);
                        z1 += z2; 
                        for (c = 0; c < feature_count; c++)
                            X[c] += z2 * s[c];

                        result = cost_logistic_regression_regularized(train_data, result_data, X, lambda, out f2, out df2);

                        if (result.has_errors())
                            return result;
                        
                        M--;
                        d2 = 0d;
                        for (c = 0; c < feature_count; c++)
                            d2 += df2[c] * s[c];

                        z3 = z3 - z2;
                    }

                    if (f2 > f1 + z1 * RHO * d1 || d2 > -SIG * d1)
                        break;
                    else if (d2 > SIG * d1) {
                        success = true; 
                        break;
                    }
                    else if (M == 0)
                        break;

                    A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3);                     // make cubic extrapolation
                    B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2);
                    z2 = -d2 * z3 *z3 / (B + Math.Sqrt(B*B - A* d2 * z3 *z3));  // num. error possible - ok!

                    if (!(B*B -A * d2 * z3*z3 >= 0) || double.IsNaN(z2) || double.IsInfinity(z2) || z2 < 0) {
                        if (limit < -0.5)                                       // if we have no upper limit
                            z2 = z1 * (EXT - 1);                                // the extrapolate the maximum amount
                        else
                            z2 = (limit - z1) / 2;                              // otherwise bisect
                    }
                    else if (limit > -0.5 && z2 + z1 > limit)                   // extraplation beyond max?
                        z2 = (limit-z1) / 2;                                    // bisect
                    else if (limit < -0.5 && z2 + z1 > z1 * EXT)                // extrapolation beyond limit
                        z2 = z1 * (EXT - 1.0);                                  // set to extrapolation limit
                    else if (z2 < -z3 * INT)
                        z2 = -z3 * INT;
                    else if (limit > -0.5 && z2 < (limit - z1) *(1.0 - INT))    // too close to limit?
                        z2 = (limit - z1) * (1.0 - INT);

                    f3 = f2; d3 = d2; z3 = -z2;                                 // set point 3 equal to point 2
                    z1 += z2;                                                   // update current estimates
                    for (c = 0; c < feature_count; c++)
                        X[c] += z2 * s[c]; 

                    result = cost_logistic_regression_regularized(train_data, result_data, X, lambda, out f2, out df2);

                    if (result.has_errors())
                        return result;

                    M--;

                    d2 = 0d;
                    for (c = 0; c < feature_count; c++)
                        d2 += df2[c] * s[c];
                }                                                               // end of line search

                if (success) {                                                  // if line search succeeded
                    f1 = f2;                                                    // f1 - cost
                    cost_progress[i] = f1;

                    A = B = C = 0;

                    // Polack-Ribiere direction
                    for (c = 0; c < s.Length; c++) {
                        A += df1[c] * df1[c];
                        B += df2[c] * df2[c];
                        C += df1[c] * df2[c];
                    }

                    for(c = 0; c < s.Length; c++)
                        s[c] = ((B - C)/ A) * s[c] - df2[c];

                    // swap derivatives
                    for (c = 0; c < s.Length; c++) {
                        temp = df1[c];
                        df1[c] = df2[c];
                        df2[c] = temp;
                    }

                    d2 = 0d;
                    for (c = 0; c < feature_count; c++)
                        d2 += df1[c] * s[c];

                    if (d2 > 0) {
                        d2 = 0;
                        for (c = 0; c < feature_count; c++) {
                            s[c] = -df1[c];                 // new slope must be negative
                            d2 += -s[c] * s[c];             // otherwise use steepest direction
                        }
                    }

                    z1 = z1 * Math.Min(RATIO, d1 / (d2 + double.Epsilon));
                    d1 = d2;
                    ls_failed = false;
                } else {
                    for (c = 0; c < feature_count; c++) {
                        X[c] = X0[c];
                        df1[c] = df0[c];
                    }
                    f1 = f0;

                    if (ls_failed)
                        break;

                    // swap derivatives
                    for (c = 0; c < s.Length; c++) {
                        temp = df1[c];
                        df1[c] = df2[c];
                        df2[c] = temp;
                    }

                    d1 = 0;
                    for (c = 0; c < df1.Length; c++) {
                        s[c] = -df1[c];                 // try steepest
                        d1 += -s[c] * s[c];
                    }
                    z1 = 1 / (1 - d1);
                    ls_failed = true;
                }
                i++;
            }

            new_theta = X0;

            return result;
        }

        // theta - parameters
        // X - features
        // lambda - cooeficient for regularization
        // h = sigmoid(X * theta) logistic regression hypothesis
        // J = (1/m .* (sum(-y .* log(h) .- (1 .- y) .* log(1 - h)))) + ((lambda/(2*m)) .* (sum(theta([2:size(theta)]) .^ 2)));
        public static result_state cost_logistic_regression_regularized(double[][] train_data, double[] result_data, double[] theta, double lambda, out double cost, out double[] gradient) {
            var result = new result_state();
            double temp = 0d, hypothesis = 0d, regularization = 0d;
            int i, t, train_data_count = train_data.Length;
            cost = 0d;
            gradient = new double[theta.Length];
            var gradient_regularization = new double[theta.Length]; 

            if (train_data.Length != result_data.Length)
                result.add_error("Arrays train_data and result_data should have same amount of entries");

            if (train_data[0].Length != theta.Length-1)
                result.add_error("train_data column count should have one less size of theta size");

            if (result.has_errors()) 
                return result;

            for (i = 0; i < train_data.Length; i++) {
                // X * theta
                temp = theta[0];
                for (t = 1; t < train_data[0].Length + 1; t++)
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

                gradient[0] += hypothesis - result_data[i]; // x(1) term is 1
                // grad = (1/m .* sum((h .- y) .* X ))' + ((lambda/m) * theta);
                for (t = 1; t < theta.Length; t++) { // skipping x(1)
                    gradient[t] += (hypothesis - result_data[i]) * train_data[i][t - 1];
                }
            }

            // 1/m .* sum(..)
            cost /= train_data_count;

            // grad = 1/m .* sum.. + ((lambda/m) * theta); adding regularization
            gradient[0] /= train_data_count;
            for (i = 1; i < gradient.Length; i++) // skipping first one
                gradient[i] = (gradient[i] / train_data_count) + (lambda/train_data_count * theta[i]);

            return result;
        }

        // h = sigmoid(X * theta);
        // J = 1/m .* (sum(-y .* log(h) .- (1 .- y) .* log(1 - h)));
        // grad = 1/m .* sum((h .- y) .* X );
        public static result_state cost_logistic_regression(double[][] train_data, double[] result_data, double[] theta, out double cost, out double[] gradient) {
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
                
                // sum((h .- y) .* X; // full formula: grad = 1/m .* sum((h .- y) .* X );
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
        public static result_state cost_linear_regression(double[][] train_data, double[] result_data, double[] theta, out double cost) {
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
        public static result_state gradient_descent(double[][] train_data, double[] result_data, double[] theta, double alpha, int iterations, out double[] after_theta, out double[] costs) {
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
