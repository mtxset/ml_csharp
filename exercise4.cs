namespace ml {
    public static class exercise4 {
        public static void test_debug_nn() {
            const double lambda = 0;
            const int input_layer_size = 3;     // input layer size
            const int hidden_layer_size = 5;    // hidden nn units
            const int output_layer_size = 3;    // labels
            const int debug_training_examples = 5;

            double cost;
            double[][] theta1_gradient, theta2_gradient;

            var random_theta_1 = ml_funcs.matrix_transpose(ml_funcs.nn_debug_random_weights(hidden_layer_size, input_layer_size + 1));
            var random_theta_2 = ml_funcs.matrix_transpose(ml_funcs.nn_debug_random_weights(output_layer_size, hidden_layer_size + 1));
            var train_data = ml.ml_funcs.matrix_transpose(ml_funcs.nn_debug_random_weights(debug_training_examples, input_layer_size));
            var result_data = new double[] { 2, 3, 1, 2, 3 };

            // do I need to do rolling of thetas..?
            // var unroll_theta_1 = ml_funcs.matrix_unroll(random_theta_1);
            // var unroll_theta_2 = ml_funcs.matrix_unroll(random_theta_2);
            // var unrolled_theta = new double[unroll_theta_1.Length + unroll_theta_2.Length];
            // unroll_theta_1.CopyTo(unrolled_theta, 0);
            // unroll_theta_2.CopyTo(unrolled_theta, unroll_theta_1.Length);

            ml_funcs.nn_cost_two_layer(train_data, result_data, random_theta_1, random_theta_2, output_layer_size, lambda, out cost, out theta1_gradient, out theta2_gradient);
        }
    }
}