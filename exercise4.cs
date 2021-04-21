using System;

namespace ml {
	public static class exercise4 {

		public static void test_flatten_unflatten() {
			Console.Write("test_flatten_unflatten");
			const int input_layer_size = 3;
			const int hidden_layer_size = 5;
			const int output_layer_size = 3;
			const int debug_training_examples = 5;

			var random_theta_1 = ml_funcs.nn_debug_random_weights(hidden_layer_size, input_layer_size + 1);
			var random_theta_2 = ml_funcs.nn_debug_random_weights(output_layer_size, hidden_layer_size + 1);
			var train_data = ml_funcs.matrix_transpose(ml_funcs.nn_debug_random_weights(debug_training_examples, input_layer_size));

			var unrolled_theta = ml_funcs.matrix_flatten_two(random_theta_1, random_theta_2);

			if (unrolled_theta.Length != (input_layer_size + 1) * hidden_layer_size + (hidden_layer_size + 1) * output_layer_size) {
				Console.WriteLine(" .. FAILED");
				return;
			}

			var theta1 = ml_funcs.matrix_unflatten(unrolled_theta, hidden_layer_size, 0, (input_layer_size + 1) * hidden_layer_size - 1);
			if (!ml_funcs.matrix_compare_deep(random_theta_1, theta1)) {
				Console.WriteLine(" .. FAILED");
				return;
			}

			var theta2 = ml_funcs.matrix_unflatten(unrolled_theta, output_layer_size, (input_layer_size + 1) * hidden_layer_size);
			if (!ml_funcs.matrix_compare_deep(random_theta_2, theta2)) {
				Console.WriteLine(" .. FAILED");
				return;
			}

			Console.WriteLine(" .. OK");

		}

		public static void test_debug_nn() {
			const double lambda = 0;
			const int input_layer_size = 3;
			const int hidden_layer_size = 5;
			const int output_layer_size = 3;
			const int debug_training_examples = 5;
			const int max_iterations = 10;

			double cost;
			double[] unrolled_theta = null, trained_theta = null;
			double[][] theta1_gradient, theta2_gradient;

			var random_theta_1 = ml_funcs.nn_debug_random_weights(hidden_layer_size, input_layer_size + 1);
			var random_theta_2 = ml_funcs.nn_debug_random_weights(output_layer_size, hidden_layer_size + 1);
			var train_data = ml_funcs.matrix_transpose(ml_funcs.nn_debug_random_weights(debug_training_examples, input_layer_size));
			var result_data = new double[] { 2, 3, 1, 2, 3 };

			unrolled_theta = ml_funcs.matrix_flatten_two(random_theta_1, random_theta_2);

			// I don't know how to make it simple
			ml_funcs.cost_delegate nn_cost_delegate = (double[][] train_data, double[] result_data, double[] theta, double lambda, out double cost, out double[] gradient) => {
				var result = new result_state();
				cost = 0;
				gradient = null;

				// 1. convert theta back to neural network layers
				var theta1 = ml_funcs.matrix_unflatten(theta, hidden_layer_size, 0, (input_layer_size + 1) * hidden_layer_size - 1);
				var theta2 = ml_funcs.matrix_unflatten(theta, output_layer_size, (input_layer_size + 1) * hidden_layer_size);

				// 2. pass to nn_cost_two_layer neural network thetas
				result = ml_funcs.nn_cost_two_layer(train_data, result_data, theta1, theta2, output_layer_size, lambda, out cost, out theta1_gradient, out theta2_gradient);

				if (result.has_errors()) {
					Console.WriteLine(result.all_errors_to_string());
					return result;
				}

				// 3. we get theta1, theta2 gradients then we flattan them into gradient
				gradient = ml_funcs.matrix_flatten_two(theta1_gradient, theta2_gradient);

				return result;
			};

			var result = ml_funcs.nn_cost_two_layer(train_data, result_data, random_theta_1, random_theta_2, output_layer_size, lambda, out cost, out theta1_gradient, out theta2_gradient);
			//var result = ml_funcs.rasmussen(train_data, result_data, unrolled_theta, lambda, max_iterations, nn_cost_delegate, out cost, out trained_theta);
		}
	}
}
