using System;
using utils;
using static ml.ml_funcs;

namespace ml {
	public static class exercise4 {

		public static void test_nn() {
			result_state result;
			string file_content_1, file_content_2;

			Console.Write("test training number recognition ");

			result = file_utils.read_file("./data/ex4_theta_1.txt", out file_content_1);
			result.combine_errors(utils.file_utils.read_file("./data/ex4_theta_2.txt", out file_content_2));
			if (result.has_errors()) {
				Console.WriteLine(result.all_errors_to_string());
				Console.WriteLine(" .. FAILED");
				return;
			}

			var theta_1 = string_utils.string_to_matrix(file_content_1, " ");
			var theta_2 = string_utils.string_to_matrix(file_content_2, " ");

			var unrolled_theta = matrix_flatten_two(theta_1, theta_2, flatten_direction.by_column);

			Console.WriteLine(" .. OK");
		}

		public static void test_flatten_unflatten() {
			Console.Write("test_flatten_unflatten");
			const int input_layer_size = 3;
			const int hidden_layer_size = 5;
			const int output_layer_size = 3;
			const int debug_training_examples = 5;

			var random_theta_1 = nn_debug_random_weights(hidden_layer_size, input_layer_size + 1);
			var random_theta_2 = nn_debug_random_weights(output_layer_size, hidden_layer_size + 1);
			var train_data = matrix_transpose(nn_debug_random_weights(debug_training_examples, input_layer_size));

			var unrolled_theta = matrix_flatten_two(random_theta_1, random_theta_2);

			if (unrolled_theta.Length != (input_layer_size + 1) * hidden_layer_size + (hidden_layer_size + 1) * output_layer_size) {
				Console.WriteLine(" .. FAILED");
				return;
			}

			var theta1 = matrix_unflatten(unrolled_theta, hidden_layer_size, 0, (input_layer_size + 1) * hidden_layer_size - 1);
			if (!matrix_compare_deep(random_theta_1, theta1)) {
				Console.WriteLine(" .. FAILED");
				return;
			}

			var theta2 = matrix_unflatten(unrolled_theta, output_layer_size, (input_layer_size + 1) * hidden_layer_size);
			if (!matrix_compare_deep(random_theta_2, theta2)) {
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
			double[] unrolled_theta = null, nn_gradient = null;
			double[][] theta1_gradient, theta2_gradient;

			var random_theta_1 = nn_debug_random_weights(hidden_layer_size, input_layer_size + 1);
			var random_theta_2 = nn_debug_random_weights(output_layer_size, hidden_layer_size + 1);
			var train_data = matrix_transpose(nn_debug_random_weights(debug_training_examples, input_layer_size));
			var result_data = new double[] { 2, 3, 1, 2, 3 };

			result_state result;

			unrolled_theta = matrix_flatten_two(random_theta_1, random_theta_2);

			// I don't know how to make it simple
			cost_delegate nn_cost_delegate = (double[][] train_data, double[] result_data, double[] theta, double lambda, out double cost, out double[] gradient) => {
				var result = new result_state();
				cost = 0;
				gradient = null;

				// 1. convert theta back to neural network layers
				var theta1 = matrix_unflatten(theta, hidden_layer_size, 0, (input_layer_size + 1) * hidden_layer_size - 1);
				var theta2 = matrix_unflatten(theta, output_layer_size, (input_layer_size + 1) * hidden_layer_size);

				// 2. pass to nn_cost_two_layer neural network thetas
				result = nn_cost_two_layer(train_data, result_data, theta1, theta2, output_layer_size, lambda, out cost, out theta1_gradient, out theta2_gradient);

				if (result.has_errors()) {
					Console.WriteLine(result.all_errors_to_string());
					return result;
				}

				// 3. we get theta1, theta2 gradients then we flattan them into gradient
				gradient = matrix_flatten_two(theta1_gradient, theta2_gradient);

				return result;
			};

			Console.Write("nn_cost_two_layer");
			{
				result = nn_cost_two_layer(train_data, result_data, random_theta_1, random_theta_2, output_layer_size, lambda, out cost, out theta1_gradient, out theta2_gradient);
				nn_gradient = matrix_flatten_two(theta1_gradient, theta2_gradient);
				if (result.has_errors()) {
						Console.WriteLine(result.all_errors_to_string());
						Console.WriteLine(" .. FAILED");
						return;
				} else if (Math.Round(cost, 4) != 2.101) {
					Console.WriteLine(" .. FAILED");
					return;
				}

				Console.WriteLine(" .. OK");
			}

			// Console.Write("numerical gradient test");
			// {
			// 	double[] numerical_gradient = new double[unrolled_theta.Length];
			// 	double[] perturbation = new double[unrolled_theta.Length];
			// 	double[][] theta_1, theta_2;
			// 	double exponent = 1e-4;
			// 	double cost_1, cost_2;

			// 	// bool success = true;

			// 	for (var i = 0; i < unrolled_theta.Length; i++) {
			// 		perturbation[i] = exponent;
			// 		theta_1 = matrix_subtract_vector_scalar(random_theta_1, perturbation);
			// 		theta_2 = matrix_subtract_vector_scalar(random_theta_2, perturbation);
			// 		nn_cost_two_layer(train_data, result_data, theta_1, theta_2, output_layer_size, lambda, out cost_1, out theta1_gradient, out theta2_gradient);

			// 		theta_1 = matrix_add_vector_scalar(random_theta_1, perturbation);
			// 		theta_2 = matrix_add_vector_scalar(random_theta_2, perturbation);
			// 		nn_cost_two_layer(train_data, result_data, theta_1, theta_2, output_layer_size, lambda, out cost_2, out theta1_gradient, out theta2_gradient);

			// 		numerical_gradient[i] = (cost_2 - cost_1) / (2 * exponent);
			// 		perturbation[i] = 0;
			// 	}

			// 	var diff_a = new double[unrolled_theta.Length];
			// 	var diff_b = new double[unrolled_theta.Length];
			// 	for (var i = 0; i < unrolled_theta.Length; i++) {
			// 		diff_a[i] = numerical_gradient[i] - nn_gradient[i];
			// 		diff_b[i] = numerical_gradient[i] + nn_gradient[i];
			// 	}
			// 	var norm = vector_norm(diff_a) / vector_norm(diff_b);

			// 	// if (!success) {
			// 	// 	Console.WriteLine(" .. FAILED");
			// 	// 	Console.WriteLine(result.all_errors_to_string());
			// 	// 	return;
			// 	// }
			// }
			Console.Write("test gradient norm ");
			{
				double[] test_nn_gradient = new double[nn_gradient.Length];
				string file_output;
				int precision = 10;
				result = file_utils.read_file("./data/ex4_nn_gradient.txt", out file_output);
				if (result.has_errors()) {
					Console.WriteLine(result.all_errors_to_string());
					Console.WriteLine(" .. FAILED");
					return;
				}

				var lines = file_output.Split("\n");
				if (lines.Length != nn_gradient.Length) {
					Console.WriteLine($" .. FAILED. lines ({lines.Length}) and nn_gradient ({nn_gradient.Length}) should be equal.");
					return;
				}

				for (var i = 0; i < lines.Length; i++) {
					test_nn_gradient[i] = Convert.ToDouble(lines[i]);

					if (Math.Round(test_nn_gradient[i], precision) != Math.Round(nn_gradient[i], precision)) {
						Console.WriteLine(" .. FAILED");
						return;
					}
				}
				Console.WriteLine(" .. OK");
			}



			// next computeNumericalGradient
			// var result = rasmussen(train_data, result_data, unrolled_theta, lambda, max_iterations, nn_cost_delegate, out cost, out trained_theta);
		}
	}
}
