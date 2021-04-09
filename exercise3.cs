using System;
using System.IO;
using utils;

namespace ml {
	public static class exercise3 {
		public static void test_rasmussen(bool force_training = false) {
			var result = new result_state();
			const string trained_theta_file_path = "./data/ex3_trained_theta.txt";
			const string cost_progression_file_path = "./data/ex3_cost_progression.txt";
			const double lambda = 0.1d;
			const int max_iterations = 10;
			var file_path = "./data/ex3data1.txt";
			double[][] train_data, trained_theta, cost_progression;
			double[] result_data, label_result_data, initial_theta;
			int i, labels, label_train_count;
			int[] predict_indices;

			var parse_result = file_utils.parse_file(file_path, out train_data, out result_data);

			if (parse_result.has_errors()) {
				Console.WriteLine(parse_result.all_errors_to_string());
				return;
			}

			label_train_count = result_data.Length / 10; // how many training examples we have for each label (numbers from 0 to 9)

			trained_theta = ml_funcs.matrix_create(10, train_data[0].Length);
			cost_progression = ml_funcs.matrix_create(10, 10);

			// don't retrain
			// delete file if you want to retrain
			if (!File.Exists(trained_theta_file_path) || force_training) {
				for (labels = 0; labels < 10; labels++) {
					Console.WriteLine($"Training label: {labels}");
					label_result_data = new double[train_data.Length];
					for (i = 0; i < label_train_count; i++)
						label_result_data[label_train_count * labels + i] = 1;

					initial_theta = new double[train_data[0].Length + 1];
					result = ml_funcs.rasmussen(train_data, label_result_data, initial_theta, lambda, max_iterations, ml_funcs.cost_logistic_regression_regularized, out cost_progression[labels], out trained_theta[labels]);

					if (result.has_errors()) {
						Console.WriteLine(result.all_errors_to_string());
						return;
					}
				}

				result = ml.ml_funcs.matrix_to_csv(trained_theta_file_path, trained_theta);
				var write_vector_result = ml_funcs.matrix_to_csv(cost_progression_file_path, cost_progression);
				result.combine_errors(write_vector_result);

				if (result.has_errors()) {
					Console.WriteLine(result.all_errors_to_string());
					return;
				}
			} else {
				result = ml_funcs.matrix_from_csv(trained_theta_file_path, out trained_theta);

				if (result.has_errors()) {
					Console.WriteLine(result.all_errors_to_string());
					return;
				}
			}

			result = ml_funcs.predict_one_vs_all(trained_theta, train_data, out predict_indices);

			if (result.has_errors()) {
				Console.WriteLine(result.all_errors_to_string());
				return;
			}

			var correct_predictions = 0;
			for (i = 0; i < result_data.Length; i++) {
				if (result_data[i] == predict_indices[i])
					correct_predictions++;
			}

			double training_accuracy = (double)correct_predictions / (double)result_data.Length * 100;

			Console.Write("rasmussen");
			if (training_accuracy > 90d)
				Console.WriteLine(" .. OK");
			else
				Console.WriteLine(" .. FAILED");

			// Console.WriteLine($"Training accuracy with {max_iterations} iterations is {training_accuracy}%");
		}
	}
}
