using System;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using utils;

namespace ml {
	public static class exercise1 {

		public static void read_data_test() {
			plot_data1();
			// var filepath = "./data/ex1data2.txt";
			// var data = utils.file_utils.parse_file(filepath);

			// Console.WriteLine(ml.ml_funcs.compute_cost_multi(
			//	 data.train_data,
			//	 data.result_data,
			//	 new double[] { 340412.65957, 110631.05028, -6649.47427 }));
		}

		public static void test() {
			var file_path = "./data/ex1data2.txt";
			double[][] train_data, norm_train_data;
			double[] result_data, mean, sigma;

			var parse_result = file_utils.parse_file(file_path, out train_data, out result_data);

			if (parse_result.has_errors()) {
				Console.WriteLine(parse_result.all_errors_to_string());
				return;
			}

			ml_funcs.feature_normalize(train_data, out mean, out sigma, out norm_train_data);

			double[] after_theta, costs;
			var theta = ml_funcs.gradient_descent(norm_train_data, result_data, new double[] { 0, 0, 0 }, 0.3, 400, out after_theta, out costs);
		}

		public static IPlotModel plot_data1() {
			var filepath = "./data/ex1data1.txt";
			double[][] train_data;
			double[] result_data;
			var file_read_result = utils.file_utils.parse_file(filepath, out train_data, out result_data);

			if (file_read_result.has_errors()) {
				Console.WriteLine(file_read_result.all_errors_to_string());
				return null;
			}

			var model = new PlotModel { Title = "Test" };
			var scatter_series = new ScatterSeries { MarkerType = MarkerType.Cross };

			for (var i = 0; i < train_data.Length; i++) {
				scatter_series.Points.Add(new ScatterPoint(train_data[i][0], result_data[i], 5, 5));
			}

			model.Series.Add(scatter_series);
			model.Axes.Add(new LinearAxis {
				Position = AxisPosition.Left, Minimum = -5, Maximum = 25 });
			model.Axes.Add(new LinearAxis {
				Position = AxisPosition.Bottom, Minimum = 5, Maximum = 25 });

			model.Axes.Add(new LinearColorAxis {
				Position = AxisPosition.Right, Palette = OxyPalettes.Hue(10) });

			double cost;
			var cost_result = ml_funcs.cost_linear_regression(train_data, result_data, new double[] {-3.6303, 1.1664}, out cost);

			if (cost_result.has_errors()) {
				Console.WriteLine(cost_result.all_errors_to_string());
				return null;
			}

			Console.WriteLine(cost);
			return model;
		}
	}
}
