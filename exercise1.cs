using System;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;

namespace ml {
    public static class exercise1 {

        public static void read_data_test() {
            var filepath = "./data/ex1data2.txt";
            var data = utils.file_utils.parse_file(filepath);

            Console.WriteLine(ml.ml_funcs.compute_cost_multi(
                data.train_data, 
                data.result_data, 
                new double[] { 340412.65957, 110631.05028, -6649.47427 }));
        }

        public static IPlotModel plot_data1() {
            var filepath = "./ex1/data1.txt";
            var file_data = utils.file_utils.read_file(filepath);

            if (file_data.failed) {
                Console.WriteLine($"error reading file {filepath}");
                return null;
            }

            var lines = file_data.result.Split("\n");

            double[] x = new double[lines.Length], y = new double[lines.Length];

            var model = new PlotModel { Title = "Test" };
            var scatter_series = new ScatterSeries { MarkerType = MarkerType.Cross };

            for (var i = 0; i < lines.Length - 1; i++) {
                x[i] = Convert.ToDouble(lines[i].Split(",")[0]);
                y[i] = Convert.ToDouble(lines[i].Split(",")[1]);
                // Console.WriteLine($"{i}: {x[i]}, {y[i]}");
                scatter_series.Points.Add( new ScatterPoint(x[i], y[i], 5, 5));
            }

            model.Series.Add(scatter_series);
            model.Axes.Add(new LinearAxis { 
                Position = AxisPosition.Left, Minimum = -5, Maximum = 25 });
            model.Axes.Add(new LinearAxis { 
                Position = AxisPosition.Bottom, Minimum = 5, Maximum = 25 });
            
            model.Axes.Add(new LinearColorAxis { 
                Position = AxisPosition.Right, Palette = OxyPalettes.Hue(10) });

            Console.WriteLine(ml_funcs.compute_cost(x, y, new double[] {-3.6303, 1.1664}));
            return model;
        }
    }
} 