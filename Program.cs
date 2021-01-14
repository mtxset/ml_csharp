using System;
using System.Linq;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Core.Drawing;
using OxyPlot.Series;

namespace ml {
    class Program {
        static void Main(string[] args) {

            var outputToFile = "./images/ex1.png";

            var width = 1024;
            var height = 768;
            var background = OxyColors.LightGray;
            var resolution = 96d;

            var model = exercise1.plot_data1();

            // export to file using static methods
            PngExporter.Export(model, outputToFile, width, height, background, resolution);
        }
        private static IPlotModel build_scatter() {
            var model = new PlotModel { Title = "ScatterSeries" };
            var scatterSeries = new ScatterSeries { MarkerType = MarkerType.Cross };
            var r = new Random(314);
            for (int i = 0; i < 100; i++)
            {
                var x = r.NextDouble();
                var y = r.NextDouble();
                var size = 5;
                var colorValue = r.Next(0, 10);
                scatterSeries.Points.Add(new ScatterPoint(x, y, size, 5));
            }

            model.Series.Add(scatterSeries);
            model.Axes.Add(new LinearColorAxis { Position = AxisPosition.Right, Palette = OxyPalettes.Hue(10) });
            model.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Minimum = -20, Maximum = 80});
            model.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Minimum = -10, Maximum = 10});

            return model;
        }

        private static IPlotModel BuildPlotModel() {
            var rand = new Random(21);

            var model = new PlotModel { Title = "Cake Type Popularity" };

            var cakePopularity = Enumerable.Range(1, 5).Select(i => rand.NextDouble()).ToArray();
            var sum = cakePopularity.Sum();
            var barItems = cakePopularity.Select(cp => RandomBarItem(cp, sum)).ToArray();
            var barSeries = new BarSeries
            {
                ItemsSource = barItems,
                LabelPlacement = LabelPlacement.Base,
                LabelFormatString = "{0:.00}%"
            };

            model.Series.Add(barSeries);

            model.Axes.Add(new CategoryAxis
            {
                Position = AxisPosition.Left,
                Key = "CakeAxis",
                ItemsSource = new[]
                {
                    "Apple cake",
                    "Baumkuchen",
                    "Bundt Cake",
                    "Chocolate cake",
                    "Carrot cake"
                }
            });
            return model;
        }

        private static BarItem RandomBarItem(double cp, double sum)
           => new BarItem { Value = cp / sum * 100, Color = RandomColor() };

        private static OxyColor RandomColor()
        {
            var r = new Random();
            return OxyColor.FromRgb((byte)r.Next(255), (byte)r.Next(255), (byte)r.Next(255));
        }
    }
    
}
