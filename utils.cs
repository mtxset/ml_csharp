using System;
using System.IO;

namespace utils {
    public struct open_file {
        public bool failed;
        public long bytes_read;
        public string result;
    }

    public struct ml_data {
        public double[,] train_data;
        public double[] result_data;

        public void init(int train_data_x, int train_data_y) {
            train_data = new double[train_data_x, train_data_y];
            result_data = new double[train_data_x];
        }
    }

    public static class file_utils {
        public static ml_data parse_file(string filepath, string split_char = ",") {
            var result = new ml_data();
            var file_data = read_file(filepath);

            if (file_data.failed) {
                Console.WriteLine($"error reading file {filepath}");
                return result;
            }

            var lines = file_data.result.Split("\n");
            // data, .. , result
            var data_size = lines[0].Split(split_char).Length - 1;
            
            result.init(lines.Length - 1, data_size);

            for (var i = 0; i < lines.Length - 1; i++) {
                var temp_line = lines[i].Split(split_char);
                
                for (var x = 0; x < temp_line.Length - 1; x++)
                    result.train_data[i, x] = Convert.ToDouble(temp_line[x]);

                result.result_data[i] = Convert.ToDouble(temp_line[temp_line.Length-1]);
            }

            return result;
        }

        public static open_file read_file(string filepath) {
            var result = new open_file();

            if (!File.Exists(filepath)) {
                result.failed = true;
                return result;
            }

            var file = File.Open(filepath, FileMode.Open);

            result.bytes_read = file.Length;

            var bytesRead = new byte[file.Length];
            file.Read(bytesRead, 0, (int)file.Length);
            file.Close();

            var charArray = new char[bytesRead.Length];

            for (int i = 0; i < bytesRead.Length; i++)
                charArray[i] = (char)bytesRead[i];

            result.result = string.Concat(charArray);
            return result;
        }
    }
}