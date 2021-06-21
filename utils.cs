using System;
using System.IO;
using ml;

namespace utils {

	public static class string_utils {
		public static double[][] string_to_matrix(string data, string split_char = ",") {
			double[][] result = null;

			var lines = data.Split("\n");
			var data_size = lines[0].Trim().Split(split_char).Length;
			result = ml_funcs.matrix_create(lines.Length, data_size, -1);

			for (var row = 0; row < lines.Length; row++) {
				var temp_line = lines[row].Trim().Split(split_char);

				for (var col = 0; col < temp_line.Length; col++)
					result[row][col] = Convert.ToDouble(temp_line[col]);
			}
			return result;
		}
	}

	public static class file_utils {
		public static result_state parse_file(string file_path, out double[][] train_data, out double[] result_data, string split_char = ",") {

			var result = new result_state();
			string file_content;
			var file_handle = read_file(file_path, out file_content);
			train_data = null;
			result_data = null;

			if (file_handle.has_errors())
				return file_handle;

			var lines = file_content.Split("\n");
			// data, .. , result
			var data_size = lines[0].Split(split_char).Length - 1;

			train_data = ml_funcs.matrix_create(lines.Length - 1, data_size);
			result_data = new double[lines.Length - 1];

			for (var row = 0; row < lines.Length - 1; row++) {
				var temp_line = lines[row].Split(split_char);

				for (var col = 0; col < temp_line.Length - 1; col++)
					train_data[row][col] = Convert.ToDouble(temp_line[col]);

				result_data[row] = Convert.ToDouble(temp_line[temp_line.Length-1]);
			}

			return result;
		}

		public static result_state read_file(string file_path, out string file_content) {
			var result = new result_state();
			file_content = "";

			if (!File.Exists(file_path)) {
				result.add_error($"File {file_path} does not exist");
				return result;
			}

			var file = File.Open(file_path, FileMode.Open);

			var bytesRead = new byte[file.Length];
			file.Read(bytesRead, 0, (int)file.Length);
			file.Close();

			var charArray = new char[bytesRead.Length];

			for (int i = 0; i < bytesRead.Length; i++)
				charArray[i] = (char)bytesRead[i];

			file_content = string.Concat(charArray);
			return result;
		}
	}
}
