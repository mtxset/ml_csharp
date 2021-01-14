using System;
using System.IO;

namespace utils {
    public struct open_file {
        public bool failed;
        public long bytes_read;
        public string result;
    }

    public static class file_utils {
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