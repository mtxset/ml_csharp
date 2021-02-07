namespace ml {
    public class result_state {
        private int index;
        private int max_error_count;

        public string[] error_list;

        public result_state(int error_count = 10) {
            error_list = new string[error_count];
            index = 0;
            max_error_count = error_count;
        }

        public void add_error(string error) {
            if (index + 1 > max_error_count)
                error_list[max_error_count-1] = "Could not fit all errors";
            else
                error_list[index++] = error;
        }

        public void combine_errors(result_state result) {
            if (!result.has_errors())
                return;
            
            foreach (var item in result.error_list)
                add_error(item);
        }

        public bool has_errors() {
            return index > 0;
        }

        public string get_errors() {
            string result = "";
            for (var i = 0; i < index; i++)
                result += error_list[i] + "\n";

            return result;
        }
    }
}