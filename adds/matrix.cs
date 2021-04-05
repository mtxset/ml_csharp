/*
I have a simple problem and a solution, but decision may have ramifications

I need to create matrix stuff; there are few ways of doing it (c#):
1. double[][], arrays of arrays - all languages have arrays, no abstractions, probably won't change ever, usage is different in every language, memory and clock wise is good
2. double[,] multidimensional arrays - not all languages have them, thus abstraction is introduced, memory and clock wise is not good
3. struct/class matrix - and provide access through matrix.row[1].col[1] or something, adds abstraction layer, usage could be same in every language, memory and clock is good because it's just hidden array of arrays

currently I write them in c#, but I may at some point rewrite it in python or c/cpp

it's such a simple thing but it's data structure which will be used everywhere above, thus changing it at some point will introduce tragical amount of boring work 
*/
namespace ml {
    public struct matrix_row {
        public double[] col;
    }

    public struct matrix {
        public matrix_row[] row;
    }

    public class matrixf {

        public static matrix create_matrix(int rows, int columns) {
            var result = new matrix();
            int r, c;

            result.row = new matrix_row[rows];

            for (r = 0; r < rows; r++) {
                result.row[r] = new matrix_row();
                result.row[r].col = new double[columns];
                
                for (c = 0; c < columns; c++)
                    result.row[r].col[c] = 0;
            }

            return result;
        }
    }
    
}