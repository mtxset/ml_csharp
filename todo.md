# DONE
3. compute_cost_multi probably should be refactored to use matrix functions
	3. 1. No, because it's more effecient to go line by line as it flattens formula, matrix operations do not
5. check in IL if in for in gradient_descent does not create copies
	5. 1. Update: https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/keywords/in-parameter-modifier says it does not
	5. 2. Update: it only applies to value types; passed pointer is read-only but not values passed
	5. 3. Remove all unnecessary 'in', because performance wise is inconsequential and language-specific

# UNODE
1. put matrix functions to other file
2. refactor functions and add all arithmetic and scalar, so that opeartion happens through switch and perform matrix length checks
4. write tests for ml, matrix functions
6. writing formulas in code is awful, create a folder with images and add link in comments
7. need some functions to perfrom dimenion and length checks on matrices (cost_linear_regression length checks)
8. write tests with non 0 theta and add tests for feature_normalize, gradient_descent from ex1
9. write simple compression algorithm for numbers
10. comment every function
11. result_state add in every matrix function
12. result_state in functions should always combine but sometimes just throw
13. internal math/ml/matrix functions should not output
14. so far most bugs are because of matrices dimensions/"tranpose" being incorrect when performing some operation on them
15. when returning the result I'm not keeping a place where exactly error happened.
16. need to check file_utils parser to ignore empty lines (parse_file is incorrect because it takes assumes that last line is empty, just ignore empty lines)
17.
