# DONE
3. compute_cost_multi probably should be refactored to use matrix functions
    3. 1. No, because it's more effecient to go line by line as it flattens formula, matrix operations do not

# UNODE
1. put matrix functions to other file
2. refactor functions and add all arithmetic and scalar, so that opeartion happens through switch and perform matrix length checks
4. write tests for ml, matrix functions
5. check in IL if in for in gradient_descent does not create copies
    5. 1. Update: https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/keywords/in-parameter-modifier says it does not
6. writing formulas in code is awful, create a folder with images and add link in comments