# Project README

## Environment Setup
1. Clone the project to your local repository.  
2. Install the required libraries using the command line:  
   ```bash
   pip install numpy pandas shapely pycryptodome
   
## Code Execution
# 1. Notes
The original datasets include twitter, newyork, and paris, each with a corresponding query file (the file name contains the keyword query).
The current experimental settings are:
Maximum secure index size: Kmax = 80
Representative point of the grid: center
Number of neighbors for each representative point: 10

# 2. quad_index.generate_databases
1) invert_database.py
Select any original dataset.
Manually modify the output path and file name of the inverted index.
Outputs the generation time of the inverted index file.

2) quad_database.py
Generates unencrypted index files.
You can modify Kmax and the output path.
All three datasets will be processed in a loop, and file names are automatically concatenated.
By default, the representative point is set to center.
To use centroid instead, change the field algorithm_type to 'centroid' and switch the result_points setting (comment/uncomment accordingly).
Outputs the generation time of the quadtree index file.

3) hash_database.py
Modify the output path of the encrypted index file.
All three datasets will be processed in a loop, and file names are automatically concatenated.
Simply execute to obtain the encrypted index file.
Outputs both the generation time and the number of rows in the encrypted index.

4) testfile_database.py
Generates test files.
You can define the number of query keywords.
The format is:
keyword1 keyword2 ... keywordn querypoint

# 2. quad_index.generate_csv
1) csv_file.py
Modify the output path of the result statistics file.
Generates a template for the final statistics file.

# 3. quad_index.exp_test
1) correct_testPlus_knn.py
Controls looping with K_values = [2, 4, 6, 8, 10].
You can modify the dataset, query set, dataset_name, algorithm_type, use_own_file, and same_file_k to control the testing method and files used.
Modify EXPERIMENT_GROUP to control the experimental group (this will be recorded in the output CSV file).
Some statistical results will be automatically written into the CSV file.

2) correct_testPlus_range_knn.py
Highly automated.
Runs tests on all three datasets with K_values = [2, 4, 6, 8, 10].
Modify exp_group to control the experimental group.
Some statistical results will be automatically written into the CSV file.
