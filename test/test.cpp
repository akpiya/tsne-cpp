#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <iostream>

#include "tsne.cpp"


TEST_CASE("4 Points into 2 Clusters") {
  vector<double> x1 = {1, 1, 1, 1, 1, 1, 1, 1};
  vector<double> x2 = {1, 0, 1, 0, 2, 1, 1, 1};
  vector<double> x3 = {3, 3, 3, 3, 3, 3, 3, 3};
  vector<double> x4 = {3, 2, 4, 3, 3, 2, 3, 4};
  vector<vector<double>> input;
  input.push_back(x1);
  input.push_back(x3);
  input.push_back(x2);
  input.push_back(x4);

  auto res = tsne(input, 2, 2.0);
  
  to_file(res, "/Users/Akash/Documents/Projects/CS/tsne-cpp/tmp/data");
}