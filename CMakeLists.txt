cmake_minimum_required(VERSION 3.10)
project(TSNE VERSION 1.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Executable target
add_executable(tsne src/tsne.cpp)
