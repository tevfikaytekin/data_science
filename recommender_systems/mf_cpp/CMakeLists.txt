cmake_minimum_required(VERSION 3.20)
project(mf_cpp)

set(CMAKE_CXX_STANDARD 14)
include_directories(.)

add_executable(mf_cpp matrix_factorization.cpp)
