cmake_minimum_required(VERSION 3.5.1)
project(miniTest)

find_package(Celerity CONFIG REQUIRED)

add_executable(miniTest main.cpp)
add_celerity_to_target(TARGET miniTest SOURCES main.cpp)