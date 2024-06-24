cmake_minimum_required(VERSION 3.16)

project(HighOrderTFEM)

# # Specify the C++ standard
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

# Make library
add_library(lib)

# Add includes
target_include_directories(lib PUBLIC include)

# Find libraries
find_package(Kokkos CONFIG REQUIRED)
target_link_libraries(lib Kokkos::kokkos)

# Make our main/demo executable
add_executable(demo main.cpp)
target_link_libraries(demo lib)

# Add sources
add_subdirectory(./src)

