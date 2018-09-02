cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(mKit VERSION 0.1.0 LANGUAGES CXX)

# Ensure we are in the correct build directory
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In source builds not allowed. Please make a new directory(called a build directory) and run CMake from there")
endif()

set(CMAKE_CXX_STANDARD 11)

# Add in the extra cmake modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(GNUInstallDirs)

find_package( Boost 1.67 COMPONENTS program_options REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

enable_testing()

add_subdirectory(perceptron)
