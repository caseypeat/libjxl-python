
cmake_minimum_required(VERSION 3.22)

project(jxlbinding)

add_subdirectory(libjxl)
add_subdirectory(pybind11)

pybind11_add_module(jxlbinding jxlbinding.cc)

target_include_directories( jxlbinding PUBLIC libjxl )

target_link_libraries(jxlbinding PUBLIC jxl)
target_link_libraries(jxlbinding PUBLIC jxl_tool)