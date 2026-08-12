cmake_minimum_required(VERSION 3.20)
project(winnex_xfactor VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Options
option(WINNEX_XFACTOR_BUILD_TESTS "Build tests" ON)
option(WINNEX_XFACTOR_BUILD_PYTHON "Build the pybind11 Python module" ON)

# --- Core static library ---------------------------------------------------
add_library(winnex_xfactor src/x_factor.cpp)
target_include_directories(winnex_xfactor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    target_compile_options(winnex_xfactor PRIVATE -O3 -funroll-loops)
endif()

# --- Python bindings (pybind11) --------------------------------------------
# scikit-build-core sets up Python3_ROOT_DIR / Python_EXECUTABLE in the build
# environment; pybind11's pybind11_add_module resolves the interpreter without
# us requiring Development/Development.Embed (which manylinux lacks).
if(WINNEX_XFACTOR_BUILD_PYTHON)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_winnex_xfactor bindings/py_module.cpp)
    target_link_libraries(_winnex_xfactor PRIVATE winnex_xfactor)
    if(DEFINED SKBUILD_PLATLIB_DIR)
        install(TARGETS _winnex_xfactor DESTINATION ${SKBUILD_PLATLIB_DIR})
    endif()
endif()

# --- Tests -----------------------------------------------------------------
if(WINNEX_XFACTOR_BUILD_TESTS)
    enable_testing()
    add_executable(test_x_factor tests/test_x_factor.cpp)
    target_link_libraries(test_x_factor PRIVATE winnex_xfactor)
    add_test(NAME x_factor COMMAND test_x_factor)
endif()
