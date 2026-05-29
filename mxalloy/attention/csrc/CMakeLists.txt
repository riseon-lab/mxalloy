# Build the fused quantized-KV attention extension (_ext.so + mxalloy_ext.metallib).
# SKELETON -- mirrors mlx/examples/extensions. Requires full Xcode for the Metal toolchain.

cmake_minimum_required(VERSION 3.25)
project(mxalloy_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MXALLOY_BUILD_METAL "Compile the Metal kernel into a metallib" ON)

# MLX ships its cmake package + the extension helper (provides mlx_build_metallib) under
# <site-packages>/mlx/share/cmake/MLX. mlx.extension.CMakeBuild puts that on CMAKE_PREFIX_PATH;
# for a manual cmake run, pass -DMLX_ROOT=$(python -c "import mlx;print(mlx.__path__[0])").
find_package(MLX CONFIG REQUIRED)
find_package(Python 3.11 COMPONENTS Interpreter Development.Module REQUIRED)

# nanobind from the active Python env.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Mirror MLX's own `nanobind_add_module(core ...)` config EXACTLY, or nanobind won't share
# the mlx::core::array type with us. The internals key is __nb_internals_<abi>_<domain>__:
# matching nanobind (MLX 0.31.2 pins v2.12.0) is necessary but NOT sufficient -- MLX builds
# with STABLE_ABI and NB_DOMAIN mlx, so we must too.
nanobind_add_module(_ext NB_STATIC STABLE_ABI NB_DOMAIN mlx quantized_sdpa.cpp bindings.cpp)
target_link_libraries(_ext PRIVATE mlx)
target_include_directories(_ext PRIVATE ${MLX_INCLUDE_DIRS})

# Compile the Metal source into a metallib named "mxalloy_ext", placed beside the module so
# Device::get_kernel(name, "mxalloy_ext") finds it at runtime.
if(MXALLOY_BUILD_METAL AND MLX_BUILD_METAL)
  mlx_build_metallib(
    TARGET mxalloy_ext_metallib
    TITLE mxalloy_ext
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/quantized_sdpa.metal
    INCLUDE_DIRS ${MLX_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  add_dependencies(_ext mxalloy_ext_metallib)
endif()
