# Dev build for the fused quantized-KV attention extension. EXPERIMENTAL / FROZEN.
#
# Build, then copy _ext*.so + mxalloy_ext.metallib next to
# mxalloy/attention/quantized_sdpa.py to activate it (otherwise the pure-MLX fallback runs):
#     # one-time: install full Xcode, then
#     sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
#     pip install nanobind cmake
#     cd research/attention_kernel && python setup.py build_ext --inplace
#
# Packaging to a *single* PyPI binary wheel: the main project uses hatchling, which is not
# first-class for compiled extensions. Two clean options (pick one when this graduates):
#   (a) Switch the wheel build to scikit-build-core (cmake-native PEP 517 backend) -- the
#       modern way to ship cmake C++/Metal extensions; per-OS/arch wheels via cibuildwheel.
#   (b) Keep hatchling, build this ext separately, and force-include the prebuilt
#       _ext*.so + *.metallib as package data. Simpler, but you maintain the build yourself.
# Until then this stays a dev-only, build-it-yourself accelerator behind the pure-MLX fallback.

from setuptools import setup

from mlx import extension

if __name__ == "__main__":
    setup(
        name="mxalloy_ext",
        version="0.0.0",
        description="mxalloy fused quantized-KV attention (Metal extension)",
        ext_modules=[extension.CMakeExtension("mxalloy.attention._ext")],
        cmdclass={"build_ext": extension.CMakeBuild},
        packages=["mxalloy.attention"],
        package_dir={"mxalloy.attention": "."},
        package_data={"mxalloy.attention": ["*.so", "*.dylib", "*.metallib"]},
        zip_safe=False,
        python_requires=">=3.11",
    )
