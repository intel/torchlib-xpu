# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025 Dmitry Rogozhkin.

import os
import subprocess
import sys
from pathlib import Path

import torch
import torchcodec
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

_ROOT_DIR = Path(__file__).parent.resolve()


class CMakeBuild(build_ext):

    def __init__(self, *args, **kwargs):
        self._install_prefix = None
        super().__init__(*args, **kwargs)

    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake is not available.") from None
        super().run()

    def build_extension(self, ext):
        """Call our CMake build system to build libtorchcodec*.so"""
        # Setuptools was designed to build one extension (.so file) at a time,
        # calling this method for each Extension object. We're using a
        # CMake-based build where all our extensions are built together at once.
        # If we were to declare one Extension object per .so file as in a
        # standard setup, a) we'd have to keep the Extensions names in sync with
        # the CMake targets, and b) we would be calling into CMake for every
        # single extension: that's overkill and inefficient, since CMake builds
        # all the extensions at once. To avoid all that we create a *single*
        # fake Extension which triggers the CMake build only once.
        assert ext.name == "FAKE_NAME", f"Unexpected extension name: {ext.name}"
        # The price to pay for our non-standard setup is that we have to tell
        # setuptools *where* those extensions are expected to be within the
        # source tree (for sdists or editable installs) or within the wheel.
        # Normally, setuptools relies on the extension's name to figure that
        # out, e.g. an extension named `torchcodec.libtorchcodec.so` would be
        # placed in `torchcodec/` and importable from `torchcodec.`. From that,
        # setuptools knows how to move the extensions from their temp build
        # directories back into the proper dir.
        # Our fake extension's name is just a placeholder, so we have to handle
        # that relocation logic ourselves.
        # _install_prefix is the temp directory where the built extension(s)
        # will be "installed" by CMake. Once they're copied to install_prefix,
        # the built .so files still need to be copied back into:
        # - the source tree (for editable installs) - this is handled in
        #   copy_extensions_to_source()
        # - the (temp) wheel directory (when building a wheel). I cannot tell
        #   exactly *where* this is handled, but for this to work we must
        #   prepend the "/torchcodec_xpu" folder to _install_prefix: this tells
        #   setuptools to eventually move those .so files into `torchcodec_xpu/`.
        # It may seem overkill to 'cmake install' the extensions in a temp
        # directory and move them back to another dir, but this is what
        # setuptools would do and expect even in a standard build setup.
        self._install_prefix = (
            Path(self.get_ext_fullpath(ext.name)).parent.absolute() / "torchcodec_xpu"
        )
        self._build_all_extensions_with_cmake()

    def _build_all_extensions_with_cmake(self):
        # Note that self.debug is True when you invoke setup.py like this:
        # python setup.py build_ext --debug install
        torch_dir = Path(torch.utils.cmake_prefix_path) / "Torch"
        torchcodec_dir = Path(torchcodec.cmake_prefix_path) / "TorchCodec"
        cmake_build_type = os.environ.get("CMAKE_BUILD_TYPE", "Release")
        torchcodec_xpu_disable_compile_warning_as_error = os.environ.get(
            "TORCHCODEC_XPU_DISABLE_COMPILE_WARNING_AS_ERROR", "OFF"
        )
        python_version = sys.version_info
        cmake_args = [
            f"-DCMAKE_INSTALL_PREFIX={self._install_prefix}",
            f"-DTorch_DIR={torch_dir}",
            f"-DTorchCodec_DIR={torchcodec_dir}",
            "-DCMAKE_VERBOSE_MAKEFILE=ON",
            f"-DCMAKE_BUILD_TYPE={cmake_build_type}",
            f"-DPYTHON_VERSION={python_version.major}.{python_version.minor}",
            f"-DTORCHCODEC_XPU_DISABLE_COMPILE_WARNING_AS_ERROR={torchcodec_xpu_disable_compile_warning_as_error}",
        ]

        self.build_temp = os.getenv("TORCHCODEC_XPU_CMAKE_BUILD_DIR", self.build_temp)
        print(f"Using {self.build_temp = }", flush=True)
        Path(self.build_temp).mkdir(parents=True, exist_ok=True)

        print("Calling cmake (configure)", flush=True)
        subprocess.check_call(
            ["cmake", str(_ROOT_DIR)] + cmake_args, cwd=self.build_temp
        )
        print("Calling cmake --build", flush=True)
        subprocess.check_call(["cmake", "--build", "."], cwd=self.build_temp)
        print("Calling cmake --install", flush=True)
        subprocess.check_call(["cmake", "--install", "."], cwd=self.build_temp)

    def copy_extensions_to_source(self):
        """Copy built extensions from temporary folder back into source tree.

        This is called by setuptools at the end of .run() during editable installs.
        """
        self.get_finalized_command("build_py")
        extensions = []
        if sys.platform == "linux":
            extensions = ["so"]
        elif sys.platform == "darwin":
            # Mac has BOTH .dylib and .so as library extensions. Short version
            # is that a .dylib is a shared library that can be both dynamically
            # loaded and depended on by other libraries; a .so can only be a
            # dynamically loaded module. For more, see:
            #   https://stackoverflow.com/a/2339910
            extensions = ["dylib", "so"]
        elif sys.platform in ("win32", "cygwin"):
            extensions = ["dll"]
        else:
            raise NotImplementedError(f"Platform {sys.platform} is not supported")

        for ext in extensions:
            for lib_file in self._install_prefix.glob(f"*.{ext}"):
                assert "libtorchcodec" in lib_file.name
                destination = Path("src/torchcodec_xpu/") / lib_file.name
                print(f"Copying {lib_file} to {destination}")
                self.copy_file(lib_file, destination, level=self.verbose)


# See `CMakeBuild.build_extension()`.
fake_extension = Extension(name="FAKE_NAME", sources=[])


def _write_version_files():
    if version := os.getenv("BUILD_VERSION"):
        # BUILD_VERSION is set by the `test-infra` build jobs. It typically is
        # the content of `version.txt` plus some suffix like "+cpu" or "+cu112".
        # See
        # https://github.com/pytorch/test-infra/blob/61e6da7a6557152eb9879e461a26ad667c15f0fd/tools/pkg-helpers/pytorch_pkg_helpers/version.py#L113
        version = version.replace("+cpu", "")
        with open(_ROOT_DIR / "version.txt", "w") as f:
            f.write(f"{version}")
    else:
        with open(_ROOT_DIR / "version.txt") as f:
            version = f.readline().strip()
        try:
            version = version.replace("+cpu", "")
            sha = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=str(_ROOT_DIR)
                )
                .decode("ascii")
                .strip()
            )
            version += "+" + sha[:7]
        except Exception:
            print("INFO: Didn't find sha. Is this a git repo?")

    with open(_ROOT_DIR / "src/torchcodec_xpu/version.py", "w") as f:
        f.write("# Note that this file is generated during install.\n")
        f.write(f"__version__ = '{version}'\n")


_write_version_files()

setup(
    ext_modules=[fake_extension],
    cmdclass={"build_ext": CMakeBuild},
)
