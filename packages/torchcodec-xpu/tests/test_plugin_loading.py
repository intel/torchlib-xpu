# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""torchcodec-xpu is built, installed and loaded by TorchCodec."""

import importlib.util
import os
import unittest
from importlib.metadata import entry_points


class TestPluginLoading(unittest.TestCase):
    def test_backend_entry_point_is_registered(self) -> None:
        # TorchCodec discovers out-of-tree device backends through this group.
        backends = entry_points(group="torchcodec.backends")
        self.assertIn(
            "torchcodec_xpu:load_torchcodec_xpu_shared_library",
            [backend.value for backend in backends],
        )

    def test_torchcodec_loads_xpu_plugin(self) -> None:
        # Importing torchcodec autoloads every registered backend and raises
        # RuntimeError if one fails to load, unless autoloading is disabled.
        self.assertNotEqual(
            os.getenv("TORCHCODEC_DEVICE_BACKEND_AUTOLOAD", "1"),
            "0",
            "backend autoloading is disabled, the plugin would not be loaded",
        )
        import torchcodec

        # The plugin library matching the FFmpeg major version in use is built.
        library = f"torchcodec_xpu.xpu_ops{torchcodec.ffmpeg_major_version}"
        spec = importlib.util.find_spec(library)
        self.assertIsNotNone(spec, f"{library} was not built or installed")

        # ...and was actually loaded into this process.
        with open("/proc/self/maps") as maps:
            self.assertIn(spec.origin, maps.read())


if __name__ == "__main__":
    unittest.main()
