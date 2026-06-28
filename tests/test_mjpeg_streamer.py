from __future__ import annotations

import numpy as np

from streaming.encoder_profile import EncoderProfile
from streaming.mjpeg_streamer import MJPEGStreamer


def test_mjpeg_streamer_index_metadata_uses_encoder_resize_size():
    streamer = MJPEGStreamer(
        port=0,
        profile=EncoderProfile(
            codec="mjpeg",
            quality=80,
            target_fps=30,
            resize_width=640,
            resize_height=360,
        ),
    )
    try:
        packed_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        streamer.set_frame(packed_frame)

        assert streamer.sbs_width == 640
        assert streamer.sbs_height == 360
        index = streamer.index_bytes.decode("utf-8")
        assert "const WIDTH = 640;" in index
        assert "const HEIGHT = 360;" in index
    finally:
        streamer.stop()


def test_mjpeg_streamer_index_metadata_uses_packed_size_without_resize():
    streamer = MJPEGStreamer(
        port=0,
        profile=EncoderProfile(codec="mjpeg", quality=80, target_fps=30),
    )
    try:
        packed_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        streamer.set_frame(packed_frame)

        assert streamer.sbs_width == 1280
        assert streamer.sbs_height == 720
    finally:
        streamer.stop()
