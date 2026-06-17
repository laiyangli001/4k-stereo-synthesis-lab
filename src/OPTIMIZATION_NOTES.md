# Optimization Notes

## OpenXR/OpenGL color precision

- Current color upload path is SDR 8-bit RGB:
  - capture/process output: RGB uint8
  - color PBO: `width * height * 3`
  - OpenGL upload: `GL_RGB` + `GL_UNSIGNED_BYTE`
  - color texture: `dtype='f1'`
- Do not change only the PBO to 10-bit. That would not improve color if the source is still 8-bit, and may add conversion cost or format bugs.
- Future 10-bit/HDR optimization needs an end-to-end review:
  - confirm whether the Windows capture source can provide real 10-bit/HDR/FP16 frames
  - choose an internal format such as `RGB10_A2`, `RGBA16F`, or `R11F_G11F_B10F`
  - update OpenGL texture allocation and upload format/type together
  - verify shader color-space handling, especially sRGB versus linear
  - verify OpenXR swapchain/runtime/headset support for the chosen format
- More realistic first optimization: fix gamma/sRGB/linear handling and consider dithering before attempting full 10-bit/HDR.
