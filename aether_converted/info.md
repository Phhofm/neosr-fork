### Dynamic Shapes Used

The ONNX files were exported with the following dynamic shape ranges, based on the default values in your conversion script:

  * **Batch Size**: 1 to 16, with an optimal batch size of 1.
  * **Height**: 32 to 512, with an optimal height of 256.
  * **Width**: 32 to 512, with an optimal width of 256.

This means the exported model is flexible and can accept any input size between 32x32 and 512x512, as long as the dimensions are a multiple of 32.

### TensorRT `trtexec` Commands

To build a TensorRT INT8 engine from the exported FP32 ONNX file, you have two options for optimal quality and speed: a dynamic engine and a static engine.

#### For Optimal Flexibility (Dynamic Engine)

This command creates a single engine that can handle any batch size from 1 to 16 and any image size from 32x32 up to 512x512, with the best performance for the optimal shape of 256x256.

```bash
trtexec --onnx="path/to/your/aether_small_2x_fp32_int8.onnx" \
        --saveEngine="aether_small_2x_int8_dynamic.plan" \
        --int8 \
        --minShapes=input:1x3x32x32 \
        --optShapes=input:1x3x256x256 \
        --maxShapes=input:16x3x512x512 \
        --workspace=4096 \
        --best
```

#### For Maximum Speed (Static Engine)

This command creates a static engine that is optimized for a fixed input size of 256x256, which is ideal if you know your input images will consistently be this size.

```bash
trtexec --onnx="path/to/your/aether_small_2x_fp32_int8.onnx" \
        --saveEngine="aether_small_2x_int8_static_256.plan" \
        --int8 \
        --inputShapes=input:1x3x256x256 \
        --workspace=4096 \
        --best
```
