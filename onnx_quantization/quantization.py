import torch
from pathlib import Path
import numpy as np
from torch.export import Dim
from onnxruntime.quantization import quantize_static, CalibrationMethod, CalibrationDataReader

from onnx_quantization.model import Digit_recognizer
from onnx_quantization.calibration_data import get_calibration_dataloader

base_dir = Path(__file__).resolve().parent.parent
model_path = base_dir / "trained_models"
onnx_path = base_dir/ "onnx_models"
quantized_path = base_dir/"quantized_models"

#Step 1 加载训练好的.pth模型
def load_trained_model(model_path, device='cpu'):
    model = Digit_recognizer() #定义模型骨架
    model.to(device)

    checkpoint = torch.load(model_path/'digit_recognizer.pth', map_location=device) #获取模型权重
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval() #必须开启为评估推理模式

    return model

#Step2 导出FP32.ONNX模型
def export_to_onnx(model, onnx_path, input_shape=(1,3,28,28), dynamic_batch=True):
    dummy_input = torch.randn(*input_shape)
    # dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} if dynamic_batch else None
    dynamic_shapes = {'x': {0: Dim('batch_size')}} if dynamic_batch else None

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        # dynamic_axes=dynamic_axes,
        dynamic_shapes=dynamic_shapes,
        opset_version=18,
        do_constant_folding=True,
        # dynamo=False #设置false，强制使用旧版导出器,用dynamic_axes替代dynamic_shapes
        dynamo=True #显式设置true，使用新版导出器，启用dynamic_shapes
    )
    print(f"✅ ONNX model exported to {onnx_path}")

#Step3 加载校准数据集【混合数据集策略】

#Step4 自定义CalibrationDataReader类
class CalibrationDataReader(CalibrationDataReader):
    def __init__(self, dataloader):
        self.iter = iter(dataloader)
    def get_next(self):
        try:
            batch = next(self.iter)
            if isinstance(batch, (tuple, list)): 
                #假如dataloader返回(image, label)，取image
                input_data = batch[0].numpy().astype(np.float32) 
            else:
                input_data = batch.numpy().astype(np.float32)
            return {'input': input_data}
        except StopIteration:
            return None

#Step5 执行静态量化
def static_quantize_onnx(onnx_path, quantized_path,calibration_dataloader):
    quantize_static(
        model_input=onnx_path,
        model_output=quantized_path,
        calibration_data_reader=CalibrationDataReader(calibration_dataloader),
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=['Conv', 'Gemm', 'Relu', 'Add']
    )
    print(f"✅ Quantized ONNX mode saved to {quantized_path}")


if __name__ == "__main__":
    #Step 1 加载训练好的.pth模型
    device='cpu'
    model = load_trained_model(model_path, device=device)

    #Step2 导出FP32.ONNX模型
    export_to_onnx(model, onnx_path=onnx_path/'digit_recognizer.onnx')

    #Step3 加载校准数据集【混合数据集策略】
    calib_loader = get_calibration_dataloader(batch_size=1, mnist_samples=40, emnist_samples=40, svhn_samples=52)
    
    #Step5 执行静态量化
    static_quantize_onnx(onnx_path=onnx_path/'digit_recognizer.onnx', quantized_path=quantized_path/'quantized_digit_recognizer.onnx', calibration_dataloader=calib_loader)

    print(f"✅ Export and quantization complete. FP32.onnx AND INT8.onnx are ready.")