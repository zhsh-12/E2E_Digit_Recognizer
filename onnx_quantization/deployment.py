import onnx
import time
from pathlib import Path
from PIL import Image
import numpy as np
import onnxruntime as ort

from common.preprocess import transform

base_dir = Path(__file__).resolve().parent.parent
onnx_path = base_dir/ "onnx_models"
quantized_path = base_dir/ "quantized_models"
image_path = base_dir/"test_imgs"

#Step1 检查模型完整性
def check_onnx_integrity(model_path):
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    print("✅ ONNX model is valid.")

#Step2 加载测试图片
def load_test_img(image_path):
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor.numpy().astype(np.float32)

#Step3 加载量化模型并创建推理会话
def create_quant_session(quantized_path):
    sess_options = ort.SessionOptions() #创建会话配置
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL#开启所有图优化
    providers = [("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}),
                  "CPUExecutionProvider"]
    quant_session = ort.InferenceSession(quantized_path, sess_options, providers=providers)
    return quant_session

if __name__ == "__main__":
    #Step1 检查模型完整性
    model_path = quantized_path/"quantized_digit_recognizer.onnx"
    check_onnx_integrity(model_path)

    #Step2 加载测试图片
    image_path = image_path/"batch_1"/"img_44[9].png"
    input_data = load_test_img(image_path)

    #Step3 加载量化模型并创建推理会话
    session = create_quant_session(model_path)

    #Step4 验证模型的输入输出数据类型
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(session.get_inputs()[0].type)
    print(session.get_outputs()[0].type)

    #Step5 执行推理，获取预测结果\单次预测时长
    start = time.time()
    result = session.run([output_name], {input_name: input_data})
    prediction = np.argmax(result[0])
    elapsed = time.time() - start

    print(f"Quantized model predicted: {prediction}")
    print(f"Single run costs: {elapsed:.4f}s")

    