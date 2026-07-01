import onnx
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort

from common.utils import get_image_files, load_test_img, extract_true_label

base_dir = Path(__file__).resolve().parent.parent
onnx_path = base_dir/ "onnx_models"
quantized_path = base_dir/ "quantized_models"
image_path = base_dir/"test_imgs"

#Step1 检查模型完整性
def check_onnx_integrity(model_path):
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    print("✅ ONNX model is valid.")

#Step2 加载量化模型并创建推理会话
def create_quant_session(quantized_path):
    sess_options = ort.SessionOptions() #创建会话配置
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL#开启所有图优化
    providers = [("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}),
                  "CPUExecutionProvider"]
    quant_session = ort.InferenceSession(quantized_path, sess_options, providers=providers)
    return quant_session

#Step3 批量加载测试图片【图片文件路径+对应的标签列表】，批量预测【预测正确率+预测时长】
def onnx_batch_predict(image_path, session, input_name, output_name, batch_size=64):
    image_files = get_image_files(image_path)
    if not image_files:
        print("No image files found")
        return
    image_files.sort()
    
    label_list = [extract_true_label(f.name) for f in image_files]

    total_time = 0.0
    total_correct = 0

    for i in range(0, len(image_files), batch_size):
        batch_paths = image_files[i:i+batch_size]
        batch_labels = np.array(label_list[i:i+batch_size]) #需统一为数组形式
        
        batch_inputs = load_test_img(batch_paths)
        
        start = time.time()
        outputs = session.run([output_name], {input_name: batch_inputs})
        predictions = np.argmax(outputs[0], axis=1)
        elapsed = time.time() - start

        total_time += elapsed
        total_correct += (predictions == batch_labels).sum() #此时均为数组形式
    
    batch_accuracy = total_correct / len(image_files) * 100
    average_time = total_time / len(image_files)

    print(f"[{image_path}], total images: {len(image_files)}")
    print(f"accuracy: {batch_accuracy:.2f}%")
    print(f"total prediction time: {total_time:.4f}s")
    print(f"average time per prediction: {average_time:.4f}s")

if __name__ == "__main__":
    #Step1 检查模型完整性
    model_path = quantized_path/"quantized_digit_recognizer.onnx"
    check_onnx_integrity(model_path)

    #Step2 加载量化模型并创建推理会话
    session = create_quant_session(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name   

    #Step3 批量加载测试图片【图片文件路径+对应的标签列表】，批量预测【预测正确率+预测时长】
    image_path = image_path/"batch_1"
    onnx_batch_predict(image_path, session, input_name, output_name, batch_size=64)
    


    