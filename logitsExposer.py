import onnx
from onnx import helper, shape_inference

# === PATHS ===
model_path = "GridFoodCollector.onnx"
new_model_path = "Food/Food_with_logitsLMAO.onnx"

# === Load ONNX model ===
model = onnx.load(model_path)

# === Find ArgMax node and its input ===
argmax_node = None
for node in model.graph.node:
    if node.op_type == "ArgMax":
        argmax_node = node
        break

if argmax_node is None:
    raise RuntimeError("ArgMax node not found.")

logits_tensor_name = argmax_node.input[0]

# === Try to find the shape of the logits tensor ===
logits_shape = None
for value_info in list(model.graph.value_info) + list(model.graph.output) + list(model.graph.input):
    if value_info.name == logits_tensor_name:
        logits_shape = [dim.dim_value if (dim.dim_value > 0) else "batch" for dim in value_info.type.tensor_type.shape.dim]
        break

# Fallback if shape not found
if logits_shape is None:
    print("⚠️ Could not infer logits tensor shape. Defaulting to ['batch', 2].")
    logits_shape = ["batch", 20]

# === Create new output for logits ===
logits_output = helper.make_tensor_value_info(
    name=logits_tensor_name,
    elem_type=onnx.TensorProto.FLOAT,
    shape=logits_shape
)

# === Add new output to model ===
model.graph.output.append(logits_output)

# === Save new model ===
onnx.save(model, new_model_path)
print(f"✅ Saved new model with logits to: {new_model_path}")
