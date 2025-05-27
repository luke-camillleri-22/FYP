import numpy as np
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import onnxruntime as ort
import struct

# === CONFIG ===
BIN_PATH = "AgentObservations.bin"
ONNX_PATH = "Pyramids2_with_logits.onnx"
OBS_SIZE = 172
NUM_ACTIONS = 5
LOOKUP_PATH = "shap_lookup.bin"

# === TAG DEFINITIONS ===
TagLabels = ["Block", "Wall", "Goal", "SwitchOff", "SwitchOn", "Stone"]

# === LOAD OBSERVATIONS ===
def load_observations(path):
    data = np.fromfile(path, dtype=np.float32)
    remainder = data.size % OBS_SIZE
    if remainder != 0:
        print(f"Trimming {remainder} extra floats")
        data = data[:-remainder]
    reshaped = data.reshape(-1, OBS_SIZE)
    mask = np.isfinite(reshaped).all(axis=1)
    return reshaped[mask]

X = load_observations(BIN_PATH)
print(f"Loaded {X.shape[0]} valid observations")

# === SEMANTIC GROUPING ===
SemanticGroups = {f"{tag} Detected": [] for tag in TagLabels}
SemanticGroups["Misses"] = []
SemanticGroups["Proximity"] = []
SemanticGroups["SwitchState"] = [168]
SemanticGroups["Velocity"] = [169, 170, 171]

for sensor_i in range(3):
    for ray_i in range(7):  # 7 rays
        base = sensor_i * 56 + ray_i * 8
        for tag_i, tag in enumerate(TagLabels):
            SemanticGroups[f"{tag} Detected"].append(base + tag_i)
        SemanticGroups["Misses"].append(base + 6)
        SemanticGroups["Proximity"].append(base + 7)

group_names = list(SemanticGroups.keys())

# === LOAD MODEL ===
session = ort.InferenceSession(ONNX_PATH)
output_names = [o.name for o in session.get_outputs()]
dummy_input = {
    "obs_0": np.zeros((1, 56), dtype=np.float32),
    "obs_1": np.zeros((1, 56), dtype=np.float32),
    "obs_2": np.zeros((1, 56), dtype=np.float32),
    "obs_3": np.zeros((1, 4), dtype=np.float32),
    "action_masks": np.ones((1, NUM_ACTIONS), dtype=np.float32)
}
outputs = session.run(None, dummy_input)
logits_name = [n for n, o in zip(output_names, outputs) if o.shape == (1, NUM_ACTIONS)][0]
print("✅ Using logits output:", logits_name)

# === PREDICTION FUNCTION ===
def predict_fn(X_batch):
    return session.run([logits_name], {
        "obs_0": X_batch[:, 0:56],
        "obs_1": X_batch[:, 56:112],
        "obs_2": X_batch[:, 112:168],
        "obs_3": X_batch[:, 168:172],
        "action_masks": np.ones((X_batch.shape[0], NUM_ACTIONS), dtype=np.float32)
    })[0]

Y = predict_fn(X)
valid_rows = np.isfinite(Y).all(axis=1)
X_clean = X[valid_rows]
Y_clean = Y[valid_rows]
print(f"✅ Kept {X_clean.shape[0]} clean observations out of {X.shape[0]}")

# === SHAP & LOOKUP ===
all_group_scores = [[] for _ in range(X_clean.shape[0])]

for action_index in range(NUM_ACTIONS):
    print(f"🎯 Training surrogate for Action {action_index}")
    model = xgb.XGBRegressor(n_estimators=200, max_depth=15)
    model.fit(X_clean, Y_clean[:, action_index])

    explainer = shap.Explainer(model)
    shap_values = explainer(X_clean)

    # Bar plot
    mean_scores = []
    for name in group_names:
        indices = SemanticGroups[name]
        score = np.abs(shap_values.values[:, indices]).mean()
        mean_scores.append(score)

    plt.figure(figsize=(10, 6))
    plt.barh(group_names, mean_scores)
    plt.xlabel("Mean |SHAP Value|")
    plt.title(f"Semantic SHAP Importance - Action {action_index}")
    plt.tight_layout()
    plt.savefig(f"shap_semantic_action{action_index}.png", dpi=300)
    plt.close()



