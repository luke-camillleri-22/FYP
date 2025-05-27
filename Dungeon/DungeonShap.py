# File: DungeonShap_MultiAgent.py
import numpy as np
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import onnxruntime as ort

# === CONFIG ===
BIN_PATHS = [
    "AgentObservations_Agent0.bin",
    "AgentObservations_Agent1.bin",
    "AgentObservations_Agent2.bin"
]
ONNX_PATH = "Dungeon_with_logits.onnx"
OBS_SIZE = 410
NUM_ACTIONS = 7
RAY_FEATURES = 8
RAYS_PER_SENSOR = 17
NUM_SENSORS = 3
RAY_TOTAL = RAYS_PER_SENSOR * NUM_SENSORS  # 51 rays

# === TAG DEFINITIONS ===
TagLabels = ["Wall", "Agent", "Dragon", "Key", "Lock", "Portal"]
SemanticGroups = {f"{tag} Detected": [] for tag in TagLabels}
SemanticGroups["Misses"] = []
SemanticGroups["Proximity"] = []
SemanticGroups["HasKey"] = [408]
SemanticGroups["DragonDead"] = [409]

for ray_i in range(RAY_TOTAL):
    base = ray_i * RAY_FEATURES
    for tag_i, tag in enumerate(TagLabels):
        SemanticGroups[f"{tag} Detected"].append(base + tag_i)
    SemanticGroups["Misses"].append(base + 6)
    SemanticGroups["Proximity"].append(base + 7)

# === LOAD MODEL ===
session = ort.InferenceSession(ONNX_PATH)
input_names = [i.name for i in session.get_inputs()]
output_names = [o.name for o in session.get_outputs()]

dummy_input = {
    "obs_0": np.zeros((1, 408), dtype=np.float32),
    "obs_1": np.zeros((1, 2), dtype=np.float32),
    "action_masks": np.ones((1, NUM_ACTIONS), dtype=np.float32)
}
outputs = session.run(None, dummy_input)
logits_name = [name for name, out in zip(output_names, outputs) if out.shape == (1, NUM_ACTIONS)][0]
print("✅ Using logits output:", logits_name)

# === PREDICT FUNCTION ===
def predict_fn(X):
    return session.run([logits_name], {
        "obs_0": X[:, :408],
        "obs_1": X[:, 408:],
        "action_masks": np.ones((X.shape[0], NUM_ACTIONS), dtype=np.float32)
    })[0]




# === AGENT LOOP ===
agent_group_scores = []

for path in BIN_PATHS:
    data = np.fromfile(path, dtype=np.float32)
    remainder = data.size % OBS_SIZE
    if remainder != 0:
        print(f"Trimming {remainder} extra floats from {path}")
        data = data[:-remainder]
    X = data.reshape(-1, OBS_SIZE)
    X = X[np.isfinite(X).all(axis=1)]

    print(f"Loaded {X.shape[0]} valid observations from {path}")
    Y = predict_fn(X)
    Y = Y[np.isfinite(Y).all(axis=1)]
    X = X[:len(Y)]  # Match length

    agent_scores = []
    for action_index in range(NUM_ACTIONS):
        model = xgb.XGBRegressor(n_estimators=200, max_depth=15)
        model.fit(X, Y[:, action_index])
        explainer = shap.Explainer(model)
        shap_values = explainer(X)

        # Semantic grouping
        group_names = []
        group_scores = []
        for name, indices in SemanticGroups.items():
            score = np.abs(shap_values.values[:, indices]).mean()
            group_names.append(name)
            group_scores.append(score)

        agent_scores.append(group_scores)
    agent_group_scores.append(agent_scores)

# === COMPARISON PLOTS ===
group_names = list(SemanticGroups.keys())

for action_index in range(NUM_ACTIONS):
    plt.figure(figsize=(10, 6))
    bar_width = 0.25
    indices = np.arange(len(group_names))

    for i, agent_scores in enumerate(agent_group_scores):
        values = agent_scores[action_index]
        plt.barh(indices + i * bar_width, values, height=bar_width, label=f"Agent {i}")

    plt.yticks(indices + bar_width, group_names)
    plt.xlabel("Mean |SHAP Value|")
    plt.title(f"Semantic SHAP Comparison - Action {action_index}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"shap_compare_action{action_index}.png", dpi=300)
    plt.close()

print("✅ Comparison SHAP plots saved.")
