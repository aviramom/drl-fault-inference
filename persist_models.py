# persist_models.py
import os
import json
from pathlib import Path
import joblib
from datetime import datetime

def save_models_by_fault(models_by_fault, domain_name, ml_model_name, save_dir="trained_models"):
    """
    models_by_fault: dict[fault_mode_str][action] -> FaultyTransitionModel (sklearn inside)
    """
    root = Path(save_dir) / domain_name / ml_model_name
    root.mkdir(parents=True, exist_ok=True)

    # optional metadata for reproducibility
    meta = {
        "domain_name": domain_name,
        "ml_model_name": ml_model_name,
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "fault_modes": list(models_by_fault.keys()),
        "schema": "models_by_fault[fault_mode_str][action] -> sklearn model wrapped by FaultyTransitionModel"
    }
    (root / "metadata.json").write_text(json.dumps(meta, indent=2))

    # save each FaultyTransitionModel separately (compressed)
    for fm, action_map in models_by_fault.items():
        fm_dir = root / fm.replace(" ", "")
        fm_dir.mkdir(exist_ok=True)
        for action, model in action_map.items():
            # You can save either the whole wrapper or just the inner sklearn model.
            # Whole wrapper (recommended if it's picklable):
            joblib.dump(model, fm_dir / f"model_action_{action}.joblib", compress=3)

    print(f"✅ Saved models under: {root}")

def load_models_by_fault(domain_name, ml_model_name, save_dir="trained_models"):
    root = Path(save_dir) / domain_name / ml_model_name
    if not root.exists():
        raise FileNotFoundError(f"No saved models at {root}")

    models_by_fault = {}
    for fm_dir in root.iterdir():
        if not fm_dir.is_dir():
            continue
        fm = fm_dir.name
        inner = {}
        for file in fm_dir.glob("model_action_*.joblib"):
            action = int(file.stem.split("_")[-1])
            model = joblib.load(file)
            inner[action] = model
        if inner:
            models_by_fault[fm] = inner

    return models_by_fault
