import copy
import math
import pathlib
import dataclasses
import collections
import numpy as np
import torch
import os
from greyhounds.config import Settings
from greyhounds.ml import (
    TrainingConfig, build_race_examples, _eligible_examples, _split_examples,
    _evaluate_permutation_model, _load_artifact, Scaler, PermutationScoringANN,
    FIXED_RUNNER_COUNT, COMMON_FEATURE_NAMES, DOG_FEATURE_NAMES, predict_upcoming_races
)

def main():
    try:
        settings = Settings()
    except TypeError:
        from dotenv import load_dotenv
        load_dotenv()
        settings = Settings(
            database_url=os.getenv("DATABASE_URL", "sqlite:///greyhounds.db"),
            artifacts_dir=os.getenv("ARTIFACTS_DIR", "artifacts"),
            gbgb_base_url="https://www.gbgb.org.uk",
            rapidapi_base_url="https://dog-racing.p.rapidapi.com",
            rapidapi_host="dog-racing.p.rapidapi.com",
            rapidapi_key=os.getenv("RAPIDAPI_KEY", ""),
            http_user_agent="Mozilla/5.0",
            request_timeout_seconds=30,
            max_runners=6
        )

    db_scheme = settings.database_url.split(":")[0]
    print(f"Database scheme: {db_scheme}")

    artifact_paths = sorted(pathlib.Path("artifacts/models").glob("train-*.pt"))
    artifact_path = next((p for p in reversed(artifact_paths) if "recovery" not in p.name), None)
    if not artifact_path:
        print("No artifact found")
        return
    print(f"Using artifact: {artifact_path.name}")

    artifact = _load_artifact(artifact_path)
    config_dict = artifact["config"]
    if isinstance(config_dict, dict):
        config = TrainingConfig(**config_dict)
    else:
        config = config_dict
        
    common_feats = artifact.get("common_feature_names", COMMON_FEATURE_NAMES)
    dog_feats = artifact.get("dog_feature_names", DOG_FEATURE_NAMES)

    scaler = Scaler(
        common_feats, 
        dog_feats, 
        artifact["scaler_common_mean"], 
        artifact["scaler_common_std"], 
        artifact["scaler_dog_mean"], 
        artifact["scaler_dog_std"]
    )
    
    model = PermutationScoringANN(
        len(common_feats),
        len(dog_feats),
        hidden_dim=config.hidden_dim,
        dropout=config.dropout
    )
    # The keys suggest state_dict might be the one
    model.load_state_dict(artifact["state_dict"])
    model.eval()

    print("Building examples...")
    examples = build_race_examples(settings)
    eligible = _eligible_examples(examples, common_feats, dog_feats)
    train_ex, val_ex = _split_examples(eligible, config.validation_fraction)
    eval_ex = val_ex if val_ex else train_ex
    
    orig_count = len(eval_ex)
    if orig_count > 600:
        indices = np.linspace(0, orig_count - 1, 600, dtype=int)
        eval_ex = [eval_ex[i] for i in indices]
        print(f"Eval subset: 600 / {orig_count}")
    else:
        print(f"Eval count: {orig_count}")

    def run_eval(label, current_eval_ex):
        res = _evaluate_permutation_model(model, scaler, current_eval_ex, config)
        print(f"--- {label} ---")
        print(f"Winner Accuracy: {res['winner_accuracy']:.4f}")
        print(f"Actual counts: {res['actual_counts']}")
        print(f"Predicted counts: {res['predicted_counts']}")
        acc_by_trap = {k: round(v, 4) for k, v in res['winner_accuracy_by_actual_trap'].items()}
        print(f"Accuracy by trap: {acc_by_trap}")

    # Baseline
    run_eval("Baseline", eval_ex)

    dog_feat_list = list(dog_feats)
    def get_ablated_ex(to_zero):
        new_ex = []
        zero_indices = [dog_feat_list.index(f) for f in to_zero if f in dog_feat_list]
        for ex in eval_ex:
            ex_copy = copy.deepcopy(ex)
            for runner in ex_copy.runners:
                for idx in zero_indices:
                    runner.dog_features[idx] = 0.0
            new_ex.append(ex_copy)
        return new_ex

    run_eval("Ablate inside_box", get_ablated_ex(["inside_box"]))
    run_eval("Ablate trap_bias_score", get_ablated_ex(["trap_bias_score"]))
    run_eval("Ablate both", get_ablated_ex(["inside_box", "trap_bias_score"]))

    print("--- Upcoming ---")
    try:
        upcoming = predict_upcoming_races(settings, artifact_path=str(artifact_path))
        print(f"Total upcoming: {len(upcoming)}")
        traps = [r['predicted_winner_trap'] for r in upcoming]
        print(f"Top predicted traps: {dict(collections.Counter(traps))}")
    except Exception as e:
        print(f"Upcoming error: {e}")

if __name__ == "__main__":
    main()
