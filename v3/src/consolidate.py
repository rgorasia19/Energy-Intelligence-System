import os
import shutil
import time

def consolidate_models():
    base_dir = r"c:\Users\ronak\OneDrive\Desktop\Random\Energy-Intelligence-System\v3\src"
    target_dir = os.path.join(base_dir, "consolidated_models")
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    runs = []
    
    # Collect all models
    for b in ['mlruns', 'mlruns_1', 'mlruns_new']:
        src_dir = os.path.join(base_dir, b)
        if not os.path.exists(src_dir): continue
        
        for root, dirs, files in os.walk(src_dir):
            if 'MLmodel' in files:
                mlmodel_path = os.path.join(root, 'MLmodel')
                mtime = os.path.getmtime(mlmodel_path)
                dt_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(mtime))
                
                # The 'artifacts' directory is usually the root we are currently in
                # e.g., mlruns/0/models/m-xyz/artifacts
                # Let's verify the path ends with artifacts
                artifacts_dir = root
                
                # We need to extract the model ID, which is usually the parent directory of 'artifacts'
                # or just the directory name if it's not named artifacts
                parts = artifacts_dir.replace("\\", "/").split("/")
                if "models" in parts:
                    idx = parts.index("models")
                    if len(parts) > idx + 1:
                        model_id = parts[idx + 1]
                    else:
                        model_id = os.path.basename(artifacts_dir)
                else:
                    model_id = os.path.basename(artifacts_dir)
                    
                runs.append((mtime, dt_str, artifacts_dir, model_id))

    # Sort by time, newest first
    runs.sort(key=lambda x: x[0], reverse=True)
    
    print("Consolidating models...")
    for idx, (mtime, dt_str, artifacts_dir, model_id) in enumerate(runs):
        # Create a tidy name: YYYYMMDD_HHMMSS_model_id
        new_name = f"{dt_str}_{model_id}"
        dest_path = os.path.join(target_dir, new_name)
        
        if not os.path.exists(dest_path):
            print(f"Copying {model_id} to {new_name}")
            shutil.copytree(artifacts_dir, dest_path)
        else:
            print(f"Already exists: {new_name}")
            
    print("\nCleanup old MLflow folders...")
    for b in ['mlruns', 'mlruns_1', 'mlruns_new']:
        src_dir = os.path.join(base_dir, b)
        if os.path.exists(src_dir):
            shutil.rmtree(src_dir)
            print(f"Deleted {b}")
            
    db_path = os.path.join(base_dir, 'mlflow.db')
    if os.path.exists(db_path):
        os.remove(db_path)
        print("Deleted mlflow.db")

    print("\nDone! Best model is:", runs[0][1] + "_" + runs[0][3])

if __name__ == '__main__':
    consolidate_models()
