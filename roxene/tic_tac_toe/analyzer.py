import pickle
import mlflow
from mlflow.tracking import MlflowClient

from trial import Trial as Run

client = MlflowClient()
last_run = client.search_runs('0', max_results=1)[0]
print(last_run)
local_path = mlflow.artifacts.download_artifacts(run_id=last_run.info.run_id, artifact_path='runner.pkl')
with open(local_path, 'rb') as f:
    runner: Run = pickle.load(f)

print(len(runner.completed_trials))
