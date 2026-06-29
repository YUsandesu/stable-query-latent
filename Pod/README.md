# Pod Deployment Notes

This folder is a VM deployment bundle. Copy the files in this directory directly to
the root directory of the cloud virtual machine / pod before running the workflow.

## Files

- `setup.ipynb`: one-time or rerunnable environment setup.
- `run.ipynb`: main pipeline runner. Run this after `setup.ipynb`.
- `realtime_reader.ipynb`: log reader for checking progress while `run.ipynb` is
  running.
- `start_old.ipynb`: older startup notebook kept for reference.
- `../tools/sync_runpod_artifacts.ps1`: Windows helper that downloads only the
  expensive RunPod artifacts from the RunPod S3 bucket.

## Workflow

1. Copy the contents of this `Pod/` directory to the VM root directory.
2. Open and run `setup.ipynb`.
3. Open and run `run.ipynb`.
4. When progress needs to be checked, open `realtime_reader.ipynb`.

`realtime_reader.ipynb` has two important cells:

- Cell 1 loads historical log output and manifest summaries.
- Cell 2 follows new log output in realtime. It starts after Cell 1's log offset,
  so it does not repeat the historical output already shown by Cell 1.

Interrupting Cell 2 only stops the log viewer. It does not stop the running
pipeline.

## Download RunPod Artifacts

Use the selective sync helper instead of syncing the whole bucket:

```powershell
.\tools\sync_runpod_artifacts.ps1
```

The helper downloads only the high-cost corpus and experiment outputs generated
by `run.ipynb`: `text_h5.h5`, `embedding_h5.h5`, their manifests,
`VICReg_review/heads`, `stable_query_latent_artifacts`, and RunPod logs. Preview
first with:

```powershell
.\tools\sync_runpod_artifacts.ps1 -DryRun
```

To print the generated `aws s3 sync` command without contacting S3:

```powershell
.\tools\sync_runpod_artifacts.ps1 -PrintOnly
```
