---
trigger: always_on
---

## Helios Slurm quick rules (for agents)

- I prefer sbatch jobs than interactive mode.
- I prefer job logs to be in `logs/` directory (use `#SBATCH --output=logs/%x-%j.out` and `#SBATCH --error=logs/%x-%j.err`).

* **Never run heavy work on `login01`**. Submit via `sbatch` (or interactive `srun --pty`). Helios uses Slurm. ([docs.cyfronet.pl][1])
* **Always use a login shell in batch scripts** (important on GH200):
  `#!/bin/bash -l` ([docs.cyfronet.pl][1])
* **Pick the right partition + time limit**:

  * CPU: `--partition=plgrid` (72h) or `plgrid-long` (168h) ([docs.cyfronet.pl][1])
  * Big memory: `--partition=plgrid-bigmem` (72h) ([docs.cyfronet.pl][1])
  * GPU GH200: `--partition=plgrid-gpu-gh200` (48h) ([docs.cyfronet.pl][1])
* **Always set `--account/-A` to the full “grant+suffix”** (plain `-A grantname` won’t work). ([docs.cyfronet.pl][1])
  Use the env you gave:

  * CPU: `-A "$PLG_CPU_GRANT"` (e.g., `plgcredibleai2025-cpu`)
  * GPU: `-A "$PLG_GPU_GRANT"` (e.g., `plgcredibleai2025-gpu-gh200`)
* **Use the correct filesystem**:

  * Work dir / scratch: `$SCRATCH` (purged for old data; don’t treat as durable storage). ([docs.cyfronet.pl][1])
  * Durable project/group data: `$PLG_GROUPS_STORAGE/$HELIOS_GROUP` (your `HELIOS_GROUP=plggmi2ai`). ([docs.cyfronet.pl][1])
* **Architecture gotcha**: GPU partition is **GH200 (aarch64/ARM + Hopper GPU)**; don’t assume x86 builds/modules will run there. ([docs.cyfronet.pl][1])
* **GPU ML jobs**: load `ML-bundle` *inside the job* before venv/pip. ([docs.cyfronet.pl][1])

## Minimal templates (copy/paste)

### CPU job (`plgrid`)

```bash
#!/bin/bash -l
#SBATCH -J myjob
#SBATCH -p plgrid
#SBATCH -A ${PLG_CPU_GRANT}
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

cd "$SCRATCH"
# run your command here
```

### GPU GH200 job (`plgrid-gpu-gh200`)

```bash
#!/bin/bash -l
#SBATCH -J mygpujob
#SBATCH -p plgrid-gpu-gh200
#SBATCH -A ${PLG_GPU_GRANT}
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

ml ML-bundle/24.06a   # do this before venv/pip
cd "$SCRATCH"
# run your GPU command here
```

### Interactive debug (short)

```bash
srun -p plgrid -A "$PLG_CPU_GRANT" -t 00:30:00 --pty bash -l
# or GPU:
srun -p plgrid-gpu-gh200 -A "$PLG_GPU_GRANT" --gres=gpu:1 -t 00:30:00 --pty bash -l
```


