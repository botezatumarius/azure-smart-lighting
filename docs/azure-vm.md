# Deployment: Azure virtual machine (Part 2)

The VM runs `smart_lighting.vm`, a FastAPI service that receives frames, counts people with
YOLOv8, writes `counts.csv` and `resources.csv`, and serves the dashboard on port 80.

## Infrastructure

| Setting | Value |
|---|---|
| Resource group | `smart-lighting-rg` |
| Region | Austria East (an allowed region for the subscription, with free-tier capacity) |
| Image | Ubuntu Server 24.04 LTS, x64 Gen2 |
| Size | Standard_B2ats_v2 (2 vCPU, 1 GiB); Standard_B1s works the same way |
| Authentication | SSH public key, user `azureuser` |
| Network security group | inbound TCP 22 (SSH) and 80 (HTTP) only |
| OS disk | Standard SSD (LRS), deleted with the VM |

## Deploy

From the project folder on the local machine (PowerShell), with `<IP>` the VM's public IP:

```powershell
git archive --format=tar -o $env:TEMP\smart-lighting.tar HEAD
scp $env:TEMP\smart-lighting.tar azureuser@<IP>:~
ssh azureuser@<IP> "mkdir -p smart-lighting && tar -xf smart-lighting.tar -C smart-lighting && bash smart-lighting/deploy/setup.sh"
```

`deploy/setup.sh` is idempotent; run the same commands to deploy a new version. It:

1. installs `python3-opencv`, `python3-pip` and `python3-venv`;
2. adds a 2 GiB swap file (1 GiB of RAM is not enough to install PyTorch and load the model);
3. installs CPU-only PyTorch and the project in a virtual environment (`--no-cache-dir`);
4. creates the upload token in `/etc/smart-lighting.env` (readable by root only) and prints it;
5. installs and starts the `smart-lighting` systemd service (port 80, non-root user, restarts
   on failure, starts at boot).

Check the service:

```powershell
ssh azureuser@<IP> "systemctl status smart-lighting --no-pager; free -h"
```

If another `ssh` (e.g. from MSYS2) comes first on the PATH, use Windows' OpenSSH explicitly:
`Set-Alias ssh C:\Windows\System32\OpenSSH\ssh.exe` (and the same for `scp`).

## Use

```powershell
$env:SMART_LIGHTING_TOKEN = "<token>"
.venv\Scripts\python -m smart_lighting.sender --server http://<IP> --frames 60
```

The dashboard is at `http://<IP>/`. Results on the VM are in `~/smart-lighting/runs/vm/`;
copy them for analysis with `scp azureuser@<IP>:smart-lighting/runs/vm/*.csv runs/vm-azure/`.

## Cost

The VM size is covered by the Azure for Students free grant (750 h/month). The OS disk
(≈ $2.4/month) and the static public IP (≈ $3.6/month) are charged even while the VM is
stopped. Stop (deallocate) the VM from the portal when it is not in use, and delete the
resource group when the project is finished.
