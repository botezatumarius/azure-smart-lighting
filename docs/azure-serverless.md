# Deployment: serverless with Azure AI (Part 3)

```text
local: uploader.py ──JPEG──▶ Storage: raw/ ──Event Grid (raw/*.jpg only)──▶ Function ProcessFrame
                                                                              │  ▲
                                                            Azure AI Vision ◀─┘  │ people + boxes
                                                                              ▼
                        browser ◀── static website ◀── Storage: $web/ (counts.csv, latest.jpg, frames/)
```

All resources are in resource group `smart-lighting-rg`, region **France Central** (allowed for
the subscription and offering the AI Vision *people* feature).

## Resources

| Resource | Configuration |
|---|---|
| Storage account `smartlightingmb` | Standard, LRS; static website enabled (index `index.html`); containers `$web` (website) and `raw` (input, private) |
| Computer Vision `smart-lighting-vision-mb` | Standard S1 (F0 is limited to 20 images/min) |
| Function App `smart-lighting-func-mb` | Flex Consumption, Linux, Python 3.12, 2,048 MB, Application Insights on; internal storage and deployment storage use a user-assigned identity |
| Event Grid subscription `raw-frames` | on the storage account; *Blob Created* only; subject begins with `/blobServices/default/containers/raw/blobs/`, ends with `.jpg`; Web Hook endpoint (below) |

## Identities and roles

| Identity | Scope | Roles |
|---|---|---|
| Function App, system-assigned | `smartlightingmb` | Storage Blob Data Owner, Storage Queue Data Contributor |
| Function App, user-assigned (`<app>-uami`) | the app's internal storage account | Storage Blob Data Owner, Storage Queue Data Contributor, Storage Account Contributor |

The event-based blob trigger needs the queue role on **both** accounts; without it on the
internal account the host fails with `403 AuthorizationPermissionMismatch`. After changing
roles, stop and start the app.

## App settings

| Name | Value |
|---|---|
| `DataStorage__blobServiceUri` | `https://<account>.blob.core.windows.net` |
| `DataStorage__queueServiceUri` | `https://<account>.queue.core.windows.net` |
| `DataStorage__credential` | `managedidentity` |
| `VISION_ENDPOINT` | the Computer Vision endpoint |
| `VISION_KEY` | the Computer Vision key |

The function reaches the data account through its managed identity; no storage key is stored.

## Deploy

1. Deploy the `functions/` folder with VS Code (*Azure Functions: Deploy to Function App*);
   `functions/.vscode/settings.json` enables the remote build.
2. Create the Event Grid subscription with the webhook endpoint
   `https://<default domain>/runtime/webhooks/blobs?functionName=Host.Functions.ProcessFrame&code=<blobs_extension key>`
   (the key is under *Functions → App keys → System keys*). The `Microsoft.EventGrid` resource
   provider must be registered on the subscription.
3. Publish the dashboard and create `counts.csv`:

```powershell
$env:AZURE_STORAGE_CONNECTION_STRING = "<connection string>"
.venv\Scripts\python -m smart_lighting.uploader --publish-website
```

## Use

```powershell
.venv\Scripts\python -m smart_lighting.uploader --frames 240                 # one frame every 0.5 s
.venv\Scripts\python -m smart_lighting.uploader --frames 240 --only-changes  # skip unchanged frames
```

The website is at the storage account's static website endpoint.

## Troubleshooting

| Symptom | Cause |
|---|---|
| Function not listed after deployment | the module failed to import (check the logs) |
| Event Grid *Delivery Failed*, app returns 503 | missing roles on the internal storage account (host restart loop) |
| Event Grid *Delivery Failed*, HTTP 401 | the `blobs_extension` key in the endpoint changed; update the endpoint |

## Recursion safety

Three independent filters keep the function from triggering itself: the trigger path
`raw/{name}`, the Event Grid subject filter on `raw/`, and the function writing only to `$web`.

## Cost

Storage and Event Grid cost cents; Flex Consumption executions fall within the monthly free
grant; AI Vision costs ≈ $1 per 1,000 images after the first 5,000. Nothing is charged while
no images arrive.
