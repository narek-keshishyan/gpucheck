# Smoke test on a rented GPU

Manual gate before any launch. Unit tests use a fake adapter; this checks the real `NvmlAdapter` and `TorchGemmLoad` on real hardware. Budget about one hour and a few dollars.

## 1. Rent a GPU

1. Pick a cheap marketplace GPU on Vast.ai or RunPod (roughly $0.30-2/hr). An RTX 3090/4090 or A-series card is fine; prefer a model listed in `specs.py`.
2. Choose a PyTorch template (CUDA-enabled image) with SSH access.
3. SSH in and confirm the GPU is idle: `nvidia-smi` should show no other processes.

## 2. Install

```
git clone <repo-url> gpucheck   # placeholder: use the real repo URL
cd gpucheck
pip install -e .
python -c "import torch; print(torch.cuda.is_available())"   # must print True
pip install torch                                            # only if the line above fails
```

Alternative without a clone: `pip install git+<repo-url>` (placeholder URL).

## 3. Quick offline run

```
gpucheck --quick --offline
echo $?
```

Expect exit code 0, a printed report, and `./gpucheck-result.json`.

## 4. Full run with a local server

On your own machine, start the server:

```
cd gpucheck-server
DATABASE_URL=sqlite:///./smoke.db uv run uvicorn gpucheck_server.app:app --port 8000
```

Expose it to the rented box with a reverse SSH tunnel (or use ngrok and pass its https URL instead):

```
ssh -R 8000:localhost:8000 <user>@<rented-host> -p <port>
```

On the rented box:

```
gpucheck --server http://localhost:8000
```

Answer `y` to the upload prompt. Note the page URL and the delete token.

## 5. Verify

- Sample counts are about 60 baseline and 600 load (check `samples` in `gpucheck-result.json`).
- Throttle flags look plausible: none on a cool, well-fed card; thermal or power flags only if `nvidia-smi -q -d PERFORMANCE` agrees.
- Sustained TFLOPs is within 2x of the datasheet figure for that model.
- Open `<server>/r/<run_id>` on a phone (via the tunnel or ngrok URL): layout fits the screen, charts render, text is readable.
- The page says "insufficient reference (n=0)" (fresh DB) or a grade, and shows "graded against n=..." when graded.
- Delete works:
  `curl -i -X DELETE -H "X-Delete-Token: <token>" http://localhost:8000/v1/runs/<run_id>` returns 204, and the page then returns 404.

## 6. Record for reference seeding

Keep for each GPU tested: the exact model name as reported, driver version, power limit, and the `gpucheck-result.json` file (and a screenshot of the page). Save results under a dated folder so they can be used to seed the reference set.

## Safety notes

- gpucheck only reads from the GPU and runs a compute load; it changes no settings.
- It refuses to run if the GPU is already in use.
- Ctrl-C stops the run immediately and releases the GPU.
- Destroy the rented instance when finished so it stops billing.
