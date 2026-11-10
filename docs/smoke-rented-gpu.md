# Smoke test on a rented GPU

Manual gate before any launch. Unit tests use a fake adapter; this checks the real `NvmlAdapter` and `TorchGemmLoad` on real hardware. Budget about one hour and a few dollars.

## 1. Rent a GPU

1. Pick a cheap marketplace GPU on Vast.ai or RunPod (roughly $0.30-2/hr). An RTX 3090/4090 or A-series card is fine; prefer a model listed in `specs.py`.
2. Choose a PyTorch template (CUDA-enabled image) with SSH access.
3. SSH in and confirm the GPU is idle: `nvidia-smi` should show no other processes.

## 2. Install

```
git clone https://github.com/narek-keshishyan/gpucheck.git gpucheck
cd gpucheck
pip install -e .
python -c "import torch; print(torch.cuda.is_available())"   # must print True
pip install torch                                            # only if the line above fails
```

Alternative without a clone: `pip install git+https://github.com/narek-keshishyan/gpucheck.git`, or `pip install "gpucheck-cli[load]"` once the PyPI release is out.

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
- Ctrl-C stops the run (load is cleaned up); no result.json is written for an interrupted run.
- Destroy the rented instance when finished so it stops billing.

## Multi-GPU machine

Multi-GPU runs have so far been tested only with simulated cards. Run this section once on real hardware before relying on `--gpus all`. Budget up to two hours for an 8-GPU machine at the standard length; use `--quick` for a first pass.

1. Rent a machine with at least two GPUs, ideally A100 or H100, and install gpucheck as in section 2.
2. Before the run, check the device order: `echo $CUDA_DEVICE_ORDER` should print nothing or `PCI_BUS_ID`. gpucheck sets it to `PCI_BUS_ID` only when it is unset. Any other value (for example `FASTEST_FIRST`) would make the card being loaded differ from the card being measured; if you see one, note it and stop.
3. Run `gpucheck --gpus all --include-serial --offline --lot-dir ./smoke-lot`. In a second terminal run `watch -n 1 nvidia-smi` and confirm:
   - exactly one card is loaded at a time, and it is the card gpucheck says it is testing;
   - no other card shows a gpucheck (python) process before its own turn; the tool must not appear on a card before that card's busy check;
   - after each card finishes, its memory use drops back (cached memory is released), and note whether the gpucheck process stays listed on earlier cards.
4. Confirm no card is refused as busy (check the lot summary table and `outcome` in `lot.json`).
5. Compare each `gpu<N>.json` manifest against `nvidia-smi -q` for serial, VBIOS version and PCIe link, and against `nvidia-smi -q -d ECC,ROW_REMAPPER` for the memory counters. Note that a PCIe link below maximum at idle can be normal power saving.
6. Record `timing.settle_s` for every card and each card's `metrics.idle_temp_c`; note whether cards 2..N read hotter at idle than card 1. These numbers decide the settle defaults.
7. Repeat steps 3 and 4 on a consumer card (or run `--gpus 0` on one) and confirm the manifest fields that card does not support are simply absent and the run completes.
8. Upload the lot to a local server, using the server and tunnel from section 4:

   ```
   gpucheck --upload-saved ./smoke-lot --server http://localhost:8000
   ```

   Open the lot page and each card page, and the card history page printed for each card. Confirm the lot page lists every position, including any card that was not tested, and that no serial appears unless you passed `--include-serial`.
9. Run the same `gpucheck --upload-saved ./smoke-lot --server http://localhost:8000` again and confirm nothing is duplicated (the same pages, no new runs) and the delete tokens in `smoke-lot/lot.json` are still there.
10. Save the lot directory under a dated folder for reference seeding.
