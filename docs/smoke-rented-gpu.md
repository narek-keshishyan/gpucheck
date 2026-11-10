# Smoke test on a rented GPU

Manual gate before any launch. Unit tests use a fake adapter; this checks the real `NvmlAdapter` and `TorchGemmLoad` on real hardware. Budget about one hour and a few dollars.

Re-run the single-card sections (1-6) with every new version, not only the multi-GPU section: the manifest queries (VBIOS, PCIe link, lifetime memory-error counts) now run on every plain run, so a plain `gpucheck` on one card exercises new NVML calls too.

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
- Ctrl-C stops a single-card run (load is cleaned up); no result.json is written for an interrupted single-card run. A multi-GPU run keeps the cards that finished (see below).
- Destroy the rented instance when finished so it stops billing.

## Multi-GPU machine

Multi-GPU runs have so far been tested only with simulated cards. Run this section once on real hardware before relying on `--gpus all`. Budget up to two hours for an 8-GPU machine at the standard length; use `--quick` for a first pass.

1. Rent a machine with at least two GPUs (four or more for step 6), ideally A100 or H100, and install gpucheck as in section 2. Record the PyTorch and CUDA versions:

   ```
   python -c "import torch; print(torch.__version__, torch.version.cuda)"
   ```

2. Check the GPU numbering environment. `echo $CUDA_VISIBLE_DEVICES` must print nothing; if it prints a value, `unset CUDA_VISIBLE_DEVICES`. Confirm the tool refuses while it is set: `CUDA_VISIBLE_DEVICES=1 gpucheck --gpus all --offline` must exit 1 with "CUDA_VISIBLE_DEVICES is set, so PyTorch and NVML would number the GPUs differently". Also run `CUDA_DEVICE_ORDER=FASTEST_FIRST gpucheck --quick --offline --gpus 1` and confirm the "overriding CUDA_DEVICE_ORDER=FASTEST_FIRST with PCI_BUS_ID" line is printed.
3. Run the lot inside `tmux` (or `screen`) so a dropped SSH session does not stop it: `gpucheck --gpus all --include-serial --offline --lot-dir ./smoke-lot`. In a second terminal run `watch -n 1 nvidia-smi` and confirm:
   - exactly one card is loaded at a time, and it is the card gpucheck says it is testing;
   - no other card shows a gpucheck (python) process before its own turn; the tool must not appear on a card before that card's busy check;
   - after each card finishes, its memory use falls to a few hundred MB, not to zero, and the gpucheck process stays listed on that card for the rest of the lot. That is the CUDA context of the still-running process, not a leak.
4. Confirm the right card was loaded from the files, since watching `nvidia-smi` for two hours is not reliable. In each `gpu<N>.json`, the load-phase samples' `power_w` should sit near the card's power limit, and `metrics.steady_temp_c` should be clearly above `metrics.idle_temp_c`. A card whose load-phase power stays near idle was not the one loaded.
5. Confirm no card is refused as busy (check the lot summary table and `outcome` in `lot.json`), and that `lot.json` has `"finished": true`.
6. Subset runs. Run `gpucheck --quick --offline --gpus 1` and, on four or more cards, `gpucheck --quick --offline --gpus 1,3 --lot-dir ./smoke-subset`. With `watch -n 1 nvidia-smi` open, confirm GPU 0 shows no gpucheck process at any point, and that only the selected cards are loaded (check `power_w` in their files as in step 4).
7. Busy card. Hold a small CUDA process on one card, for example `python -c "import torch, time; x = torch.ones(1, device='cuda:1'); time.sleep(3600)"`, then run `gpucheck --quick --offline --gpus all --lot-dir ./smoke-busy`. Confirm that card is recorded as not tested with a "GPU busy" reason while the others run. Inside a container the process list can be hidden from NVML; note what the tool says for the busy card there (it may not see the other process at all).
8. Ctrl-C mid-lot. Start `gpucheck --quick --offline --gpus all --lot-dir ./smoke-ctrlc` and press Ctrl-C during the second card. Confirm exit code 130 (`echo $?`), `gpu0.json` and `lot.json` present, `"finished": true`, and the remaining cards recorded as interrupted.
9. Killed run. Start `gpucheck --quick --offline --gpus all --lot-dir ./smoke-killed`, and during the second card run `kill -9 <pid>` from another terminal. Confirm the finished cards' `gpu<N>.json` files are present, `lot.json` has `"finished": false` with the rest "not reached", and no temporary files are left. Then `gpucheck --upload-saved ./smoke-killed --server http://localhost:8000` (server and tunnel from section 4) must upload, printing "note: this lot did not finish".
10. Compare each `gpu<N>.json` manifest against `nvidia-smi -q` for serial, VBIOS version and PCIe link, and against `nvidia-smi -q -d ECC,ROW_REMAPPER` for the memory counters. Note that a PCIe link below maximum at idle can be normal power saving.
11. Record `timing.settle_s` for every card and each card's `metrics.idle_temp_c`; note whether cards 2..N read hotter at idle than card 1. These numbers decide the settle defaults.
12. Repeat steps 3 and 5 on a consumer card (or run `--gpus 0` on one) and confirm the manifest fields that card does not support are simply absent and the run completes.
13. Upload the lot to a local server, using the server and tunnel from section 4:

   ```
   gpucheck --upload-saved ./smoke-lot --server http://localhost:8000
   ```

   Open the lot page and each card page, and the card history page printed for each card. Confirm the lot page lists every position, including any card that was not tested, and that the serials recorded in `smoke-lot/lot.json` appear on the card pages (that run used `--include-serial`).
14. Run a short lot without the flag, `gpucheck --quick --offline --gpus 0,1 --lot-dir ./smoke-noserial`, and confirm no serial is in its files: no `gpu<N>.json` contains a `serial` key (`grep -l serial smoke-noserial/gpu*.json` prints nothing), and every card's `serial` in `smoke-noserial/lot.json` is `null`.
15. Run the same `gpucheck --upload-saved ./smoke-lot --server http://localhost:8000` again and confirm nothing is duplicated (the same pages, no new runs) and the delete tokens in `smoke-lot/lot.json` are still there.
16. Save the lot directories under a dated folder for reference seeding.
