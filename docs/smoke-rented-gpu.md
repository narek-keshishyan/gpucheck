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
- Load-phase telemetry matches an independent reading. During the load phase run `nvidia-smi --query-gpu=power.draw,temperature.gpu --format=csv -l 1` in a second terminal: the `power_w` and `temp_c` of the load samples in `gpucheck-result.json` should agree with it (on an A100, near the power limit). Samples that read idle power under load mean the telemetry is being taken in the pause between load bursts.
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

This section was first run on real hardware on 2026-11-12 (four RTX 3090s). Run it again on real hardware before relying on `--gpus all` after a change to the run loop, the load or the lot code. Budget up to two hours for an 8-GPU machine at the standard length; use `--quick` for a first pass.

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
10. Compare each `gpu<N>.json` manifest against `nvidia-smi -q` for serial, VBIOS version and PCIe link, and against `nvidia-smi -q -d ECC,ROW_REMAPPER` for the memory counters. The current PCIe link in the file is read under load (`pcie_link_under_load` is true), so compare it with `nvidia-smi` while a run is loading the card, not at idle, where the link drops to a lower speed.
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
17. One card as a lot. On any machine run `gpucheck --quick --offline --gpus 0 --label one-card --lot-dir ./smoke-one` and confirm a lot directory with `gpu0.json` and a `lot.json` of size 1 is written, not a plain `gpucheck-result.json`.
18. Progress in a log. Run a lot with its output redirected (`... > lot.log 2>&1 &`) and confirm `lot.log` gains a `Testing GPU N (card i of n)...` line as each card starts and a `GPU N: tested` line as it ends, not only at the end of the lot.

## Extra checks

Not yet run on real hardware. This section is the gate for `--driver-errors`, `--answer-check`, `--memtest`, `--links`, `--burn-in`, `--together` and `--thorough`: none of them should be described as working until it has been run. They were developed on a machine with no NVIDIA GPU and no PyTorch, against simulated cards and a small stand-in for PyTorch, so the first real run is also the first time real PyTorch and a real driver see this code. Earlier real runs found bugs that tests with fakes had missed; expect some here.

Nobody rents a machine without the owner's go-ahead. The proposal, its cost and what each step is for:

### What needs which hardware

| Check | Hardware it needs | What only real hardware can show |
| --- | --- | --- |
| `--driver-errors` | Any NVIDIA card. Run it once inside a rented container (Vast.ai instances are containers) and, if possible, once on a bare-metal or VM host. | Whether the driver lets an unprivileged process in a container register for error events at all. If it refuses, the result must say "could not be started on this system" and the run must be otherwise unchanged. |
| `--answer-check` | Any card. One consumer card (RTX 3090 or 4090) and one datacenter card (A100 or H100). | That real PyTorch accepts every call, that all four number formats multiply, and above all that a working card gets zero wrong products. A single wrong product on a card that passes everything else means the check is wrong, not the card, until shown otherwise. |
| `--memtest` | Any card. One with ECC (A100, H100) and one without (RTX 3090 or 4090). | How long it takes, how much memory it really reaches, that the memory is released afterwards, and that a working card gets zero mismatches. |
| `--burn-in` | Any card. One hour is enough for everything except the 24-hour figure itself. | Memory use of the process over time, the progress lines, Ctrl-C, and the size of the result file. |
| `--together` | Two or more cards in one machine; four or more is better. | That the cards really are loaded at the same moment, and what loading them from one process costs each card in throughput compared with the same card tested alone. |
| `--links` | Cards with NVLink between them: SXM cards (A100 SXM4, H100 SXM5) in an HGX or DGX chassis. PCIe cards in an ordinary server have no NVLink, and consumer cards have at most an unused bridge connector. | Whether the link queries work in a container, whether the per-link error counters or only the card-wide totals answer, and whether the NVLink data counters rise when PyTorch copies between cards. |

### What to rent, and what it costs

One machine covers all six: **two A100 SXM4 80GB cards in one chassis**. On Vast.ai that was $2.1 per hour on 13 November 2026. Filter for "A100 SXM4", 2 GPUs, a PyTorch image, and check before renting that `nvidia-smi nvlink -s` is going to mean something: the listing should be an HGX or DGX host, not two PCIe A100s.

- Steps 1 to 9 below take about 2.5 to 3 hours including set-up: **about $6 to $7**.
- Optional second machine, for a card without ECC or NVLink and for four cards at once: four RTX 3090s, as rented on 12 November 2026, for about an hour. Use that day's price from the Vast.ai billing page; at the "$0.30 to $2 per hour" range in section 1 it is **$2 or less**.
- Optional, later: a full 24-hour burn-in on one cheap card. At $0.20 to $0.40 per hour for an RTX 3090 that is **$5 to $10**; it only adds the 24-hour memory and file-size figures, so it can wait until someone asks for burn-in.

So the proposal is **about $7, or about $9 with the second machine; allow $10**. Prices move; read the hourly price on the listing before renting and destroy the instance as soon as the steps are done.

Install as in section 2, from the `deeper-tests` branch. Start the local server from the server's `deeper-tests` branch and open the tunnel as in section 4.

### Steps

1. **Check the PyTorch calls first** (two minutes, before any real run):

   ```
   python -m gpucheck.deepchecks --device cpu
   python -m gpucheck.deepchecks --device cuda:0 --seconds 20
   ```

   The `cuda:0` run must end with "self-test: every call was accepted and nothing differed" and exit 0. In its output check: `"skipped": []` (every number format multiplied), `"wrong_products": 0`, `"mismatched_words": 0`, that `bytes_tested` is close to 90% of the card's free memory, and that `seconds` for the memory test is recorded. The `cpu` run uses small sizes and is only a first look: if it lists `float16` or `bfloat16` under `skipped`, that PyTorch build cannot multiply half-precision matrices on a CPU, which says nothing about the GPU; any other failure there is a bug in the check. If a call is rejected or a format is skipped, stop and fix that before spending more time. While the memory test runs, `nvidia-smi` in a second terminal should show the card's memory nearly full; afterwards it should fall back to a few hundred MB.

2. **A plain run is unchanged.** `gpucheck --quick --offline --json plain.json`. The result must have no `checks` key and no `load_bucket_s`, and sustained throughput should match earlier runs of this model.

3. **Thorough run, one card.** `gpucheck --quick --offline --thorough --json thorough.json`, then the same without `--quick` and uploaded to the local server. Check:
   - the report has an "Extra checks" section with a line for the driver error log, the memory test, the answer check and NVLink, and no finding on a card that passes a plain run;
   - `metrics` in `thorough.json` are within noise of `plain.json` (the checks run after the load);
   - `checks.driver_errors.status` is `ran`. If it is `unavailable`, record the `reason` text exactly: it decides whether the event log works in containers at all;
   - `checks.answer`: `products` in the hundreds or thousands, `wrong_products` 0, `skipped` empty, `seconds` about 20 (60 without `--quick`). Record `products`;
   - `checks.memtest`: `bytes_tested` / `bytes_total`, `seconds`, `mismatched_words` 0, `max_temp_c`. Record them: the README still says the time is not measured;
   - `checks.nvlink.status` is `ran` with the link count `nvidia-smi nvlink -s -i 0` shows, and `errors` holds counters (if it is empty, the driver answered neither the per-link nor the card-wide query: record that);
   - the "Not covered" note no longer lists memory cells or silent data corruption and says "data transfer over links between GPUs";
   - on the result page: the "Extra checks in this run" block, the same note, a grade unchanged from a plain run, and the phone layout as in section 5.

4. **Compare the link readings with `nvidia-smi`.** `nvidia-smi nvlink -s` (state per link), `nvidia-smi nvlink -e` (error counters), `nvidia-smi topo -p2p n` (which pairs are NVLink peers). `checks.nvlink.links` and `active` must agree with the first; a counter that is non-zero in the second and does not move during a run must show as a rise of 0, not as its absolute value.

5. **Driver errors are actually delivered** (optional, and only an attempt). There is no safe way to make a sound card fault. During a run with `--driver-errors`, in a second terminal, a deliberate out-of-range index on the same card makes the driver log an application error, which on some drivers is an Xid:

   ```
   python -c "import torch; torch.zeros(4, device='cuda:0')[torch.tensor([10**6], device='cuda:0')].sum().item()"
   ```

   Then compare `sudo dmesg | grep -i xid` (if root is available) with `checks.driver_errors.events`. If `dmesg` shows an Xid and the result does not, record the Xid number: NVML only delivers the ones it classes as critical, and the README must then say which kinds it sees. If neither shows anything, this step proved nothing either way; say so.

6. **One after another, with checks.** `gpucheck --quick --offline --thorough --gpus all --lot-dir ./x-seq`. Each `gpu<N>.json` has its own `checks`; no `transfer` in `checks.nvlink` (copies run only with `--together`); `lot.json` has no `together` key. Keep each card's `metrics.sustained_tflops` for step 7.

7. **All cards at once.** `gpucheck --quick --offline --together --thorough --gpus all --lot-dir ./x-tog`, with `watch -n 1 nvidia-smi` in a second terminal. Check:
   - during the first 30 seconds no card shows a gpucheck process (every card idles through its baseline); then all cards go to full power within a second or two of each other, and stay there together;
   - no card is refused as busy;
   - each result has `checks.together.cards` equal to the number of cards, `preflight.mode` `measurements_only`, and load-phase `power_w` near the power limit on every card (step 4 of the multi-GPU section);
   - **throughput against step 6**: each card's `sustained_tflops` here against the same card tested alone. A few percent lower is the price of sharing power and cooling, or of driving all loads from one process; record the figures per card. If the cards are far lower, or uneven, the one-process design is costing throughput and needs changing before this mode is offered;
   - "Copying test data between N linked cards" is printed; each `checks.nvlink.transfer` has `mismatched_words` 0, and `nvlink_kib` greater than 0. **If `nvlink_kib` is 0 or missing**, PyTorch's copy did not show on the NVLink data counters: compare `nvidia-smi nvlink -gt d` before and after by hand and record what moved. The report must then say the counters did not show the copies, and the note must still list data transfer over links;
   - upload the lot to the local server: the lot page says the cards were loaded at the same time, every position shows "No grade: loaded together with other cards", and none of these runs is counted as a reference (upload a plain run of the same model afterwards and check the `n` it is graded against has not grown).

8. **Ctrl-C.** Start the step 7 command again and press Ctrl-C about a minute in. Expect "Interrupted: stopping every card at its next sample", exit code 130, a `gpu<N>.json` per card with `"aborted": "interrupted"`, `lot.json` with `"finished": true`, and every card's memory released in `nvidia-smi`.

9. **Burn-in, one hour, both cards at once.** In `tmux`: `gpucheck --burn-in 1 --together --thorough --gpus all --offline --lot-dir ./x-burn 2>&1 | tee burn.log`. Check:
   - a `GPU N burn-in: 0 h 10 min of 1 h 0 min of load done, ...` line every ten minutes for each card;
   - the process's memory stays modest (`ps -o rss= -p <pid>` at the start and near the end; expect growth of a few MB per card);
   - each result has `checks.burn_in.completed_s` 3600, `early_tflops` and `late_tflops` within a percent or two of each other (record them: they are the first real data for the 5% `THROUGHPUT_FELL` threshold), 3600 load samples and no `load_bucket_s` (one hour is still stored sample for sample);
   - the "Not covered" note says "faults that appear only after more than 1 hour of use".

   Then, on one card, `gpucheck --burn-in 1.5 --offline --json burn.json` and press Ctrl-C once after about 70 minutes: expect exit code 130, `burn.json` written with `"aborted": "interrupted"`, `samples.load_bucket_s` 2.0, about 2100 load samples and `checks.burn_in.completed_s` near 4200. Upload it with `--upload-saved` and check the chart captions say "Each point covers 2 seconds". (Skip this second part if time is short: it costs another 75 minutes, about $2.60.)

10. **Second machine (optional): four RTX 3090s.** Repeat steps 1, 3 and 7 with `--thorough`. What to look for that the A100s cannot show: the memory test on a card without ECC (zero mismatches, time and coverage on 24 GB); `checks.nvlink.status` `no_links` on every card, no NVLink finding, no "Copying test data" line, and the "Not covered" note still listing links between GPUs; four cards at once against the same four one after another (the multi-GPU section's 12 November figures are the baseline).

11. **Record** for each machine: model, driver, PyTorch and CUDA versions, whether it was a container, and every figure the steps above say to record. Save the result files under a dated folder. Then update the README: replace "not yet run on real hardware" for each check with what was actually run, and fill in the memory test's time.

### If something fails

A finding on a card that passes a plain run is the likeliest failure, and until proven otherwise it is a fault in the check. Keep the result file, note the driver and PyTorch versions, and do not upload it to the public server.
