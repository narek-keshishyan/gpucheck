# gpucheck

One command that stress-tests an NVIDIA GPU and reports throttling, power capping, recorded memory errors and below-reference throughput.

Website: [gpucheck.dev](https://gpucheck.dev).

It records a short idle baseline, runs a sustained matrix-multiply load, and samples temperature, clocks, power, and throttle reasons the whole time. It then explains what it saw in plain language: thermal throttling, power-limit throttling, memory problems, or throughput well below what that model should deliver.

## Install

```
uvx --from "gpucheck-cli[load]" gpucheck --quick
# or
pip install "gpucheck-cli[load]"
```

The load generator needs PyTorch with CUDA. It is an optional extra (`[load]`) because torch is large; if you already have a CUDA build of torch installed, plain `pip install gpucheck-cli` is enough. (The PyPI package is `gpucheck-cli`; the installed command is `gpucheck`.) Without torch/CUDA a single-card run still runs but generates no load (it prints a warning), so performance metrics are left blank. A multi-GPU run needs the load generator and refuses to start without it.

## Usage

```
gpucheck                 # standard run: 60 s baseline, 10 min load
gpucheck --quick         # 30 s baseline, 150 s load
gpucheck --long          # 60 s baseline, 30 min load
gpucheck --offline       # never contacts a server
gpucheck --dry-run       # print exactly what would be uploaded, upload nothing
gpucheck --yes-upload    # upload without the prompt
gpucheck --server URL    # upload server; default https://gpucheck-api.fly.dev (an API key is only sent over https)
gpucheck --json PATH     # where to write result.json (default ./gpucheck-result.json)
gpucheck --certify       # use a one-time server challenge so the result can earn a certificate (off on the default server)
gpucheck --gpus all      # test every GPU in the machine, one after another (a "lot")
gpucheck --gpus 0,2      # test only these GPU indices
gpucheck --gpus all --label "Rack 4"   # name a lot (at most 80 characters)
gpucheck --gpus all --lot-dir DIR      # where a lot writes its files (default ./gpucheck-lot-<id>)
gpucheck --include-serial        # also record the card's serial number
gpucheck --api-key KEY   # send an account key with the upload (needed to upload a lot); prefer the GPUCHECK_API_KEY environment variable
gpucheck --upload-saved PATH     # upload a saved result file or lot directory; tests no GPU

# Extra checks: all off by default (see "Extra checks" below)
gpucheck --thorough      # --driver-errors, --answer-check, --memtest and --links together
gpucheck --driver-errors # listen for hardware error events (Xid) the driver reports during the run
gpucheck --answer-check  # after the load, compute products with a known result and count wrong ones
gpucheck --memtest       # after the load, write patterns across most of the card's memory and read them back
gpucheck --links         # read NVLink link state and error counters before and after the run
gpucheck --burn-in 24    # a load of 24 hours (1 to 72) instead of 10 minutes
gpucheck --gpus all --together   # load every selected card at the same time (measurements only)
```

Without `--gpus`, gpucheck tests GPU 0 only and says so when the machine has more than one card.

A single-card result is always written locally to `result.json` first, before any upload; a lot (several cards, or one card run with `--lot-dir` or `--label`) writes its lot directory card by card as each card finishes. The tool only reads from the GPU; it does not change any settings. Ctrl-C stops a run at any time. For a plain single-card run, an interrupted run writes no result.json and uploads nothing. An interrupted or killed lot keeps the cards that finished: see below.

### Upload flow

After a run, unless you passed `--offline`, gpucheck asks whether to upload (to `https://gpucheck-api.fly.dev` unless you pass `--server URL`). If you say yes you get a result page URL, a grade such as `Grade B — 94.1% of the median clean reference run for <model> (n=12, low confidence)` (or the reason there is none, for example "insufficient reference (n=X)" while too few comparable runs exist, or "measurements only" for MIG), and a delete token. Save the token: it is the only way to delete the page later.

Running the test, reading the report on your own machine, and uploading a single card are free and need no account. Uploading a lot (every card of a machine in one report, see below) to the gpucheck server needs an account key; write to support@gpucheck.dev to get one.

```
curl -X DELETE -H "X-Delete-Token: <token>" <server>/v1/runs/<run_id>
```

## Testing every GPU in a machine

`gpucheck --gpus all` (or `--gpus 0,2` for a subset, up to 16 cards) tests the cards one after another, never at the same time (unless you ask for that with `--together`, see "Extra checks"), and calls the whole run a lot. Each card gets an ordinary single-card run. Before every card after the first, gpucheck waits for that card's idle temperature to stop falling, up to 5 minutes, so that heat from the previous card does not skew the idle baseline. An 8-GPU server can therefore take up to about two hours at the standard length; `--quick` shortens each run. While a lot runs, gpucheck prints one line when each card starts and one when it ends, also when the output goes to a file; the per-card reports and the summary follow when every card is done.

A lot writes a directory (`--lot-dir`, default `./gpucheck-lot-<id>`) containing `gpu<position>.json` for each card that produced a result, and `lot.json`, which summarises every card, including those that were not tested and why. Before the first card runs, gpucheck creates `lot.json`, which both checks that the directory is writable and claims it for this run: one lot directory is used by one run only. A run pointed at a directory that already holds a `lot.json` is refused, whether that lot finished long ago or is still running (for example a `nohup` run and the same command started again), so an earlier lot is never overwritten. In a lot run `--json` is not used.

The card files follow your umask, like the single-run result file, while `lot.json` is readable only by the user who ran the test, because it holds delete tokens after an upload. Run `--upload-saved` as that same user, or copy the directory preserving ownership.

Results are saved card by card: as soon as a card finishes, its `gpu<position>.json` is written and `lot.json` is rewritten. The first `lot.json`, written before any card runs, is created in place to claim the directory; every later write of a card file or of `lot.json` goes through a temporary file and a rename, so it is never left half-written. While the lot is running, `lot.json` has `"finished": false` and lists the cards not yet reached as not tested ("not reached: the run stopped before this card"). If the process is killed, the SSH session drops or the machine crashes, the cards that finished are on disk, and `gpucheck --upload-saved DIR` uploads them, with the cards not reached recorded as not tested. A long lot can take hours, so run it inside `tmux` or `screen`, or with `nohup`, so that a dropped connection does not stop it. If a card's files cannot be saved partway through, gpucheck prints a warning, keeps going and tries again at the end.

A card that is busy, cannot be read, or fails does not stop the lot; it is recorded as not tested with the reason, and the remaining cards still run. Ctrl-C during a lot keeps the cards already finished, writes `lot.json` with `"finished": true` (the remaining cards are recorded as interrupted), uploads nothing, and exits 130.

A multi-GPU run needs the load generator (PyTorch with CUDA, the `[load]` extra): without it every card would sit idle, so gpucheck refuses to start and names the install command.

GPU numbering: PyTorch and NVML must agree on which card is GPU N, or the tool would load one card while measuring another. gpucheck therefore sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` before loading PyTorch (if it was set to something else, it says it is overriding it), and on a machine with more than one GPU it refuses to run while `CUDA_VISIBLE_DEVICES` is set, because PyTorch would then number only the visible cards. This includes `CUDA_VISIBLE_DEVICES` set to an empty value, which hides every GPU from PyTorch. Unset it and choose cards with `--gpus` instead. This also applies to a single-card run on a multi-GPU machine. MIG-enabled machines are not supported for multi-GPU runs, because the device numbering is not guaranteed there.

`--certify` cannot be used with a lot. A single card (a machine with one GPU and `--gpus all`, or `--gpus N`) is an ordinary single run, unless you pass `--lot-dir` or `--label`: then that one card is written as a lot of one, and `--json` is not used. On a machine with more than one GPU, `--lot-dir` or `--label` needs `--gpus` as well: gpucheck refuses rather than make a labelled lot out of GPU 0 alone.

The lot is always tested and saved on your machine, with or without an account. Uploading it to the gpucheck server needs an account key, because the public lot report is the paid part of gpucheck; a single card can be uploaded by anyone, so a buyer can re-run the test on one card and compare. To get a key, write to support@gpucheck.dev. Give it to gpucheck in the `GPUCHECK_API_KEY` environment variable:

```
export GPUCHECK_API_KEY='<your key>'
gpucheck --gpus all
```

With a key set, gpucheck asks once after the lot whether to upload (or uses `--yes-upload`; with `--dry-run` it prints the request instead). In a non-interactive shell it prints how to upload later with `--upload-saved`. Ctrl-C during the upload stops it and prints the same retry command.

With no key set, gpucheck does not ask about the gpucheck server: it says that the lot is saved, how to get a key, and the command that uploads the saved lot once the key is set, for example:

```
Not uploaded: https://gpucheck-api.fly.dev takes a lot only with an account key, and no key was given. (A single card uploads without one.)
The lot is saved in gpucheck-lot-1a2b3c4d.
To get a key, write to support@gpucheck.dev.
Then set the GPUCHECK_API_KEY environment variable to the key and run: gpucheck --upload-saved gpucheck-lot-1a2b3c4d
```

Nothing has to be tested again. `--yes-upload` and `--upload-saved` send the lot even without a key and print the same message if the server refuses it. With `--server URL` pointing at another server, gpucheck asks as before, since a server you run yourself may accept lots without a key; if that server refuses for want of a key, the message says to ask whoever runs it. A key that the server does not know is reported as "API key was rejected by the server".

Uploading a lot needs a server deployed with lot support; an older server answers 404, the upload is reported as failed, and the lot stays saved for `--upload-saved`.

## Extra checks

A plain `gpucheck` run is exactly what it was before these existed: the same baseline, the same load, the same result file. Each check below is opt-in, and a result says which ones ran, in an "Extra checks" section of the report and a `checks` block in the result file. None of them has been run on a real GPU yet (see Limitations); they were developed against simulated cards.

`--thorough` turns on the four that only add minutes and leave the graded measurement alone: `--driver-errors`, `--answer-check`, `--memtest` and `--links`. They run after the load (or only listen during it), so a thorough run's throughput, temperatures and grade are measured exactly as in a plain run, and it can still serve as a reference. `--burn-in` and `--together` are not part of it: one costs hours, the other changes what is measured.

| Flag | What it checks | Time it adds | What it cannot catch |
| --- | --- | --- | --- |
| `--driver-errors` | Listens for the hardware error events the NVIDIA driver reports for the card (Xid errors, double-bit and other memory errors) from before the baseline to the end of the run. | None. | Errors the driver does not report as an event, anything from before the run started, and errors on a system where the driver refuses a listener (some containers): the report then says the log was not available. It never reads the kernel log, which needs root. |
| `--answer-check` | After the load, while the card is hot, multiplies matrices whose correct product is known in advance, in four number formats (float32, float16, bfloat16, float64), and counts products that differ. | About 1 minute (20 s with `--quick`). | Most silent data corruption. It catches a card that returns wrong numbers for these products in that minute. Corruption that needs other operations, other data or more time does not show. By published accounts from large fleets, short synthetic checks miss most affected chips. |
| `--memtest` | After the load has released its memory, writes six patterns (all zeros, all ones, two checkerboards, each word's own address, pseudo-random) across 90% of the card's free memory, keeping 1 GiB back, reads each back and counts the 64-bit words that differ. | Not measured yet. It depends on the card's memory size; expect under a minute to a few minutes. | Memory it could not allocate (the driver's own and the part kept back). On a card with ECC, an error the card corrects on the way out: that shows in the ECC counters, which the run records during the test and reports as `MEMORY_ERRORS`. Faults that only appear at other temperatures or with other patterns. |
| `--links` | Reads each NVLink link's state and its replay, recovery and CRC error counters before the baseline and after the run. With `--together` it also copies 256 MiB of test data between every pair of linked cards, there and back, and compares. | Seconds. | Whether a link is fast. Faults that a short copy does not provoke. Links to cards that were not selected. A card without an active NVLink link reports nothing: that is not a pass. |
| `--burn-in HOURS` | The standard run with a load of 1 to 72 hours instead of 10 minutes. Throttling, memory errors, driver errors and a temperature abort are watched the whole time, and throughput in the final part of the run is compared with the first ten minutes. | The hours you ask for. | Faults that need longer than the run. A burn-in is graded but never used as a reference. |
| `--together` | In a lot, loads every selected card at the same time instead of one after another, so the machine's power supply and cooling carry the full load. | Makes a lot shorter: one run length for all cards. | Which card is at fault when all slow down together. The results are measurements only: not graded, and never used as references for cards measured alone. |

### Driver error log (`--driver-errors`)

gpucheck registers with the driver, through NVML's event functions, as a listener for error events on the card it is testing, and collects what arrives at every sample. Registering changes no GPU setting. Each event is recorded with the phase it arrived in (baseline, load, or the checks after the load) and, for an Xid, its number. Any event is a `DRIVER_ERRORS` finding. Xid numbers are NVIDIA's driver error codes; NVIDIA publishes what each means.

### Answer check (`--answer-check`)

The operands are built from a Hadamard matrix, whose rows are all at right angles to each other. With its rows shuffled and scaled by signed powers of two, the product of one such matrix with the transpose of another has exactly one known non-zero entry in each row and zeros everywhere else. The expected result is written down from the shuffles; nothing is compared with a second computation on the same card. Every case is built so that it is exact in its own number format whatever order the card adds the terms in, so a card that works cannot be reported as wrong for rounding. A product with any entry that differs is counted, and any wrong product is a `WRONG_ANSWERS` finding. A number format the card cannot multiply is skipped and listed, not counted as wrong.

A result that ran this check says so in these words and no stronger ones: it can catch some silent data corruption and cannot rule it out.

### Memory test (`--memtest`)

It runs as its own phase after the load and the answer check, once the load has freed its buffers, and frees its own memory when it ends, so it never holds memory the load needs. Every pattern is written to the whole tested area before any of it is read back. The card keeps being sampled between slices of work, so the temperature limit still stops the run. Any word that differs is a `MEMTEST_MISMATCH` finding.

### Burn-in (`--burn-in HOURS`)

`--burn-in` cannot be combined with `--quick`, `--long` or `--certify`. Run it inside `tmux` or `screen`, or with `nohup`.

- A line is printed every ten minutes of load (`burn-in: 3 h 10 min of 24 h 0 min of load done, 71 C, 724.1 TFLOPS`), so a log shows how far a run got if the machine goes down.
- The first Ctrl-C stops the run at its next sample and writes what was measured as a run that was stopped early (`"aborted": "interrupted"`, exit code 130, never uploaded on its own; upload it later with `--upload-saved` if you want it on a page). A second Ctrl-C quits at once without a result, like Ctrl-C in any other single-card run.
- All measurements are computed from every one-second sample. The result file would be far too large to upload with a day of them, so a load of more than an hour is stored as at most 3600 summaries: `samples.load_bucket_s` says how many seconds each covers (24 for a 24-hour run), and each summary keeps the highest temperature and fan speed, the lowest clock, every throttle reason seen, the latest error count and the mean power and throughput of its span. A 72-hour result is about half a megabyte. Loads of up to an hour (`--quick`, the standard run, `--long`) are stored sample for sample, as before.
- The result's `checks.burn_in` records how long the load really ran, the throughput where a standard run reads it (the last two minutes of the first ten) and the throughput over the final 20%. A fall of more than 5% is a `THROUGHPUT_FELL` finding. That threshold has not been calibrated on real hardware.
- The "not covered" note names the hours the load really ran: "faults that appear only after more than 24 hours of use".
- One-second samples are held in memory until the run ends: about 20 MB per card per day.

### All cards at once (`--together`)

`gpucheck --gpus all --together` needs at least two selected cards. Every card idles through the baseline at the same time; then all loads start together, and each card is sampled as in a single run. A card that is busy, cannot be read or fails is recorded as not tested and does not hold the others up. No load is started on any card before every card has passed its busy check.

A card loaded alongside others shares the machine's power supply and cooling with them, so its result is marked as measured together (`checks.together`, and `preflight.mode` is `measurements_only` with the reason). It is not graded and is never used as a reference. For grades, run the ordinary lot (one card after another) as well. If only one card ends up being loaded, because the others were refused, its run is an ordinary one.

Ctrl-C stops every card at its next sample, keeps what each measured as a run that was stopped early, uploads nothing and exits 130. A second Ctrl-C quits without keeping anything. Unlike the one-after-another lot, the card files are written when all cards have finished, not one by one.

All loads run in one process, one thread per card. Whether that costs throughput compared with a card tested alone has not been measured.

### Links between cards (`--links`)

Reading the links uses NVML getters only. Counters that rose during the run, or a link that was active before it and down after it, are an `NVLINK_ERRORS` finding (a warning: low error rates on a link can be normal). Some links inactive is reported as `NVLINK_LINKS_INACTIVE`, for information.

The copy between cards runs only with `--together`, after every card has finished, and only between cards that NVML reports as NVLink peers. PyTorch and the driver choose the path a copy takes, so gpucheck also reads the cards' NVLink data counters around the copies: only when they rose does the result say the copy went over the links. Otherwise it says the counters did not show it, and the "not covered" note still lists data transfer over the links. Data that comes back changed is a `LINK_TRANSFER_MISMATCH` finding on both cards of the pair, because a copy cannot tell which end is at fault; the finding says with how many of its peers each card differed, and the card that differs with all of them is the likelier cause.

### Checking the PyTorch calls without a GPU run

`python -m gpucheck.deepchecks` runs the answer check and the memory test against the installed PyTorch and prints what they report. On a machine without a GPU it uses the CPU and small sizes: it only shows that PyTorch accepts every call and that a working machine gets no mismatch. With `--device cuda:0` it runs both at full size on that card.

## What gets uploaded

Nothing, unless you opt in. Uploading is off in non-interactive shells unless you pass `--yes-upload`. `--dry-run` prints the exact payload so you can read it first.

The payload is the contents of `result.json`: GPU model and driver details, measurements, findings, and the time series of samples. Besides the model and driver, each card's record includes, where the card reports them: its GPU index, firmware (VBIOS) version, PCIe link generation and width (current and maximum), and lifetime corrected and uncorrected memory-error counts. A value the card does not report is simply absent. The current PCIe link is read while the load runs, because a link drops to a lower speed at idle to save power, and the record says that it was read under load. If the link then runs below what the card and slot support, the report shows a `PCIE_LINK_DEGRADED` finding. When no load ran, the link is read at idle: only a narrower width is reported, and the speed is not judged. The result also carries the run's start and end times and, in a lot, the lot ID, the card's position and your label. A run with extra checks also carries a `checks` block with what each check counted (for the driver error log, the events with their Xid numbers), and a burn-in's load samples are summaries (see "Burn-in"). The GPU's UUID is replaced by a salted hash, so the raw UUID is never sent. The salt is a public constant in this source code, so anyone who already knows your GPU's UUID can recompute the hash and recognise your card.

The serial number is never read or uploaded unless you pass `--include-serial`. When you do, the serial appears in the uploaded result and on the public pages, and is also written to your local result files.

All runs for the same card are listed together on a public card history page, which the upload prints as "Card history".

A lot is uploaded in one request. The upload tool does not follow HTTP redirects (so an API key can never be forwarded to another host); `--server` must be the final URL, and a redirect is reported as an error.

Result pages are public to anyone with the link, and the server also serves back the full uploaded JSON (every field above, including the sample time series) at `/v1/runs/<run_id>`. The run ID is derived from the content. Without an API key the server returns a delete token (stored only as a hash) for each run that lets you remove that page. The gpucheck server takes a lot only with an account key, so a lot uploaded there has no delete tokens. A server that accepts a lot without a key returns one token per card, and gpucheck saves them in the `uploaded` section of `lot.json`, so anyone who has that file can delete those result pages; keep it private.

With an API key (`--api-key`, or better the `GPUCHECK_API_KEY` environment variable, since command-line arguments are visible to other users of the machine) the upload is attributed to your account, whose name is shown on the public lot page and card history pages, and gets no delete token: the uploader cannot delete it (write to support@gpucheck.dev to have a page removed). A key is optional for a single card and required for a lot on the gpucheck server. The key is never written to a file, printed, or put in a URL, and it is only sent over https: with a key set, gpucheck refuses to upload to a plain `http://` server unless the host is `localhost`, `127.0.0.1` or `[::1]`, and treats that as a failed upload (the files are kept). An anonymous upload that comes back without a delete token was already uploaded before, and gpucheck says so.

### Uploading later (`--upload-saved`)

`gpucheck --upload-saved PATH` uploads a saved result file or a lot directory without touching a GPU, and without a prompt. It uses `--server`, `--api-key` and `GPUCHECK_API_KEY` as usual, and cannot be combined with flags that run a test (`--gpus`, `--quick`, `--long`, `--burn-in`, `--label`, `--lot-dir`, `--include-serial`, `--offline`, `--dry-run`, `--certify`, and the extra-check flags). It refuses a lot whose card files were changed after the run, and a single `gpu<N>.json` from a lot (upload its directory instead). A lot that did not finish (its run was killed) is uploaded with a note that the cards not reached are recorded as not tested. After a successful lot upload it records the URLs and delete tokens in `lot.json`; uploading the same lot again does not create duplicates and keeps tokens recorded earlier, and tokens from an upload to a different server are kept under `previous_uploads`. A single result file's delete token is printed, not saved. If the server refuses a lot because no key was given, gpucheck exits 1, leaves the directory unchanged, and prints how to get a key and the command to run again once `GPUCHECK_API_KEY` is set.

## Certificates (`--certify`)

Certification is switched off on the default server and certificates are not currently offered. Against the default server, gpucheck --certify stops before the run with "certification unavailable: server has certification disabled" (exit code 1). The rest of this section describes the feature for a server that has it turned on.

`gpucheck --certify` asks the server for a one-time challenge, runs the workload the server prescribes (a set of matrix sizes and an execution order) instead of the default one, and uploads the result with the challenge attached. If the run qualifies, the result page offers a signed certificate for it. On a server that offers certificates, one can be requested from the result page.

What you need:

- A server that has certification turned on. The default server, `https://gpucheck-api.fly.dev`, has it turned off; pass `--server URL` for a server that has it on. If the challenge cannot be fetched (the server has certification turned off, or cannot be reached), gpucheck stops before the run with a message and exit code 1.
- Working PyTorch with CUDA. Without it, `--certify` stops with exit code 1 instead of falling back to a run with no load.
- The normal upload prompt still applies. A certificate can only be made from an uploaded result, so answer yes, or pass `--yes-upload` in a script.

`--certify` cannot be combined with `--offline`, `--dry-run`, or `--quick`; the argument parser rejects the combination at startup. Certificates need a full-length run (at least 600 s of load), so a quick run could never earn one.

If the challenge expires before the run finishes (challenges are valid for 2 hours), gpucheck prints one warning, `challenge expired before upload; run again to certify — result saved locally`, does not upload, and exits normally. The result is still written to the `--json` path.

Fake mode: `--fake` with `--certify` runs mechanically, but a real server rejects it with "upload arrived too soon after challenge", because the simulated run finishes instantly. This is intended: fakes cannot earn certificates.

After a `--certify` upload, gpucheck prints the server's verdict: `Certifiable: yes`, or `Not certifiable: <reason>` with the reason the server gave (`certify_note`). If the server refuses to certify a run, the upload still succeeds and the result is still stored; the run simply gets no purchase button. A certifiable run also needs a grade, a complete (not aborted, not quarantined) load phase, and a card identity before the result page offers a certificate.

### What a certificate proves

A certificate raises the cost of lying about a GPU. It is not proof of anything on its own.

- The run used a fresh one-time server challenge and could not be uploaded before the full run time had passed. The uploaded result echoes the challenge's workload parameters, but the server cannot see whether that workload actually ran.
- The signed payload binds that certificate to one stored result (by its SHA-256), one card (its salted-hash identifier, the same one in the uploaded result), one grade, and one run.
- Certificates issued while the server is in payment test mode carry a TEST banner and `livemode: false` in the signed payload. They are not real purchases.

### What a certificate does not prove

- It does not show that the card is healthy or will stay healthy. A single run is a snapshot, as described under Limitations.
- It does not stop someone who controls the machine. Someone with full control can fabricate telemetry that matches the prescribed workload and timing. The server cannot see the hardware directly.
- It does not stop re-running. A card owner can keep running until a run looks good and then buy a certificate for that one. When a newer certificate is issued for the same card, the older one is marked as superseded, which makes this visible, but it is not prevented.
- Revocation is not part of the signature. A revoked certificate still verifies offline; only the live verify endpoint knows about revocation.

Verifying a certificate offline needs the certificate payload, its signature, and the public key from `/v1/pubkey`. The exact steps and a one-line Python check are on the server's `/verify` page and in the server's README.

## Findings

The report lists plain-language findings, each with a severity and a confidence. Besides the thermal, power and throughput findings (`THERMAL_THROTTLING`, `POWER_CAPPED`, `HOT_ENVIRONMENT`, `UNDERPERFORMING_UNKNOWN`, `INSUFFICIENT_EVIDENCE`), it can report `HOT_UNDER_LOAD` (the card throttled quickly and ran near its temperature limit even though it idled cool), `MEMORY_ERRORS` (errors during the run, remapped memory rows, or uncorrected memory errors over the card's lifetime, recorded by the card), `MEMORY_NOT_CHECKED` (the card exposes no memory-error counters, so memory health was not checked) and `PCIE_LINK_DEGRADED`. `POWER_CAPPED` means the card's power limit is set below 90% of its default; a card that regulates at its stock limit under full load is not flagged. (When the limits cannot be read, power capping seen on a cool card is reported instead.) `UNDERPERFORMING_UNKNOWN` is reported when sustained throughput is below 90% of the datasheet peak with no throttling seen, or below 40% on a card that only shows power regulation at its stock limit. That second figure is only a floor for gross shortfalls: healthy A100s regulating at their own limit measured between 60% and 86% of the datasheet peak depending on the variant's power limit, so the datasheet is a poor yardstick there and the grade on the result page, which compares a card with others of its model, is the one to read. Every report ends with a "Not covered" note listing what the test does not examine.

The extra checks add these: `DRIVER_ERRORS` (the driver reported hardware error events during the run), `MEMTEST_MISMATCH` (the memory test read back data that differed from what it wrote), `WRONG_ANSWERS` (the answer check got a product wrong), `LINK_TRANSFER_MISMATCH` (test data copied between two cards came back changed), `NVLINK_ERRORS` (NVLink error counters rose or a link went down during the run), `NVLINK_LINKS_INACTIVE` (some of the card's NVLink links were inactive), `THROUGHPUT_FELL` (a burn-in's final throughput was more than 5% below its first ten minutes) and `CHECK_NOT_RUN` (a check that was asked for could not run or did not finish, so what it would have covered is not covered). ECC errors counted while the answer check or the memory test ran are reported under `MEMORY_ERRORS`. On the result page a failed memory test, answer check or card-to-card copy makes the grade an F, and driver error events cap it at D.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Run completed |
| 1 | Preflight refused to run (for example the GPU is busy), or the request was refused (for example a bad `--gpus` value) |
| 2 | No NVIDIA GPU or driver available |
| 3 | Run aborted partway (thermal limit, ECC errors, or lost telemetry); result.json is still written |
| 130 | Interrupted with Ctrl-C during a lot (finished cards are kept), during a burn-in (what was measured is written) or during an `--upload-saved` upload (nothing on disk is changed) |

For a lot: 0 when every card was tested, 3 when some cards were not tested or a run was aborted (the lot is still written), 1 when no card produced a result or the lot could not be started or saved, and 130 when interrupted. A failed upload does not change the exit code. `--upload-saved` exits 0 on success, 1 on any failure, and 130 when the upload is interrupted with Ctrl-C (nothing on disk is changed).

## What this test does not cover

This is a pre-flight check, not proof of health. It does not test memory cells, silent data corruption, faults that appear only after hours of use, links between GPUs, or physical damage.

That is the note for a plain run. The note is written per result, so a run that did an extra check says what that check reached instead of saying the area is untested, and a run that did not still says it is untested:

- After `--memtest`, "memory cells" is replaced by how much of the card's memory the patterns reached and that they cannot rule out faults they do not provoke.
- After `--answer-check`, "silent data corruption" is replaced by the number of products compared and "it can catch some silent data corruption and cannot rule it out".
- After a burn-in, the note says "faults that appear only after more than N hours of use", with the hours the load really ran.
- After `--links` on a card with active NVLink links, the note says the link state and counters were read and that data transfer over the links is not tested, unless a copy between cards was seen on the NVLink data counters.
- A check that could not run, or stopped early, changes nothing in the note.

Physical damage is never tested. A lot's summary, and the lot and card pages, list only what no run on the page tested and point to each run's own report for the rest.

## Limitations

- NVIDIA GPUs on Linux only.
- MIG instances give measurements only; they are not graded.
- Needs PyTorch with CUDA to generate load; without it, performance numbers are blank.
- It refuses to run on a GPU that is already busy.
- Grades depend on a reference set of comparable runs; for new or rare models you will see "insufficient reference".
- A single run is a snapshot. A run with no findings is a snapshot of one test, not a guarantee.
- None of the extra checks (`--driver-errors`, `--answer-check`, `--memtest`, `--links`, `--burn-in`, `--together`, `--thorough`) has been run on a real GPU yet: they were developed on a machine without one, against simulated cards. The answer check and the memory test have been run against real PyTorch (2.15.0) on that machine's CPU only (14 November 2026): every number format multiplied at its full matrix size with no wrong product, the memory patterns read back without a mismatch, and faults injected on purpose were counted. That says nothing about a GPU's own multiply routines, memory or driver. The driver error log, the NVLink readings, the card-to-card copy and the all-at-once mode have not touched real hardware or a real driver at all. Until the procedure in `docs/smoke-rented-gpu.md` ("Extra checks") has been run, treat every one of them as unverified, including the claim that a working card gets no finding from them.
- The extra checks need a server that knows them. An older server still accepts the upload but drops the `checks` block and shows the new findings only as "findings this server does not recognise".
- Real-hardware coverage so far is one machine with four RTX 3090s and one A100 80GB (November 2026, procedure in `docs/smoke-rented-gpu.md`). Lots on datacenter cards, machines with more than four GPUs, H100 and other models have not been run on real hardware; treat them as unverified.

## License

Apache-2.0
