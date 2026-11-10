# gpucheck

One command that stress-tests an NVIDIA GPU and reports throttling, power capping, recorded memory errors and below-reference throughput.

Website: [gpucheck.dev](https://gpucheck.dev) — includes the optional $29 signed certificate for used-GPU listings.

It records a short idle baseline, runs a sustained matrix-multiply load, and samples temperature, clocks, power, and throttle reasons the whole time. It then explains what it saw in plain language: thermal throttling, power-limit throttling, memory problems, or throughput well below what that model should deliver.

## Install

```
uvx --from "gpucheck-cli[load]" gpucheck --quick
# or
pip install "gpucheck-cli[load]"
```

The load generator needs PyTorch with CUDA. It is an optional extra (`[load]`) because torch is large; if you already have a CUDA build of torch installed, plain `pip install gpucheck-cli` is enough. (The PyPI package is `gpucheck-cli`; the installed command is `gpucheck`.) Without torch/CUDA the tool still runs but generates no load (it prints a warning), so performance metrics are left blank.

## Usage

```
gpucheck                 # standard run: 60 s baseline, 10 min load
gpucheck --quick         # 30 s baseline, 150 s load
gpucheck --long          # 60 s baseline, 30 min load
gpucheck --offline       # never contacts a server
gpucheck --dry-run       # print exactly what would be uploaded, upload nothing
gpucheck --yes-upload    # upload without the prompt
gpucheck --server URL    # upload server (http or https); default https://gpucheck-api.fly.dev
gpucheck --json PATH     # where to write result.json (default ./gpucheck-result.json)
gpucheck --certify       # use a one-time server challenge so the result can earn a certificate
gpucheck --gpus all      # test every GPU in the machine, one after another (a "lot")
gpucheck --gpus 0,2      # test only these GPU indices
gpucheck --label "Rack 4"        # name a multi-GPU run (at most 80 characters)
gpucheck --lot-dir DIR   # where a multi-GPU run writes its files (default ./gpucheck-lot-<id>)
gpucheck --include-serial        # also record the card's serial number
gpucheck --api-key KEY   # send an API key with the upload; prefer the GPUCHECK_API_KEY environment variable
gpucheck --upload-saved PATH     # upload a saved result file or lot directory; tests no GPU
```

Without `--gpus`, gpucheck tests GPU 0 only and says so when the machine has more than one card.

The result is always written locally to `result.json` first. The tool only reads from the GPU; it does not change any settings. Ctrl-C stops a run at any time; an interrupted run writes no result.json and uploads nothing.

### Upload flow

After a run, unless you passed `--offline`, gpucheck asks whether to upload (to `https://gpucheck-api.fly.dev` unless you pass `--server URL`). If you say yes you get a result page URL, a grade such as `Grade B — 94.1% of the median clean reference run for <model> (n=12, low confidence)` (or the reason there is none, for example "insufficient reference (n=X)" while too few comparable runs exist, or "measurements only" for MIG), and a delete token. Save the token: it is the only way to delete the page later.

```
curl -X DELETE -H "X-Delete-Token: <token>" <server>/v1/runs/<run_id>
```

## Testing every GPU in a machine

`gpucheck --gpus all` (or `--gpus 0,2` for a subset, up to 16 cards) tests the cards one after another, never at the same time, and calls the whole run a lot. Each card gets an ordinary single-card run. Before every card after the first, gpucheck waits for that card's idle temperature to stop falling, up to 5 minutes, so that heat from the previous card does not skew the idle baseline. An 8-GPU server can therefore take up to about two hours at the standard length; `--quick` shortens each run.

A lot writes a directory (`--lot-dir`, default `./gpucheck-lot-<id>`) containing `gpu<position>.json` for each card that produced a result, and `lot.json`, which summarises every card, including those that were not tested and why. gpucheck checks that the directory is writable before the first card runs, and refuses to start if it already holds a `lot.json`, so an earlier lot is never overwritten. In a lot run `--json` is not used.

A card that is busy, cannot be read, or fails does not stop the lot; it is recorded as not tested with the reason, and the remaining cards still run. Ctrl-C during a lot keeps the cards already finished, writes the lot, uploads nothing, and exits 130.

`--certify` cannot be used with several cards. A machine with one GPU and `--gpus all` is an ordinary single run.

After a lot, gpucheck asks once whether to upload (or uses `--yes-upload`; with `--dry-run` it prints the request instead). In a non-interactive shell it prints how to upload later with `--upload-saved`.

## What gets uploaded

Nothing, unless you opt in. Uploading is off in non-interactive shells unless you pass `--yes-upload`. `--dry-run` prints the exact payload so you can read it first.

The payload is the contents of `result.json`: GPU model and driver details, measurements, findings, and the time series of samples. Besides the model and driver, each card's record includes, where the card reports them: its GPU index, firmware (VBIOS) version, PCIe link generation and width (current and maximum), and lifetime corrected and uncorrected memory-error counts. A value the card does not report is simply absent. If the PCIe link runs below what the card supports, the report shows a `PCIE_LINK_DEGRADED` finding; a lower generation at idle can be normal power saving. The result also carries the run's start and end times and, in a lot, the lot ID, the card's position and your label. The GPU's UUID is replaced by a salted hash, so the raw UUID is never sent. The salt is a public constant in this source code, so anyone who already knows your GPU's UUID can recompute the hash and recognise your card.

The serial number is never read or uploaded unless you pass `--include-serial`. When you do, the serial appears in the uploaded result and on the public pages, and is also written to your local result files.

All runs for the same card are listed together on a public card history page, which the upload prints as "Card history".

A lot is uploaded in one request. The upload tool does not follow HTTP redirects (so an API key can never be forwarded to another host); `--server` must be the final URL, and a redirect is reported as an error.

Result pages are public to anyone with the link, and the server also serves back the full uploaded JSON (every field above, including the sample time series) at `/v1/runs/<run_id>`. The run ID is derived from the content. Without an API key the server returns a delete token (stored only as a hash) for each run that lets you remove that page. A lot upload saves its tokens in the `uploaded` section of `lot.json`, so anyone who has that file can delete those result pages; keep it private.

With an API key (`--api-key`, or better the `GPUCHECK_API_KEY` environment variable, since command-line arguments are visible to other users of the machine) the upload is attributed to your account and gets no delete token: the uploader cannot delete it. The key is never written to a file, printed, or put in a URL.

### Uploading later (`--upload-saved`)

`gpucheck --upload-saved PATH` uploads a saved result file or a lot directory without touching a GPU, and without a prompt. It uses `--server`, `--api-key` and `GPUCHECK_API_KEY` as usual, and cannot be combined with flags that run a test (`--gpus`, `--quick`, `--long`, `--label`, `--lot-dir`, `--include-serial`, `--offline`, `--dry-run`, `--certify`). It refuses a lot whose card files were changed after the run. After a successful lot upload it records the URLs and delete tokens in `lot.json`; uploading the same lot again does not create duplicates and keeps tokens recorded earlier, and tokens from an upload to a different server are kept under `previous_uploads`. A single result file's delete token is printed, not saved.

## Certificates (`--certify`)

`gpucheck --certify` asks the server for a one-time challenge, runs the workload the server prescribes (a set of matrix sizes and an execution order) instead of the default one, and uploads the result with the challenge attached. If the run qualifies, the result page offers a signed certificate for it. Certifying a run is a separate purchase, made on the result page.

What you need:

- A server that has certification turned on. The default server is `https://gpucheck-api.fly.dev`; pass `--server URL` for another one. If the challenge cannot be fetched (the server has certification turned off, or cannot be reached), gpucheck stops before the run with a message and exit code 1.
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

The report lists plain-language findings, each with a severity and a confidence. Besides the thermal, power and throughput findings (`THERMAL_THROTTLING`, `POWER_CAPPED`, `HOT_ENVIRONMENT`, `UNDERPERFORMING_UNKNOWN`, `INSUFFICIENT_EVIDENCE`), it can report `HOT_UNDER_LOAD` (the card throttled quickly and ran near its temperature limit even though it idled cool), `MEMORY_ERRORS` (errors or remapped memory rows recorded by the card), `MEMORY_NOT_CHECKED` (the card exposes no memory-error counters, so memory health was not checked) and `PCIE_LINK_DEGRADED`. Every report ends with a "Not covered" note listing what the test does not examine.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Run completed |
| 1 | Preflight refused to run (for example the GPU is busy), or the request was refused (for example a bad `--gpus` value) |
| 2 | No NVIDIA GPU or driver available |
| 3 | Run aborted partway (thermal limit, ECC errors, or lost telemetry); result.json is still written |
| 130 | Interrupted with Ctrl-C during a multi-GPU run (finished cards are kept) |

For a multi-GPU run: 0 when every card was tested, 3 when some cards were not tested or a run was aborted (the lot is still written), 1 when no card produced a result or the lot could not be started or saved, and 130 when interrupted. A failed upload does not change the exit code. `--upload-saved` exits 0 on success and 1 on any failure.

## What this test does not cover

This is a pre-flight check, not proof of health. It does not test memory cells, silent data corruption, faults that appear only after hours of use, links between GPUs, or physical damage.

## Limitations

- NVIDIA GPUs on Linux only.
- MIG instances give measurements only; they are not graded.
- Needs PyTorch with CUDA to generate load; without it, performance numbers are blank.
- It refuses to run on a GPU that is already busy.
- Grades depend on a reference set of comparable runs; for new or rare models you will see "insufficient reference".
- A single run is a snapshot. A run with no findings is a snapshot of one test, not a guarantee.
- Multi-GPU runs have been tested only with simulated cards and have not yet been run on real multi-GPU hardware. Until the procedure in `docs/smoke-rented-gpu.md` has been carried out, treat them as unverified.

## License

Apache-2.0
