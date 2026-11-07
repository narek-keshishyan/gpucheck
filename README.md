# gpucheck

One command that stress-tests an NVIDIA GPU and tells you whether it is healthy, throttling, or degraded.

It records a short idle baseline, runs a sustained matrix-multiply load, and samples temperature, clocks, power, and throttle reasons the whole time. It then explains what it saw in plain language: thermal throttling, power-limit throttling, memory problems, or throughput well below what that model should deliver.

## Install

```
uvx --from "gpucheck[load]" gpucheck --quick
# or
pip install "gpucheck[load]"
```

The load generator needs PyTorch with CUDA. It is an optional extra (`[load]`) because torch is large; if you already have a CUDA build of torch installed, plain `pip install gpucheck` is enough. Without torch/CUDA the tool still runs but generates no load (it prints a warning), so performance metrics are left blank.

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
gpucheck --certify       # run the server-prescribed workload so the result can earn a certificate
```

The result is always written locally to `result.json` first. The tool only reads from the GPU; it does not change any settings. Ctrl-C stops a run at any time; an interrupted run writes no result.json and uploads nothing.

### Upload flow

After a run, unless you passed `--offline`, gpucheck asks whether to upload (to `https://gpucheck-api.fly.dev` unless you pass `--server URL`). If you say yes you get a result page URL, a grade such as `Grade B — delivers 94.1% of a healthy <model> (n=12, low confidence)` (or the reason there is none, for example "insufficient reference (n=X)" while too few comparable runs exist, or "measurements only" for MIG), and a delete token. Save the token: it is the only way to delete the page later.

```
curl -X DELETE -H "X-Delete-Token: <token>" <server>/v1/runs/<run_id>
```

## What gets uploaded

Nothing, unless you opt in. Uploading is off in non-interactive shells unless you pass `--yes-upload`. `--dry-run` prints the exact payload so you can read it first.

The payload is the contents of `result.json`: GPU model and driver details, measurements, findings, and the time series of samples. The GPU identifier is replaced by a salted-hash UUID, so the raw serial is never sent. The salt is a public constant in this source code, so anyone who already knows your GPU's UUID can recompute the hash and recognise your card.

Result pages are public to anyone with the link, and the server also serves back the full uploaded JSON (every field above, including the sample time series) at `/v1/runs/<run_id>`. The run ID is derived from the content, and the server returns a delete token (stored only as a hash) that lets you remove your page.

## Certificates (`--certify`)

`gpucheck --certify` asks the server for a one-time challenge, runs the workload the server prescribes (a set of matrix sizes and an execution order) instead of the default one, and uploads the result with the challenge attached. If the run qualifies, the result page offers a signed certificate for it. Certifying a run is a separate purchase, made on the result page.

What you need:

- A server that has certification turned on. The default server is `https://gpucheck-api.fly.dev`; pass `--server URL` for another one. If the challenge cannot be fetched (the server has certification turned off, or cannot be reached), gpucheck stops before the run with a message and exit code 1.
- Working PyTorch with CUDA. Without it, `--certify` stops with exit code 1 instead of falling back to a run with no load.
- The normal upload prompt still applies. A certificate can only be made from an uploaded result, so answer yes, or pass `--yes-upload` in a script.

`--certify` cannot be combined with `--offline` or `--dry-run`; the argument parser rejects the combination at startup.

Fake mode: `--fake` with `--certify` runs mechanically, but a real server rejects it with "upload arrived too soon after challenge", because the simulated run finishes instantly. This is intended: fakes cannot earn certificates.

If the server refuses to certify a run, the upload still succeeds and the result is still stored; the run simply gets no purchase button. The reason is recorded in the server's upload response (`certify_note`), but the result page and the terminal do not show it yet.

### What a certificate proves

A certificate raises the cost of lying about a GPU. It is not proof of anything on its own.

- The run used a challenge the server issued, executed the workload the server prescribed, and was uploaded only after it should have taken to finish.
- The signed payload binds that certificate to one stored result (by its SHA-256), one card (its salted-hash identifier, the same one in the uploaded result), one grade, and one run.
- Certificates issued while the server is in payment test mode carry a TEST banner and `livemode: false` in the signed payload. They are not real purchases.

### What a certificate does not prove

- It does not show that the card is healthy or will stay healthy. A single run is a snapshot, as described under Limitations.
- It does not stop someone who controls the machine. Someone with full control can fabricate telemetry that matches the prescribed workload and timing. The server cannot see the hardware directly.
- It does not stop re-running. A card owner can keep running until a run looks good and then buy a certificate for that one. When a newer certificate is issued for the same card, the older one is marked as superseded, which makes this visible, but it is not prevented.
- Revocation is not part of the signature. A revoked certificate still verifies offline; only the live verify endpoint knows about revocation.

Verifying a certificate offline needs the certificate payload, its signature, and the public key from `/v1/pubkey`. The exact steps and a one-line Python check are in the server's README.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Run completed |
| 1 | Preflight refused to run (for example the GPU is busy) |
| 2 | No NVIDIA GPU or driver available |
| 3 | Run aborted partway (thermal limit, ECC errors, or lost telemetry); result.json is still written |

## Limitations

- NVIDIA GPUs on Linux only.
- MIG instances give measurements only; they are not graded.
- Needs PyTorch with CUDA to generate load; without it, performance numbers are blank.
- It refuses to run on a GPU that is already busy.
- Grades depend on a reference set of comparable runs; for new or rare models you will see "insufficient reference".
- A single run is a snapshot. A healthy result does not guarantee the card will never fail.

## License

Apache-2.0
