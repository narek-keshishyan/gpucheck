# gpucheck

One command that stress-tests an NVIDIA GPU and tells you whether it is healthy, throttling, or degraded.

It records a short idle baseline, runs a sustained matrix-multiply load, and samples temperature, clocks, power, and throttle reasons the whole time. It then explains what it saw in plain language: thermal throttling, power-limit throttling, memory problems, or throughput well below what that model should deliver.

## Install

```
uvx --from "gpucheck[load]" gpucheck --quick
# or
pip install "gpucheck[load]"
```

The load generator needs PyTorch with CUDA. It is an optional extra (`[load]`) because torch is large; if you already have a CUDA build of torch installed, plain `pip install gpucheck` is enough. Without torch the tool cannot generate load.

## Usage

```
gpucheck                 # standard run: 60 s baseline, 10 min load
gpucheck --quick         # 30 s baseline, 150 s load
gpucheck --long          # 60 s baseline, 30 min load
gpucheck --offline       # never contacts a server
gpucheck --dry-run       # print exactly what would be uploaded, upload nothing
gpucheck --yes-upload    # upload without the prompt
gpucheck --server URL    # use a different server
gpucheck --json PATH     # where to write result.json (default ./gpucheck-result.json)
```

The result is always written locally to `result.json` first. The tool only reads from the GPU; it does not change any settings. Ctrl-C stops a run at any time.

### Upload flow

After a run, unless you passed `--offline`, gpucheck asks whether to upload. If you say yes you get a result page URL, a grade (or "insufficient reference (n=X)" while too few comparable runs exist), and a delete token. Save the token: it is the only way to delete the page later.

```
curl -X DELETE -H "X-Delete-Token: <token>" <server>/v1/runs/<run_id>
```

## What gets uploaded

Nothing, unless you opt in. Uploading is off in non-interactive shells unless you pass `--yes-upload`. `--dry-run` prints the exact payload so you can read it first.

The payload is the contents of `result.json`: GPU model and driver details, measurements, findings, and the time series of samples. The GPU identifier is replaced by a salted-hash UUID, so the raw serial is never sent. The run ID is derived from the content, and the server returns a delete token (stored only as a hash) that lets you remove your page.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Run completed |
| 1 | Preflight refused to run (for example the GPU is busy) |
| 2 | No NVIDIA GPU or driver available |
| 3 | Run aborted partway |

## Limitations

- NVIDIA GPUs on Linux only.
- MIG instances give measurements only; they are not graded.
- Needs PyTorch with CUDA to generate load.
- It refuses to run on a GPU that is already busy.
- Grades depend on a reference set of comparable runs; for new or rare models you will see "insufficient reference".
- A single run is a snapshot. A healthy result does not guarantee the card will never fail.

## License

Apache-2.0
