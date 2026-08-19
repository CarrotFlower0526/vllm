# Stack-spec Fleet worker guide

Read this guide before changing this custom vLLM fork for a stack-spec Fleet
experiment. These rules supplement the repository-wide contribution rules and
the parent stack-spec Fleet worker contract.

## Independent implementation input

This vLLM checkout is an independent repository, even when it appears under
`external/vllm` in stack-spec. A worker must:

1. start from the experiment's intended vLLM base commit;
2. commit and push every vLLM change to the registered fork;
3. pin the resulting full commit in stack-spec's implementation definition; and
4. test from a clean worktree of that commit.

Never make a Fleet task depend on a dirty detached checkout, copy individual
vLLM files during preparation, or assume the stack-spec commit captures nested
vLLM changes. An A/B comparison pins two complete repository combinations; it
does not edit one checkout between arms.

## Relocatable runtime behavior

- Do not embed usernames, hosts, home directories, canonical project paths,
  environment paths, model-cache paths, or physical GPU indices.
- Resolve experiment inputs from arguments or the router-provided environment.
  Keep package-internal resources relative to the installed package or checkout.
- Do not discover checkpoints or traces by scanning unrelated historical run
  directories. Missing declared input is an error.
- Treat source, models, upstream results, and the certified environment as
  read-only. Logs, traces, checkpoints, and generated configuration belong under
  the experiment's declared output directory.
- Do not install packages, compile an undeclared extension, download resources,
  or require network access from the experiment command.

Verify the implementation from a fresh checkout whose path and user differ from
the development machine assumptions.

## Frozen-path changes

System-algorithm experiments often add metadata or one output while claiming an
existing proposal path is frozen. Source-level similarity is not sufficient:

- Preserve the original operation shapes, ordering, dtype, device placement,
  kernel eligibility, and synchronization behavior unless the protocol declares
  that path changed.
- Adding rows to a packed projection can change GEMM selection and BF16 rounding
  for existing rows. Compute auxiliary outputs separately until exact parity and
  matched timing justify a fused implementation.
- Create tensors and load auxiliary weights on the consuming module/input device
  and dtype; never assume CPU placement survives model loading.
- Assert the protocol's frozen observable at every relevant state, such as token
  IDs, candidate rows, logits, tree metadata, or verifier queries. Stop at the
  first mismatch and report it as a scientific gate failure.
- Diagnostic tracing must not silently enter the serving hot path. Mark traced
  runs invalid for latency and test the ordinary no-trace path separately.

## Runtime-impact declaration

Classify the diff and update the stack-spec experiment runtime accordingly:

- Python-only source with unchanged dependencies and native interfaces may reuse
  a compatible certified environment.
- Imports, package metadata, requirements, or invoked commands change the runtime
  capability declaration.
- C++/CUDA sources, generated kernels, build flags, extension ABI, or compiled
  package inputs require a rebuilt and recertified runtime.

Do not hide a native rebuild inside a launcher. The router must know before GPU
admission whether it can reuse the target host's environment.

## Required verification and handoff

Before the full Fleet run:

1. run focused vLLM tests for the modified components;
2. run the smallest representative command in a relocated clean worktree;
3. run the protocol's GPU canary when the changed path is GPU-dependent;
4. verify no durable files were created outside declared outputs;
5. verify frozen-path parity and, for hot-path changes, paired timing; and
6. push the tested commit before sealing the experiment lock.

The handoff names the vLLM base and final commits, changed behavior, runtime-impact
class, tests, parity/timing gates, and exact stack-spec implementation that pins
it. If any of those is unknown, the implementation is not ready to submit.
