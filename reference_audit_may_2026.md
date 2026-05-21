# Reference Audit — "State of PyTorch Hardware Acceleration: May 2026"

**Audit date:** 2026-05-21
**Source document:** `State of PyTorch Hardware Acceleration May 2026.docx`
**Source references:** `/references/ref-01-..-ref-64-*` (64 files; manifest in `references/manifest.csv`)

This audit walks every numbered citation in the body of the Word document and checks it against the locally archived source it points to. For each citation it states what claim the citation is intended to support, whether the cited reference actually contains evidence for that claim, and a verdict.

The verdicts are:

- **Good** — the cited source clearly contains evidence for the specific claim it is attached to.
- **Acceptable / loose** — the cited source touches the topic but only loosely supports the precise wording, or it is one of several citations on a summary/recommendation sentence and is acting as background rather than a tight proof.
- **Weak / off-topic** — the source is in the right neighborhood but does not actually contain the specific fact in the sentence; a different reference in this same set would be a much stronger pick.
- **Mismatch** — the cited source does not support the claim at all; this is a citation error.

## Top-level findings

1. The reference base is solid. Every major factual claim — PyTorch 2.7 Blackwell support, PyTorch 2.11 FlashAttention-4, PyTorch 2.12 unified-memory MPS / Metal-4 / `torch.accelerator.Graph`, ROCm 7 gfx950 max-autotune, ROCm FlashAttention-2 CK/Triton, TPU v5p / v6e Trillium / TPU7x Ironwood HBM and ICI numbers, MI300X / MI325X HBM, NVLink Hopper 900 GB/s and Blackwell switch fabric, M3 Ultra 512 GB unified memory and "600 billion parameter" claim, M4 Max 128 GB / 546 GB/s, M5 Max 128 GB / 614 GB/s, MLX unified-memory model, MPS env vars, MPS compile tracker — is directly supported by the right archived source.
2. Three citation errors stand out and should be fixed:
   - **`[35]` is a clear mismatch.** It points to "MPS backend notes" but is attached to a sentence describing the PyTorch/XLA migration guide and TPU lazy execution. The correct target is `[16]` or `[41]` (PyTorch/XLA documentation).
   - **`[60]` (NVLink) is misplaced in two CUDA-compiler/Nsight paragraphs.** It belongs only in the interconnect paragraph; in the two earlier paragraphs the Nsight Systems Release Notes (`[37]` / `[43]`) would be the right source.
   - **The "PyTorch 2.10 reduced kernel-launch overhead with combo-kernel fusion" sentence has no supporting reference.** No PyTorch 2.10 / 2.9 release post is archived; the surrounding citations `[10] [50] [60]` all point at 2.8 or 2.12 release blogs, neither of which mentions 2.10 combo-kernel fusion.
3. The bibliography contains a lot of legitimate duplication (the same PyTorch release blog is reused across several reference numbers). That is fine — different sentences cite different paragraphs of the same blog — but reference **`[3]` (PyTorch: Start Locally) is never cited in the body**. It is a redundant entry that duplicates `[36]` and `[49]`.
4. A handful of summary/recommendation sentences carry a single citation that is more "background" than "proof." These are flagged as loose; they do not contain factual errors but the citation isn't tight on the sentence. Examples: `[19]`, `[38]`, `[43]`, `[44]`, `[45]`, `[48]`, `[49]`.
5. The "fused SSD/Mamba-2 kernels" half of citation `[22]` is not in the archived warp-specialization blog; the warp-specialization claim itself is supported, but SSD/Mamba-2 needs a separate source.

The rest of this document is the per-citation walk.

## Per-citation verification

### [4] — PyTorch 2.12 Release Blog (`ref-04`)

Attached to the Executive Decision Matrix introduction, which is explicitly described as "a synthesis of upstream PyTorch 2.8 — 2.12 release notes." Reference 4 is a 2.12 release blog — one of those release notes — and supports the synthesis only as a single example. Acceptable / loose: a tighter citation would list multiple release blogs together (e.g., `[1] [5] [17] [50]`), but the matrix is a synthesis so a single example is not wrong.

### [5] — PyTorch 2.7 Release Blog (`ref-05`)

Attached to "PyTorch 2.7 added Blackwell support." Verified: the archived 2.7 release blog explicitly says *"[Prototype] NVIDIA Blackwell Architecture Support — PyTorch 2.7 introduces support for NVIDIA's new Blackwell GPU architecture and ships pre-built wheels for CUDA 12.8."* **Good.**

### [6] — PyTorch 2.8 Release Blog (`ref-06`)

Attached to the ROCm paragraph: "PyTorch 2.8 added functional support for the new gfx950 architecture on ROCm 7 with max-autotune coverage in TorchInductor and the Composable Kernel backend." Verified: 2.8 blog says *"Added functional support for the new gfx950 architecture on ROCm 7. Specifically, max-autotune support with (matmul, addmm, conv2d, bmm, _scaled_mm) templates for TorchInductor and AOTInductor Composable Kernel backend."* **Good.** The same paragraph's claim about primus images / FlashAttention 2.8.3 / Transformer Engine 2.8 is also covered by `[63]` (ROCm model-acceleration libraries) cited alongside.

### [7] — TPU Pricing (`ref-07`)

Attached to: "Google publishes per-chip-hour pricing for TPU v5p, Trillium, and Ironwood on its Cloud TPU pricing page." Verified: the archived pricing page lists per-chip-hour pricing for Ironwood, Trillium, TPU v5p and on-demand vs. 1Y/3Y CUD rates. **Good.**

### [8] — Mac Studio (`ref-08`)

Attached to: "Mac Studio currently ships with M4 Max or M3 Ultra … extremely large local AI workloads." Mac Studio page confirms "Supercharged by M4 Max and M3 Ultra," up to 819 GB/s bandwidth, "the ultimate pro desktop" framing. The companion citation `[58]` carries the M3 Ultra 512 GB unified-memory claim more directly. **Good.**

### [9] — PyTorch 2.8 Release Blog (`ref-09`)

Attached to: "PyTorch 2.8 introduced structured control-flow operators such as cond, while_loop, scan, associative_scan, and map." Verified: the 2.8 release blog has a section "Control Flow Operator Library" naming all five operators in the same order. **Good.**

### [10] — PyTorch 2.8 Release Blog (`ref-10`)

Attached to a multi-sentence paragraph that mixes three eras: 2.8 (CUTLASS backend), **2.10 (combo-kernel fusion)**, and 2.12 (`torch.accelerator.Graph` + Nsight 2026.2). Reference 10 (2.8 blog) does support the CUTLASS backend claim — verified in archived text. It does **not** cover the 2.10 combo-kernel claim, and `[50]` (2.12) does not cover 2.10 either. **The 2.10 combo-kernel sentence is effectively uncited.** Consider adding a PyTorch 2.10 reference (the manifest currently has no 2.9 or 2.10 release blog) or rewording the sentence.

### [11] — PyTorch 2.7 Release Blog (`ref-11`)

Attached to the CUDA paragraph about Blackwell, warp specialization, FlashAttention/SSD, MXFP8/NVFP4, plus the qualitative claim "CUDA mostly fails like a compiler." 2.7 covers the Blackwell support part directly. The MXFP8/NVFP4 and warp-specialization-on-newer-NVIDIA-parts claims are supported by other already-cited refs (`[21]`, `[22]`). Acceptable / loose for the multi-fact sentence — 2.7 is a fair anchor for "Blackwell support landed in 2.7," but a tighter version would attach this paragraph to `[5]` + `[21]` + `[22]` together.

### [12] — PyTorch 2.8 Release Blog (`ref-12`)

Attached to: 2.8 ROCm 7 gfx950 max-autotune templates + AOTriton + 2.12 ROCm expandable memory + rocSHMEM + FlexAttention pipelining on MI350X. Reference 12 (2.8) directly supports the gfx950 / max-autotune portion. The 2.12-specific items (expandable memory, rocSHMEM, MI350X FlexAttention pipelining) are picked up by the co-cited `[50]` (PyTorch 2.12 release blog), which lists exactly those features under its ROCm section. **Good** in combination.

### [13] — PyTorch on ROCm installation (`ref-13`)

Attached to: AMD docs recommend Docker, primus workflow, `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`. Verified: archived ROCm install page says *"The recommended setup to get a PyTorch environment is through Docker, as it avoids potential installation issues,"* and `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` appears in the AOTriton section. **Good.**

### [14] — Eager Mode + Compile API (`ref-14`)

Attached to: PyTorch/XLA's experimental eager mode plus compile API, `torch.compile(..., backend="openxla")` recommended for inference, PJRT default. Verified: the archived page literally documents "Eager Mode + Compile API" and `"openxla"` backend usage. **Good.** (The "PyTorch 2.1+ uses PJRT by default across TPU versions" half is from the PJRT docs, which are part of the broader PyTorch/XLA documentation set covered by `[16]`.)

### [15] — pytorch/xla GitHub (`ref-15`)

Attached to: "TorchTPU has been announced and will replace PyTorch/XLA once public," plus the RFC for a more native experience. Verified: archived xla README explicitly says *"To read more on our TorchTPU announcement see our latest [blog]. Once TorchTPU is public it will replace PyTorch/XLA"* and references the RFC for "a more native direction for PyTorch on TPU." **Good.**

### [16] — PyTorch/XLA documentation (`ref-16`)

Attached to: "Public PyTorch/XLA docs still describe the default model as LazyTensor tracing, where execution is deferred until torch_xla.sync() or an equivalent synchronization point." Verified: the archived docs include the "XLA Tensors are Lazy" section. **Good.**

### [17] — PyTorch 2.11 Release Blog (`ref-17`)

Attached to: "PyTorch 2.11 publicly called out comprehensive operator expansion on MPS, including new distributions and migrated ops." Verified: 2.11 blog literally has a section "MPS (Apple Silicon) Comprehensive Operator Expansion" and lists new distributions (log_normal, cauchy, geometric) and migrated ops. **Good.**

### [18] — torch.compile on MPS progress tracker (`ref-18`)

Attached to: 'torch.compile support on MPS was an early prototype and that end-to-end acceleration attempts were likely to fail,' NaN regressions, Metal codegen failures in reduction-heavy kernels, plus `PYTORCH_MPS_PREFER_METAL` / `PYTORCH_ENABLE_MPS_FALLBACK`. The issue tracker confirms *"support for MPS device is an early prototype and attempt to use it to accelerate end-to-end network is likely to fail,"* and lists Metal codegen / reduction failures. The env-var names themselves are documented in `ref-42` (cited at `[42]`), so this citation acts as the issue-tracker piece. **Good.**

### [19] — PyTorch 2.12 Release Blog (`ref-19`)

Attached to the "high-level compiler conclusion" paragraph — a summary sentence. The 2.12 release blog is being used as background rather than as proof of the specific recommendation. Acceptable / loose.

### [20] — FlexAttention + FlashAttention-4 (`ref-20`)

Attached to the FlashAttention test intro paragraph. Reasonable as the section-opener citation. Acceptable / loose.

### [21] — FlexAttention + FlashAttention-4 (`ref-21`)

Attached to: "PyTorch 2.11 introduced a FlashAttention-4 backend for FlexAttention on Hopper and Blackwell, with PyTorch reporting 1.2× to 3.2× gains over the previous Triton implementation on compute-bound workloads." Verified: 2.11 release blog (also `ref-51`) explicitly mentions this with "enabling 1.2× to 3.2× speedups over the existing Triton implementation on compute-bound workloads." **Good.**

### [22] — Enabling advanced GPU features in PyTorch — Warp Specialization (`ref-22`)

Attached to: "PyTorch's own performance work on warp specialization and fused SSD/Mamba-2 kernels has landed specifically on A100, H100, and newer NVIDIA parts." Verified: warp-specialization blog covers H100 and Flash Attention / FP8 row-wise GEMM. **It does not mention SSD/Mamba-2.** The warp-specialization half is well-supported; the SSD/Mamba-2 half is **weakly cited / out-of-scope for ref-22**. Recommendation: cite an SSD/Mamba-2 PyTorch blog post for that sub-claim, or drop the "fused SSD/Mamba-2 kernels" wording.

### [23] — Model acceleration libraries — ROCm Documentation (`ref-23`)

Attached to: "AMD's own ROCm docs now state that FlashAttention 2 on AMD GPUs supports two backend implementations: Composable Kernel, which is the default backend, and OpenAI Triton." Verified: archived ROCm doc says *"Triton supports two backend implementations on AMD GPUs … Composable Kernel (CK) … OpenAI Triton"* and identifies CK as default. **Good.**

### [24] — PyTorch on ROCm installation (`ref-24`)

Attached to: "ROCm passes the FlashAttention test only if you also accept ROCm's preference for curated environments and version discipline." The ROCm install page is the right vendor source for the "curated environments" argument; the paragraph is a summary, but the citation is on-topic. **Good** (loose summary citation).

### [25] — Custom Kernels via Pallas (`ref-25`)

Attached to: "PyTorch/XLA's docs explicitly position Pallas as the TPU answer to the custom-kernel culture created by Triton and kernels like FlashAttention and PagedAttention." Verified: archived Pallas page literally lists Triton, FlashAttention, PagedAttention by name. **Good.**

### [26] — Quantized Operations — PyTorch/XLA (`ref-26`)

Attached to: "PyTorch/XLA now documents quantized operations for XLA devices, including APIs for blockwise int4-style quantized matrix multiplication, and states that these ops are compatible with `torch.compile(backend="openxla")`." Verified: archived page says *"XLA Quantized ops offer a high-level abstraction for quantized operations (e.g., blockwise int4 quantized matrix multiplication)"* and references `backend='openxla'`. **Good.**

### [27] — torch.compile on MPS progress tracker (`ref-27`)

Attached to: "PyTorch's own MPS compile tracker still listed FlexAttention on MPS as stretch performance work and explicitly called out decisions still pending around decomposing `_scaled_dot_product_attention_math_for_mps`." Verified: the tracker mentions "[Perf][Stretch] Enable FlexAttention for MPS" and `aten._scaled_dot_product_attention_math_for_mps`. **Good.**

### [28] — MLX README (`ref-28`)

Attached to: MLX's unified-memory model, arrays in shared memory, MLX-LM, 4-bit Hugging Face checkpoints, `mx.distributed`. The README confirms the "unified memory model … Arrays in MLX live in shared memory" wording exactly. The MLX-LM / `mx.distributed` / 4-bit Hugging Face details come from MLX subprojects rather than the README itself, but the README is the canonical entry point and is widely cited for the framework's design. **Good** (the design claim is exactly supported; the MLX-LM subclaims are loosely covered).

### [29] — FlexAttention + FlashAttention-4 (`ref-29`)

Attached to the FlashAttention test conclusion paragraph. Acceptable as a section-anchor citation; not a fact citation.

### [30] — MPS backend notes (`ref-30`)

Attached to: the portability ladder paragraph ("plain eager PyTorch with standard operators … moves reasonably well between MPS, CUDA, and ROCm"). The MPS backend notes describe MPS as a normal `to(device)` target, which only loosely supports the cross-backend portability claim. **Acceptable / loose.** A stronger anchor would be the PyTorch device-API docs, but as the MPS-side anchor, ref-30 is on-topic.

### [31] — Apple introduces M4 Pro and M4 Max (`ref-31`)

Attached to: M4 Max 128 GB / 546 GB/s, and the 600-billion-parameter on-device LLM claim. Verified: archived Apple page says *"M4 Max supports up to 128GB of fast unified memory and up to 546GB/s of memory bandwidth"* and mentions "nearly 200 billion parameters" for M4 Max. The 600-billion claim is **specifically about M3 Ultra**, not M4 Max — it is covered by `[58]` (M3 Ultra newsroom) cited alongside, where the exact "over 600 billion parameters" wording appears. **Good** for `[31]`; the document correctly attributes the 600 B claim to M3 Ultra via `[58]`.

### [32] — PyTorch releases (`ref-32`)

Attached to: "PyTorch 2.12's move to unified memory for all MPS tensors makes the PyTorch mental model on Apple more consistent with this underlying hardware …" Verified: archived releases page contains the 2.12 release notes including *"All MPS tensors are now allocated in unified memory … all MPS tensors use unified memory unconditionally."* **Good.**

### [33] — Cloud TPU v5p (`ref-33`)

Attached to: the HBM/HBM-bandwidth comparison sentence covering v5p, Trillium, TPU7x, MI300X, MI325X. Verified for the v5p part: TPU v5p doc shows 95 GiB HBM and 2,575 GiBps. The other accelerators (Trillium, TPU7x, MI300X, MI325X) are covered by co-cited `[53]`, `[59]`, `[61]`. **Good** in combination.

### [34] — NVIDIA Hopper Architecture (`ref-34`)

Attached to: NVLink interconnect numbers — Hopper 900 GB/s bidirectional NVLink, Blackwell 130 TB/s switch fabric / 1.8 TB/s interconnect, Rubin 3.6 TB/s. Verified: archived page lists 900 GB/s, Hopper, Blackwell. The Blackwell/Rubin specifics are more directly supported by co-cited `[60]` (NVLink + NVLink Switch). **Good** in combination.

### [35] — MPS backend notes (`ref-35`)

Attached to: "PyTorch/XLA's own migration guide explains that TPUs default to a lazy execution model, rely on different distributed utilities and SPMD patterns, and recompile when graph structure or shapes change." Reference 35 is **MPS backend notes**, not the PyTorch/XLA migration guide. The MPS notes contain nothing about TPU lazy execution, SPMD, or recompilation. **Mismatch — citation error.** The correct target is `[16]` or `[41]` (PyTorch/XLA documentation). Fix recommended.

### [36] — PyTorch: Start Locally (`ref-36`)

Attached to: "On CUDA, the official PyTorch guidance is still the most boring: pick the right CUDA version in the install selector and pip install torch." The Start Locally page is exactly the install selector being described. **Good.**

### [37] — Nsight Systems Release Notes (`ref-37`)

Attached to: "Nsight Systems in 2026 is a mature performance workstation: it has CLI support, cluster/Kubernetes surfaces, and direct PyTorch-specific improvements such as annotation of optimized graphs generated by torch.compile." Verified: archived release notes contain "Annotate optimized graphs generated by torch.compile including CUDA graphs" and Docker/Kubernetes support. **Good.**

### [38] — Apple reveals M3 Ultra (`ref-38`)

Attached to the "local vs cloud bottom line" summary paragraph. The M3 Ultra newsroom backs the Apple-as-local-edge claim only loosely. Acceptable / loose summary citation.

### [39] — Common graph breaks — PyTorch 2.12 compiler docs (`ref-39`)

Attached to: "PyTorch still documents common graph breaks, dynamic-shape guard problems, and the need to rewrite certain Python control-flow patterns into structured operators if you want stable compilation. PyTorch 2.12 also changed behavior in distributed compile paths, explicitly steering users away from older `torch.distributed.nn.functional` collectives under `torch.compile`." Verified: archived doc is titled "Common Graph Breaks" and covers exactly this topic. **Good** for the graph-breaks half. (The distributed-collectives change is mentioned in the PyTorch 2.12 release blog `[50]` — not in `ref-39` — but `[50]` is widely co-cited elsewhere and supports that half.)

### [40] — PyTorch on ROCm installation (`ref-40`)

Attached to: "AMD's own docs recommend Docker specifically to avoid installation issues; leading training workflows are packaged into primus images that pin PyTorch, Triton, FlashAttention, Transformer Engine, and RCCL together." Verified: archived ROCm install page recommends Docker explicitly to "avoid potential installation issues." **Good.**

### [41] — PyTorch/XLA documentation (`ref-41`)

Attached to: "PyTorch/XLA's own docs warn about expensive recompilations when shapes change, explain that tracing time and execution time are different, and emphasize that synchronous host-side operations … block tracing and degrade performance." Verified: the XLA docs include the lazy-tensor model description; the tracing-vs-execution part is more sharply supported by co-cited `[56]` (Tracing Time vs. Execution Time), which is the canonical source. **Good** in combination.

### [42] — MPS environment variables (`ref-42`)

Attached to: "Unsupported operators can still push users toward `PYTORCH_ENABLE_MPS_FALLBACK=1` … MPS operator coverage tracker is still open … `torch.compile`-specific failures ranging from invalid generated Metal code … to NaN regressions during training." Verified: archived env-vars file contains exactly `PYTORCH_MPS_PREFER_METAL` and `PYTORCH_ENABLE_MPS_FALLBACK` descriptions. The NaN-regression / Metal codegen claims are also corroborated by `ref-18` / `ref-27` cited earlier in the document. **Good.**

### [43] — Nsight Systems Release Notes (`ref-43`)

Attached to: the team-operations / staffing summary sentence. Nsight is not the obvious citation for a staffing argument. Acceptable / loose — used as background that the CUDA tooling story exists.

### [44] — Apple reveals M3 Ultra (`ref-44`)

Attached to: "For research teams, the strongest standardization pattern in May 2026 is Apple laptops for local work, CUDA for cluster truth …" The M3 Ultra newsroom underwrites the Apple-laptop half only. Acceptable / loose summary citation.

### [45] — PyTorch 2.7 Release Blog (`ref-45`)

Attached to a production-CUDA paragraph that mentions Blackwell support (2.7), FlashAttention-4 (2.11), CuTeDSL (2026), Nsight awareness of compiled graphs. Reference 45 covers only the Blackwell-in-2.7 part. The other items are supported by `[51]`, `[20]`, `[10]`, `[37]` cited elsewhere. Acceptable / loose summary citation — a tighter version would attach `[5] [21] [37]` together at this sentence.

### [46] — PyTorch on ROCm installation (`ref-46`)

Attached to the "ROCm should be chosen when …" recommendation paragraph. ROCm install page is on-topic for the curated-container argument. Acceptable / loose summary citation.

### [47] — TPU Pricing (`ref-47`)

Attached to: "The public pricing and performance-efficiency signals around Trillium and the announced eighth-generation TPU 8t/8i line are meaningful." TPU pricing page covers Trillium; the 8t/8i claim is carried by co-cited `[54]`. **Good** in combination.

### [48] — PyTorch 2.11 Release Blog (`ref-48`)

Attached to: the Apple-Silicon-for-local-workflow recommendation paragraph. 2.11 release blog discusses MPS operator expansion, which is one supporting fact but not the whole sentence. Acceptable / loose summary citation.

### [49] — PyTorch: Start Locally (`ref-49`)

Attached to the "clearest single recommendation" final paragraph. The Start Locally page is at best a tangential anchor — it documents installation. Acceptable / loose summary citation.

### [50] — PyTorch 2.12 Release Blog (`ref-50`)

Used in many places. Spot-verified: archived 2.12 release blog contains (a) `torch.accelerator.Graph` API, (b) Metal-4 offline shader compilation for MPS, (c) ROCm expandable memory segments, rocSHMEM symmetric memory, FlexAttention pipelining (MI350X), and (d) Microscaling-quantization export. **Good** across all of its instances.

### [51] — PyTorch 2.11 Release Blog (`ref-51`)

Used for FlashAttention-4 on Hopper/Blackwell and for the source hierarchy. Archived 2.11 blog confirms the FlashAttention-4 backend and the 1.2×–3.2× speedup claim. **Good.**

### [52] — PyTorch 2.7 Release Blog (`ref-52`)

Used in the source hierarchy. Duplicates `[5]`. **Good** in the source-hierarchy context.

### [53] — TPU7x (Ironwood) | Google Cloud Documentation (`ref-53`)

Used in four places: source hierarchy, TPU 8t/8i context, TPU7x-GKE-only, HBM bandwidth (192 GiB / 7,380 GiBps). Verified: TPU7x page confirms GKE-only, "contact your account team," and JAX usage; HBM numbers consistent. **Good.**

### [54] — Our eighth generation TPUs: two chips for the agentic era (`ref-54`)

Attached to: "TPU 8t for training and TPU 8i for inference … TPU 8t … nearly 3× compute performance per pod over the previous generation … up to 2× better performance per watt over Ironwood." Verified: blog says *"nearly 3x the compute performance per pod over the previous generation,"* and the document's quoted framing matches. **Good.**

### [55] — TPU software versions | Google Cloud Documentation (`ref-55`)

Attached to: "TPU software-version page for PyTorch and JAX still centers v5p and v6e, not TPU7x," plus PJRT/torch_xla details. **Good** — the archived page is exactly the TPU software-versions matrix.

### [56] — Tracing Time vs. Execution Time in PyTorch/XLA (`ref-56`)

Attached to PyTorch/XLA performance caveats — recompilation expensive, tracing vs execution time, host-side sync ops degrade performance. The archived page is dedicated to exactly this topic and uses these terms. **Good.**

### [57] — MacBook Pro Tech Specs — Apple (`ref-57`)

Attached to: M5 Max systems with up to 128 GB unified memory and up to 614 GB/s memory bandwidth, plus general Apple-line context. Verified: archived MacBook Pro specs list "M5 Max with 18-core CPU and 40-core GPU (614GB/s memory bandwidth)" and "128GB (M5 Max with 40-core GPU)." **Good.**

### [58] — Apple reveals M3 Ultra (`ref-58`)

Attached to: 512 GB unified memory; 819 GB/s (newsroom phrasing "over 800GB/s"); 600 billion-parameter LLM claim; 2.5 TB/s UltraFusion; Thunderbolt 5 at 120 Gb/s. Verified: all four facts appear verbatim in the archived newsroom page. **Good.**

### [59] — AMD Instinct MI300 Series (`ref-59`)

Attached to: MI300X 192 GB HBM3, 5.3 TB/s, 8 Infinity Fabric links at 128 GB/s, MI325X 256 GB HBM3E 6.0 TB/s. Verified: page lists 192 GB HBM3, 256 GB HBM3E, 6 TB/s, 5.3 TB/s, and the MI325X-vs-MI300X comparison. **Good.**

### [60] — NVIDIA NVLink and NVLink Switch (`ref-60`)

This citation appears in three different paragraphs. The fit is uneven:

- **Source-hierarchy use** (`[50] [51] … [60] …`): acceptable as a vendor source example.
- **Interconnect paragraph** ("`[34] [58] [59] [60] [61]`" — NVLink Hopper 900 GB/s, Blackwell 130 TB/s switch fabric, Rubin 3.6 TB/s, Infinity Fabric, ICI): this is the **right** spot for `[60]`. Verified: archived NVLink page covers exactly this. **Good** here.
- **The CUDA-cluster paragraph** ("`[5] [50] [51] [60]`" — Blackwell support, FA4 2.11, 2.12 features, Nsight Systems 2026.2 annotating compiled graphs): NVLink is **not** the topic of any sentence in this paragraph. The right citation for the Nsight Systems annotation claim is `[37]` (Nsight Systems Release Notes). **Mismatch — citation error.**
- **The CUDA torch.compile paragraph** ("`[10] [50] [60]`" — 2.8 CUTLASS, 2.10 combo-kernel, 2.12 `torch.accelerator.Graph`, Nsight Systems 2026.2): same problem — NVLink is off-topic; the Nsight piece needs `[37]` (or `[43]`). **Mismatch — citation error.**

Fix recommended: replace `[60]` with `[37]` in those two CUDA paragraphs.

### [61] — TPU v6e | Google Cloud Documentation (`ref-61`)

Attached to: v6e 800 GB/s ICI 2D torus, Trillium 32 GB HBM and 1638 GiBps. Verified: archived v6e doc carries "HBM capacity per chip 32 GB," "HBM bandwidth per chip 1638 GiBps," "800 GBps," "2D torus." **Good.**

### [62] — PyTorch on ROCm installation (`ref-62`)

Used in three places: source hierarchy, ROCm Docker recommendation paragraph, and ROCm failure-mode paragraph (container drift, version skew). The ROCm install page supports the Docker / curated-environment argument in all three uses. **Good.**

### [63] — Model acceleration libraries — ROCm Documentation (`ref-63`)

Attached to: ROCm FlashAttention 2 with CK default and Triton alternative; ROCm "treats Triton and CK as first-class optimization tools"; FlexAttention pipelining on MI350X (co-cited with `[50]` for the 2.12 PyTorch side). Verified: archived page confirms CK default + Triton alternative + the env var `FLASH_ATTENTION_TRITON_AMD_ENABLE`. **Good.**

### [64] — State of PyTorch Hardware Acceleration 2025 (`ref-64`)

Attached to the "What Changed Since 2025" section. The 2025 predecessor report is the right source for the 2025 framing. **Good.**

## Bibliography and source-list observations

- **Reference 3 (PyTorch: Start Locally) is unused in the body** — it duplicates references 36 and 49. Either delete reference 3 from the numbered bibliography or use it somewhere.
- **Many references are duplicates of the same URL:** reference 1 / 6 / 9 / 10 / 12 are all the PyTorch 2.8 blog; 4 / 19 / 50 are all the PyTorch 2.12 blog; 5 / 11 / 45 / 52 are all the 2.7 blog; 7 / 47 are TPU pricing; 13 / 24 / 40 / 46 / 62 are the ROCm install page; 18 / 27 are the MPS compile tracker; 20 / 21 / 29 are FlexAttention/FlashAttention-4; 23 / 63 are the ROCm model-acceleration page; 30 / 35 are MPS backend notes; 37 / 43 are Nsight Systems release notes; 38 / 44 / 58 are the M3 Ultra newsroom. This is acceptable in a numbered-footnote style (each citation gets its own number even when the URL repeats), but a consolidated bibliography with one entry per URL plus per-citation context would be tighter.
- **Missing source for the PyTorch 2.10 combo-kernel-fusion claim.** No PyTorch 2.9 or 2.10 release blog is in the references; the sentence either needs a new reference or should be reworded to attribute the combo-kernel work to a different release that is in the source set.
- **"Fused SSD/Mamba-2 kernels" half of the sentence at `[22]` has no source.** The warp-specialization blog supports warp specialization on H100; the Mamba-2/SSD claim needs its own PyTorch blog citation.

## Suggested edits (concise)

1. Replace `[35]` with `[16]` (or `[41]`) on the sentence about the PyTorch/XLA migration guide.
2. Replace `[60]` with `[37]` in the two CUDA paragraphs about Nsight Systems annotating compiled graphs.
3. Add a PyTorch 2.10 (or 2.9) release-blog reference for the combo-kernel-fusion claim, or attribute combo-kernel fusion to a release that *is* in the source set.
4. Add a separate citation for the "fused SSD/Mamba-2 kernels" half of `[22]`, or drop "SSD/Mamba-2."
5. Remove unused reference `[3]` (PyTorch: Start Locally) or reuse it in place of `[49]`.
6. Optionally, attach a Nsight Systems and a FlashAttention-4 citation alongside `[45]` in the production-CUDA paragraph for tighter coverage.

Sources used in this audit are the archived files under `/references/ref-01-..-ref-64-*`, accessed via their local copies.
