import os
import json
import time
from concurrent.futures import ThreadPoolExecutor

import tvm

from .agents import TirCodeAgent, TirAnalysisAgent, TirSummaryAgent
from .agents.base import build_agent_config
from .agents.tir_code import parse_tir_response


def _try_parse_tir(code: str):
    """Returns (IRModule, error_msg). Captures C-level stderr (TVM diagnostics) via fd redirect."""
    r_fd, w_fd = os.pipe()
    saved_stderr = os.dup(2)
    os.dup2(w_fd, 2)
    os.close(w_fd)
    # Strip top-level import lines that some LLM outputs may include
    # (e.g. `from tvm.script import ir as I, relax as R, tir as T`).
    # TVM's from_source does not implement a visitor for ImportFrom, so we
    # defensively drop any bare `import` / `from ... import ...` lines here.
    filtered_lines = []
    for line in code.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        filtered_lines.append(line)
    code = "\n".join(filtered_lines)

    parse_err = None
    parsed = None
    try:
        parsed = tvm.script.from_source(code)
    except Exception as e:
        parse_err = e
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        diag_chunks = []
        try:
            while True:
                chunk = os.read(r_fd, 4096)
                if not chunk:
                    break
                diag_chunks.append(chunk.decode("utf-8", errors="replace"))
        finally:
            os.close(r_fd)
    diag = "".join(diag_chunks).strip()
    if parse_err is not None:
        if diag and ("error:" in diag or "Error" in diag):
            return None, diag
        return None, str(parse_err)
    if isinstance(parsed, tvm.ir.IRModule):
        return parsed, None
    return None, f"Parsed result is not an IRModule, got {type(parsed)}"


def _inject_metadata_from_parent(parent_mod, code: str) -> str:
    """
    If LLM output references metadata[...] but does not define a top-level
    `metadata = {...}` block, re-attach the original metadata block taken
    from parent_mod.script(show_meta=True).
    """
    try:
        full_script = parent_mod.script(show_meta=True)
    except Exception:
        return code
    lines = full_script.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("metadata ="):
            start = idx
            break
    if start is None:
        return code
    # Extract only the metadata block, not the whole module that follows.
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        stripped = lines[idx].lstrip()
        if stripped.startswith("@I.ir_module") or stripped.startswith("@R.function") or stripped.startswith("class "):
            end = idx
            break
    metadata_block = "\n".join(lines[start:end]).strip()
    if not metadata_block:
        return code
    # Drop any LLM-invented top-level metadata or comments and keep only the
    # actual Relax/TIR module starting from the first decorator/class.
    code_lines = code.splitlines()
    start_idx = 0
    for idx, line in enumerate(code_lines):
        stripped = line.lstrip()
        if stripped.startswith("@I.ir_module") or stripped.startswith("@R.function") or stripped.startswith("class "):
            start_idx = idx
            break
    trimmed_code = "\n".join(code_lines[start_idx:]).lstrip() if start_idx > 0 else code.lstrip()
    # Place metadata BEFORE the module so that `metadata` is in scope for any
    # Relax code that references it (R.call_tir(..., metadata["..."], ...)).
    return metadata_block + "\n\n" + trimmed_code


def _merge_tir_from_llm(parent_mod, llm_mod):
    """
    Return a new IRModule where all TIR PrimFuncs from llm_mod replace
    the corresponding PrimFuncs in parent_mod. Relax functions and
    metadata from parent_mod are preserved.
    """
    new_mod = parent_mod.clone()
    for gv, func in llm_mod.functions.items():
        # Only replace TIR PrimFuncs; keep Relax/@R.function as-is.
        if isinstance(func, tvm.tir.PrimFunc):
            if gv in new_mod.functions:
                new_mod.update_func(gv, func)
    return new_mod


def _run_summary_and_update(step_log, steps_summary, summary_agent, step_n, code, correctness, profile_text, error, latency_ms, on_step_save):
    msg = summary_agent.build_user_message(
        step_n, steps_summary, step_log["recommendations"], code,
        correctness, profile_text, error, latency_ms
    )
    # Summary agent is best-effort: failures (e.g. 500 from OpenRouter) should not
    # abort the whole TIR optimization loop. Try once, and on error record a
    # textual note instead of raising.
    try:
        step_summary = summary_agent.call(msg)
    except Exception as e:
        step_summary = f"[summary agent failed: {e}]"
    step_log["step_summary"] = step_summary
    new_summary = f"{steps_summary}\n\n{step_summary}".strip() if steps_summary else step_summary
    if on_step_save:
        on_step_save(step_log)
    return new_summary


def llm_transform_tir_feedback(
    mod,
    config: dict,
    inputs_tvm,
    dev,
    torch_model,
    inputs_torch,
    max_steps: int = 3,
    on_step_save=None,
    baseline_mean_ms=None,
    reference_python_code: str = "",
):
    """
    Multi-step LLM TIR optimization loop (analysis -> code -> summary per step).
    Returns (best_mod, steps_log) where steps_log is a list of per-step dicts.
    """
    from .profiler import profile_tir
    from .validator import validate_correctness_safe

    code_agent = TirCodeAgent(build_agent_config(config, "tir_code"))
    analysis_agent = TirAnalysisAgent(build_agent_config(config, "tir_analysis"))
    summary_agent = TirSummaryAgent(build_agent_config(config, "tir_summary"))

    tir_script = mod.script()
    best_mod = None
    best_latency = float("inf")
    best_code = None
    best_profile_text = None
    steps_log = []
    prev_result = None
    steps_summary = ""

    for step in range(max_steps):
        step_n = step + 1
        step_log = {"step": step_n}

        print(f"      [LLM TIR step {step_n}/{max_steps}] Analysis agent...", flush=True)
        analysis_msg = analysis_agent.build_user_message(tir_script, prev_result, reference_python_code)
        # Analysis agent is also best-effort: API failures (e.g. 500) should not
        # abort the whole loop. On error, record a textual note and continue.
        try:
            recommendations = analysis_agent.call(analysis_msg)
        except Exception as e:
            recommendations = f"[analysis agent failed: {e}]"
        step_log["analysis_user_message"] = analysis_msg
        step_log["recommendations"] = recommendations
        if on_step_save:
            on_step_save(step_log)

        print(f"      [LLM TIR step {step_n}/{max_steps}] Code agent...", flush=True)
        user_prompt = code_agent.build_user_prompt(tir_script, prev_result, recommendations, reference_python_code)
        step_log["user_prompt"] = user_prompt
        if on_step_save:
            on_step_save(step_log)

        # Code agent is critical for this step, but API failures should not crash
        # the whole benchmark run. On error, mark this step as failed and move on.
        try:
            llm_result = code_agent.call(user_prompt)
        except Exception as e:
            err = f"Code agent failed: {e}"
            print(f"      [step {step_n}] {err}")
            step_log["error"] = err
            step_log["is_best"] = False
            steps_log.append(step_log)
            # Run summary agent with the failure info (no new code).
            print(f"      [LLM TIR step {step_n}] Summary agent...", flush=True)
            steps_summary = _run_summary_and_update(
                step_log, steps_summary, summary_agent, step_n,
                tir_script, None, None, err, None, on_step_save
            )
            prev_result = {
                "error": err,
                "failed_code": None,
                "latency_ms": None,
                "profile_text": None,
                "best_code": best_code,
                "best_profile_text": best_profile_text,
                "best_latency_ms": best_latency if best_mod else None,
                "steps_summary": steps_summary,
            }
            continue

        step_log["llm_response"] = llm_result["llm_response"]
        step_log["model"] = llm_result["model"]
        step_log["system_prompt"] = llm_result["system_prompt"]
        step_log["transformed_content"] = llm_result["transformed_content"]
        step_log["request_data"] = llm_result["request_data"]
        if on_step_save:
            on_step_save(step_log)

        parsed_response, parse_err = parse_tir_response(llm_result["transformed_content"])
        if parse_err:
            print(f"      [step {step_n}] Parse error: {parse_err}")
            step_log["error"] = parse_err
            step_log["is_best"] = False
            steps_log.append(step_log)
            print(f"      [LLM TIR step {step_n}] Summary agent...", flush=True)
            steps_summary = _run_summary_and_update(
                step_log, steps_summary, summary_agent, step_n,
                llm_result.get("transformed_content", ""), None, None, parse_err, None, on_step_save
            )
            prev_result = {"error": parse_err, "failed_code": llm_result.get("transformed_content"), "latency_ms": None,
                          "profile_text": None, "best_code": best_code, "best_profile_text": best_profile_text,
                          "best_latency_ms": best_latency if best_mod else None, "steps_summary": steps_summary}
            continue

        code = parsed_response.code
        step_log["structured_output"] = parsed_response.model_dump()
        if on_step_save:
            on_step_save(step_log)
        # Auto-restore original metadata block if LLM output still references metadata[...]
        # but dropped the definition. This mirrors the MCTS mutation executor logic.
        code = _inject_metadata_from_parent(mod, code)
        # Save the exact code that will be passed to tvm.script.from_source,
        # so that line numbers in TVM errors (e.g. line 504) match a concrete file.
        step_log["code_for_parse"] = code
        if on_step_save:
            on_step_save(step_log)

        print(f"      [step {step_n}] Parsing TIR...", flush=True)
        llm_mod, tir_err = _try_parse_tir(code)
        if tir_err:
            print(f"      [step {step_n}] TIR parse error: {tir_err}")
            step_log["error"] = tir_err
            step_log["is_best"] = False
            steps_log.append(step_log)
            print(f"      [LLM TIR step {step_n}] Summary agent...", flush=True)
            steps_summary = _run_summary_and_update(
                step_log, steps_summary, summary_agent, step_n, code, None, None, tir_err, None, on_step_save
            )
            prev_result = {"error": tir_err, "failed_code": code, "latency_ms": None, "profile_text": None,
                          "best_code": best_code, "best_profile_text": best_profile_text,
                          "best_latency_ms": best_latency if best_mod else None, "steps_summary": steps_summary}
            continue

        # Merge only TIR PrimFuncs from LLM module into the original Relax IRModule.
        candidate_mod = _merge_tir_from_llm(mod, llm_mod)

        print(f"      [step {step_n}] Checking correctness...", flush=True)
        correctness = validate_correctness_safe(
            torch_model, candidate_mod, inputs_torch, inputs_tvm, dev
        )
        step_log["correctness"] = correctness
        if on_step_save:
            on_step_save(step_log)
        if not correctness["is_correct"]:
            err_msg = f"Build failed: {correctness.get('build_error')}" if correctness.get("build_error") else (
                f"Correctness check failed: max_abs_diff={correctness['max_abs_diff']:.2e}, "
                f"max_rel_diff={correctness['max_rel_diff']:.2e}"
            )
            print(f"      [step {step_n}] {err_msg}")
            step_log["error"] = err_msg
            step_log["is_best"] = False
            steps_log.append(step_log)
            print(f"      [LLM TIR step {step_n}] Summary agent...", flush=True)
            steps_summary = _run_summary_and_update(
                step_log, steps_summary, summary_agent, step_n, code, correctness, None, err_msg, None, on_step_save
            )
            prev_result = {"error": err_msg, "failed_code": code, "latency_ms": None, "profile_text": None,
                          "best_code": best_code, "best_profile_text": best_profile_text,
                          "best_latency_ms": best_latency if best_mod else None, "steps_summary": steps_summary}
            continue

        print(f"      [step {step_n}] Profiling...", flush=True)
        prof_cfg = config.get("profile", {})
        prof_result = profile_tir(
            candidate_mod, inputs_tvm, dev,
            warmup_iters=prof_cfg.get("warmup_iters", 3),
            number=prof_cfg.get("number", 10),
            repeat=prof_cfg.get("repeat", 5),
            baseline_mean_ms=baseline_mean_ms,
            baseline_multiplier=prof_cfg.get("baseline_multiplier", 5),
        )
        latency_ms = prof_result["mean_ms"]
        profile_text = prof_result.get("profiler_report")
        profile_text = profile_text.table() if profile_text else None

        step_log["latency_ms"] = latency_ms
        step_log["profile_text"] = profile_text
        if prof_result.get("aborted_reason"):
            step_log["aborted_reason"] = prof_result["aborted_reason"]
        if on_step_save:
            on_step_save(step_log)
        print(f"      [step {step_n}] Latency: {latency_ms:.3f} ms", flush=True)

        is_best = latency_ms < best_latency
        step_log["is_best"] = is_best
        if is_best:
            best_mod = candidate_mod
            best_latency = latency_ms
            best_code = code
            best_profile_text = profile_text
            print(f"      [step {step_n}] New best: {latency_ms:.3f} ms")

        steps_log.append(step_log)
        print(f"      [LLM TIR step {step_n}] Summary agent...", flush=True)
        steps_summary = _run_summary_and_update(
            step_log, steps_summary, summary_agent, step_n, code, correctness, profile_text, None, latency_ms, on_step_save
        )
        prev_result = {"error": None, "latency_ms": latency_ms, "profile_text": profile_text,
                      "best_code": best_code, "best_profile_text": best_profile_text,
                      "best_latency_ms": best_latency, "steps_summary": steps_summary}

    if best_mod is None:
        print("      All LLM TIR steps failed, returning original module.")
        best_mod = mod

    return best_mod, steps_log


def llm_transform_tir_sampling(
    mod,
    config: dict,
    inputs_tvm,
    dev,
    torch_model,
    inputs_torch,
    max_steps: int = 3,
    on_step_save=None,
    baseline_mean_ms=None,
    reference_python_code: str = "",
    llm_logs_dir: str | None = None,
):
    """
    Single-round LLM TIR optimization:
      1) 5 parallel zero-shot generations
      2) sequential deterministic eval for each sample
      3) 5 parallel diagnosis calls
      4) final generation + eval

    Extra resilience:
      - if cached step_1..step_5 exist in llm_logs_dir, reuse them (do not regenerate)
      - final generation retries up to 3 attempts with 120s pauses for non-token errors
      - retry prompt after first final error uses ONLY best attempt (not all 5)

    Note: max_steps is ignored (fixed pipeline); kept for API compatibility with llm_transform_tir_iterative.
    """
    _ = max_steps
    from .profiler import profile_tir
    from .validator import validate_correctness_safe

    code_agent = TirCodeAgent(build_agent_config(config, "tir_code"))
    summary_agent = TirSummaryAgent(build_agent_config(config, "tir_summary"))

    NUM_SAMPLES = 5
    tir_script = mod.script()
    def _is_token_error(text: str) -> bool:
        t = (text or "").lower()
        return any(
            m in t for m in [
                "max_tokens",
                "context length",
                "context window",
                "prompt is too long",
                "too many tokens",
                "token",
            ]
        )

    def _load_text(path: str) -> str | None:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _load_cached_initial_steps() -> list | None:
        if not llm_logs_dir:
            return None
        out = []
        for i in range(1, NUM_SAMPLES + 1):
            step_dir = os.path.join(llm_logs_dir, f"step_{i}")
            summary_path = os.path.join(step_dir, "summary.json")
            if not os.path.exists(summary_path):
                return None
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    s = json.load(f)
            except Exception:
                return None
            sl = {"step": i, "is_best": False}
            for k in ("latency_ms", "error", "model", "aborted_reason"):
                if k in s:
                    sl[k] = s[k]
            if "correctness" in s:
                sl["correctness"] = s["correctness"]
            if "status" in s or "modification" in s:
                sl["structured_output"] = {"status": s.get("status"), "modification": s.get("modification")}
            code_parsed = _load_text(os.path.join(step_dir, "tir_code_parsed.py"))
            raw = _load_text(os.path.join(step_dir, "llm_response_raw.txt"))
            prof = _load_text(os.path.join(step_dir, "profile.txt"))
            summ = _load_text(os.path.join(step_dir, "step_summary.txt"))
            if code_parsed is not None:
                sl["code_for_parse"] = code_parsed
            if raw is not None:
                sl["transformed_content"] = raw
            if prof is not None:
                sl["profile_text"] = prof
            if summ is not None:
                sl["step_summary"] = summ
            out.append(sl)
        return out

    def _evaluate_candidate(step_n: int, step_log: dict, code: str):
        llm_mod, tir_err = _try_parse_tir(code)
        if tir_err:
            step_log["error"] = tir_err
            step_log["is_best"] = False
            return None
        candidate_mod = _merge_tir_from_llm(mod, llm_mod)
        correctness = validate_correctness_safe(torch_model, candidate_mod, inputs_torch, inputs_tvm, dev)
        step_log["correctness"] = correctness
        if not correctness.get("is_correct"):
            err_msg = (
                f"Build failed: {correctness.get('build_error')}"
                if correctness.get("build_error")
                else (
                    f"Correctness check failed: max_abs_diff={correctness['max_abs_diff']:.2e}, "
                    f"max_rel_diff={correctness['max_rel_diff']:.2e}"
                )
            )
            step_log["error"] = err_msg
            step_log["is_best"] = False
            return None
        prof_cfg = config.get("profile", {})
        prof_result = profile_tir(
            candidate_mod,
            inputs_tvm,
            dev,
            warmup_iters=prof_cfg.get("warmup_iters", 3),
            number=prof_cfg.get("number", 10),
            repeat=prof_cfg.get("repeat", 5),
            baseline_mean_ms=baseline_mean_ms,
            baseline_multiplier=prof_cfg.get("baseline_multiplier", 5),
        )
        step_log["latency_ms"] = prof_result["mean_ms"]
        pr = prof_result.get("profiler_report")
        step_log["profile_text"] = pr.table() if pr else None
        if prof_result.get("aborted_reason"):
            step_log["aborted_reason"] = prof_result["aborted_reason"]
        return candidate_mod

    generation_recommendations = (
        "Apply ONE conservative optimization transformation. Preserve @R.function and R.call_tir exactly; "
        "modify only @T.prim_func bodies. Maintain functional correctness and type annotations.\n"
        "For GPU code: prefer thread binding for the most parallel spatial axes, and keep "
        "threadIdx.x * threadIdx.y * threadIdx.z <= 1024. Ensure the T.thread_binding loops wrap the "
        "T.sblock that uses the corresponding loop vars.\n"
        "T.axis.remap: use ONLY loop iterator Vars (no T.int64 constants, no expressions like k*1024, "
        "and do not remap block vars). Do not invent metadata[...] or change R.call_tir argument lists.\n"
        "Never return unchanged code; status must be 'improved'."
    )
    user_prompt = code_agent.build_user_prompt(
        tir_script, prev_result=None, recommendations=generation_recommendations, reference_python_code=reference_python_code
    )

    steps_log = []
    initial_step_logs = _load_cached_initial_steps()
    gen_results = None

    if initial_step_logs is None:
        print(f"      [LLM TIR] Launching {NUM_SAMPLES} parallel zero-shot generations...", flush=True)

        def _safe_call_code(sample_idx: int) -> dict:
            try:
                llm_result = code_agent.call(user_prompt)
                return {"ok": True, "llm_result": llm_result}
            except Exception as e:
                return {"ok": False, "error": f"Code agent failed (sample {sample_idx}): {e}"}

        with ThreadPoolExecutor(max_workers=NUM_SAMPLES) as ex:
            futures = {i: ex.submit(_safe_call_code, i) for i in range(1, NUM_SAMPLES + 1)}
            gen_results = {i: f.result() for i, f in futures.items()}
        initial_step_logs = []
    else:
        print("      [LLM TIR] Reusing cached step_1..step_5; skip regeneration.", flush=True)
        steps_log.extend(initial_step_logs)

    best_initial_mod = None
    best_initial_step_idx = None
    best_initial_code = None
    best_initial_latency_ms = float("inf")
    best_initial_profile_text = None

    if gen_results is not None:
        for step_n in range(1, NUM_SAMPLES + 1):
            step_log = {"step": step_n, "user_prompt": user_prompt}
            r = gen_results[step_n]
            if not r["ok"]:
                step_log["error"] = r["error"]
                step_log["is_best"] = False
            else:
                llm_result = r["llm_result"]
                step_log["llm_response"] = llm_result["llm_response"]
                step_log["model"] = llm_result["model"]
                step_log["system_prompt"] = llm_result["system_prompt"]
                step_log["transformed_content"] = llm_result["transformed_content"]
                step_log["request_data"] = llm_result["request_data"]
                parsed, parse_err = parse_tir_response(llm_result["transformed_content"])
                if parse_err:
                    step_log["error"] = parse_err
                    step_log["is_best"] = False
                else:
                    code = _inject_metadata_from_parent(mod, parsed.code)
                    step_log["structured_output"] = parsed.model_dump()
                    step_log["code_for_parse"] = code
                    cand = _evaluate_candidate(step_n, step_log, code)
                    if cand is not None and step_log.get("latency_ms", float("inf")) < best_initial_latency_ms:
                        best_initial_mod = cand
                        best_initial_step_idx = step_n
                        best_initial_code = code
                        best_initial_latency_ms = step_log["latency_ms"]
                        best_initial_profile_text = step_log.get("profile_text")
                    step_log["is_best"] = False
            if on_step_save:
                on_step_save(step_log)
            steps_log.append(step_log)
            initial_step_logs.append(step_log)

    # compute best from cached path if needed
    if best_initial_code is None:
        for sl in initial_step_logs:
            if (sl.get("correctness") or {}).get("is_correct") and sl.get("latency_ms") is not None:
                if sl["latency_ms"] < best_initial_latency_ms:
                    best_initial_latency_ms = sl["latency_ms"]
                    best_initial_step_idx = sl.get("step")
                    best_initial_code = sl.get("code_for_parse") or sl.get("transformed_content")
                    best_initial_profile_text = sl.get("profile_text")
        if best_initial_code:
            lm, err = _try_parse_tir(best_initial_code)
            if err is None and lm is not None:
                best_initial_mod = _merge_tir_from_llm(mod, lm)
        else:
            best_initial_code = tir_script
            best_initial_latency_ms = float("inf")

    # diagnosis (only missing)
    missing_diag = [sl for sl in initial_step_logs if not (sl.get("step_summary") or "").strip()]
    if missing_diag:
        print(f"      [LLM TIR] Launching {NUM_SAMPLES} parallel diagnosis requests...", flush=True)
        def _safe_diagnose(step_log: dict) -> str:
            try:
                msg = summary_agent.build_user_message(
                    step_log["step"], "", None,
                    step_log.get("code_for_parse") or tir_script,
                    step_log.get("correctness"),
                    step_log.get("profile_text"),
                    step_log.get("error"),
                    step_log.get("latency_ms"),
                )
                return summary_agent.call(msg)
            except Exception as e:
                return f"[diagnosis agent failed: {e}]"
        with ThreadPoolExecutor(max_workers=NUM_SAMPLES) as ex:
            futs = {sl["step"]: ex.submit(_safe_diagnose, sl) for sl in initial_step_logs}
            for sl in initial_step_logs:
                if not (sl.get("step_summary") or "").strip():
                    sl["step_summary"] = futs[sl["step"]].result()
                    if on_step_save:
                        on_step_save(sl)

    # Build final recommendation strings
    unique_errors = []
    for sl in initial_step_logs:
        e = sl.get("error")
        if e and e not in unique_errors:
            unique_errors.append(e)
    unique_errors = unique_errors[:5]

    correct_latencies = [sl.get("latency_ms") for sl in initial_step_logs if (sl.get("correctness") or {}).get("is_correct") and sl.get("latency_ms") is not None]
    best_correct_latency = min(correct_latencies) if correct_latencies else None

    blocks = []
    for sl in initial_step_logs:
        is_correct = bool((sl.get("correctness") or {}).get("is_correct"))
        lat = sl.get("latency_ms")
        if is_correct and lat is not None and best_correct_latency is not None:
            speed = "FAST" if lat <= best_correct_latency * 1.1 else "SLOW"
            quality = "GOOD" if speed == "FAST" else "LESS_GOOD"
        elif is_correct:
            speed = "UNKNOWN_FASTNESS"
            quality = "GOOD"
        else:
            speed = "N/A"
            quality = "BAD"
        code_text = (sl.get("code_for_parse") or sl.get("transformed_content") or tir_script).strip()
        why = sl.get("error") or (f"Correctness PASSED; latency_ms={lat:.6f}" if lat is not None else "No latency")
        diag_short = ((sl.get("step_summary") or "").strip().splitlines() or [""])[0]
        blocks.append(
            f"Candidate {sl.get('step')}:\n"
            f"  Quality: {quality}\n"
            f"  Speed: {speed}\n"
            f"  Why: {why}\n"
            f"  LLM_diagnosis: {diag_short}\n"
            f"  Profile: {sl.get('profile_text') or 'N/A'}\n"
            f"  Code:\n```tvm\n{code_text}\n```"
        )
    candidates_overview = "\n\n".join(blocks).strip()

    final_reco_all = (
        "Best practices:\n"
        "- Preserve @R.function and R.call_tir exactly; modify only @T.prim_func.\n"
        "- Keep GPU threadIdx product <= 1024 and ensure thread_binding directly wraps sblocks.\n"
        "- T.axis.remap: use ONLY loop iterator Vars (no constants/expressions).\n"
        "- Make one conservative change; keep signatures/types.\n\n"
        "All 5 candidate programs with deterministic and LLM summary:\n"
        f"{candidates_overview}\n\n"
        "Errors to avoid:\n"
        + ("\n".join(f"- {e}" for e in unique_errors) if unique_errors else "- None")
    )
    final_reco_best = (
        "Best practices:\n"
        "- Preserve @R.function and R.call_tir exactly; modify only @T.prim_func.\n"
        "- Keep GPU threadIdx product <= 1024 and ensure thread_binding directly wraps sblocks.\n"
        "- T.axis.remap: use ONLY loop iterator Vars (no constants/expressions).\n"
        "- Make one conservative change; keep signatures/types.\n\n"
        f"Use ONLY best attempt (step {best_initial_step_idx}, latency_ms={best_initial_latency_ms}):\n"
        f"Profile:\n{best_initial_profile_text or 'N/A'}\n"
        f"Code:\n```tvm\n{best_initial_code or tir_script}\n```"
    )

    prev_result = {
        "error": None,
        "latency_ms": float(best_initial_latency_ms),
        "profile_text": best_initial_profile_text or "",
        "best_code": None,
        "best_latency_ms": None,
        "best_profile_text": None,
        "steps_summary": "",
    }
    final_base_code = best_initial_code or tir_script

    print("      [LLM TIR] Final code generation...", flush=True)
    final_llm_result = None
    final_user_prompt = None
    final_reco_used = final_reco_all
    final_err = None
    for attempt in range(1, 4):
        reco = final_reco_all if attempt == 1 else final_reco_best
        final_reco_used = reco
        final_user_prompt = code_agent.build_user_prompt(
            final_base_code, prev_result=prev_result, recommendations=reco, reference_python_code=reference_python_code
        )
        try:
            final_llm_result = code_agent.call(final_user_prompt)
            final_err = None
            break
        except Exception as e:
            final_err = f"Final code agent failed (attempt {attempt}/3): {e}"
            if attempt < 3 and not _is_token_error(str(e)):
                print(f"      [final] retrying in 120s due to non-token error: {e}", flush=True)
                time.sleep(120)
                continue
            break

    final_step_log = {"step": NUM_SAMPLES + 1, "user_prompt": final_user_prompt, "recommendations": final_reco_used}
    if final_err is not None or final_llm_result is None:
        final_step_log["error"] = final_err or "Final generation failed without response"
        final_step_log["is_best"] = False
        if on_step_save:
            on_step_save(final_step_log)
        steps_log.append(final_step_log)
    else:
        final_step_log["llm_response"] = final_llm_result["llm_response"]
        final_step_log["model"] = final_llm_result["model"]
        final_step_log["system_prompt"] = final_llm_result["system_prompt"]
        final_step_log["transformed_content"] = final_llm_result["transformed_content"]
        final_step_log["request_data"] = final_llm_result["request_data"]
        parsed, parse_err = parse_tir_response(final_llm_result["transformed_content"])
        if parse_err:
            final_step_log["error"] = parse_err
            final_step_log["is_best"] = False
            if on_step_save:
                on_step_save(final_step_log)
            steps_log.append(final_step_log)
        else:
            code = _inject_metadata_from_parent(mod, parsed.code)
            final_step_log["structured_output"] = parsed.model_dump()
            final_step_log["code_for_parse"] = code
            cand = _evaluate_candidate(NUM_SAMPLES + 1, final_step_log, code)
            final_step_log["is_best"] = cand is not None
            if cand is not None:
                best_initial_mod = cand
                best_initial_step_idx = NUM_SAMPLES + 1
            if on_step_save:
                on_step_save(final_step_log)
            steps_log.append(final_step_log)

    # Return best available module
    best_mod = None
    final_log = next((s for s in steps_log if s.get("step") == NUM_SAMPLES + 1), None)
    if final_log and final_log.get("is_best") and final_log.get("code_for_parse"):
        lm, err = _try_parse_tir(final_log["code_for_parse"])
        if err is None and lm is not None:
            best_mod = _merge_tir_from_llm(mod, lm)
    if best_mod is None and best_initial_mod is not None:
        best_mod = best_initial_mod
        if best_initial_step_idx is not None:
            for s in steps_log:
                if s.get("step") == best_initial_step_idx:
                    s["is_best"] = True
                    if on_step_save:
                        on_step_save(s)
                elif s.get("is_best"):
                    s["is_best"] = False
    if best_mod is None:
        best_mod = mod
    return best_mod, steps_log


def llm_transform_tir_iterative(
    mod,
    config: dict,
    inputs_tvm,
    dev,
    torch_model,
    inputs_torch,
    max_steps: int = 3,
    on_step_save=None,
    baseline_mean_ms=None,
    reference_python_code: str = "",
    llm_logs_dir: str | None = None,
    strategy: str | None = None,
):
    """
    Dispatch LLM TIR optimization by config key ``llm_tir_strategy`` (or ``strategy`` override):
    ``feedback`` -> multi-step analysis/code loop; ``sampling`` -> parallel samples + final round.
    """
    raw = strategy if strategy is not None else config.get("llm_tir_strategy", "feedback")
    s = str(raw).strip().lower()
    if s not in ("feedback", "sampling"):
        print(
            f"      [LLM TIR] Unknown llm_tir_strategy={raw!r}; using 'feedback'. "
            f"Valid: feedback, sampling.",
            flush=True,
        )
        s = "feedback"

    if s == "sampling":
        return llm_transform_tir_sampling(
            mod,
            config=config,
            inputs_tvm=inputs_tvm,
            dev=dev,
            torch_model=torch_model,
            inputs_torch=inputs_torch,
            max_steps=max_steps,
            on_step_save=on_step_save,
            baseline_mean_ms=baseline_mean_ms,
            reference_python_code=reference_python_code,
            llm_logs_dir=llm_logs_dir,
        )

    return llm_transform_tir_feedback(
        mod,
        config=config,
        inputs_tvm=inputs_tvm,
        dev=dev,
        torch_model=torch_model,
        inputs_torch=inputs_torch,
        max_steps=max_steps,
        on_step_save=on_step_save,
        baseline_mean_ms=baseline_mean_ms,
        reference_python_code=reference_python_code,
    )
