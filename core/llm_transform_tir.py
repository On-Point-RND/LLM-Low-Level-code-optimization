import os
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


def _run_summary_and_update(step_log, steps_summary, summary_agent, step_n, code, correctness, profile_text, error, latency_ms, on_step_save):
    msg = summary_agent.build_user_message(
        step_n, steps_summary, step_log["recommendations"], code,
        correctness, profile_text, error, latency_ms
    )
    step_summary = summary_agent.call(msg)
    step_log["step_summary"] = step_summary
    new_summary = f"{steps_summary}\n\n{step_summary}".strip() if steps_summary else step_summary
    if on_step_save:
        on_step_save(step_log)
    return new_summary


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
):
    """
    Multi-step LLM TIR optimization loop.
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
        analysis_msg = analysis_agent.build_user_message(tir_script, prev_result)
        recommendations = analysis_agent.call(analysis_msg)
        step_log["analysis_user_message"] = analysis_msg
        step_log["recommendations"] = recommendations
        if on_step_save:
            on_step_save(step_log)

        print(f"      [LLM TIR step {step_n}/{max_steps}] Code agent...", flush=True)
        user_prompt = code_agent.build_user_prompt(tir_script, prev_result, recommendations)
        step_log["user_prompt"] = user_prompt
        if on_step_save:
            on_step_save(step_log)

        llm_result = code_agent.call(user_prompt)
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

        print(f"      [step {step_n}] Parsing TIR...", flush=True)
        candidate_mod, tir_err = _try_parse_tir(code)
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
