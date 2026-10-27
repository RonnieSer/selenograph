"""Evaluate painted maps independently without changing the active model."""
import importlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app import assistant_model, paths


def file_stamp(path):
    """Return file revision metadata, or None if the file is inaccessible."""
    try:
        stat = Path(path).stat()
        return stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino
    except OSError:
        return None


def report_stamp(output_root):
    """Return revision metadata for completed evaluation reports."""
    root = Path(output_root) / "evaluations"
    candidates = root.glob("evaluation-*.json")
    return tuple((str(path.relative_to(root)), file_stamp(path)) for path in sorted(candidates))


@st.cache_data(show_spinner=False, max_entries=4)
def completed_report(output_root, stamp):
    """Load the latest evaluation report for the supplied cache revision."""
    return importlib.import_module("app.evaluation").latest_report(output_root)


def percent(value):
    """Format a reported fraction as a percentage."""
    return "Not reported" if value is None else f"{value:.1%}"


st.subheader("Evaluation")
st.caption("Each map is evaluated separately. Training maps are left out for their turn; the active model stays unchanged.")
bundle = assistant_model.load()
samples = bundle.get("samples") or {}
evaluation = None
latest = None
try:
    evaluation = importlib.import_module("app.evaluation")
except ModuleNotFoundError as error:
    if error.name != "app.evaluation":
        raise
    st.caption("The evaluation engine is not available yet.")

maps = {}
folds = []
if evaluation is not None:
    maps = {name: meta for name, meta in evaluation.catalog().items()
            if meta.get("painted", bool(meta.get("label_source")))}
    try:
        latest = completed_report(str(paths.OUTPUT_DIR), report_stamp(paths.OUTPUT_DIR))
    except (OSError, ValueError, KeyError, TypeError) as error:
        st.warning(f"The latest evaluation report could not be read: {error}")
previous = latest[1].get("selected_maps", []) if latest else []
st.session_state.setdefault("evaluation_maps", [n for n in previous if n in maps])
st.session_state.evaluation_maps = [n for n in st.session_state.evaluation_maps if n in maps]


def map_label(name):
    """Return a map label with its section and verification status."""
    status = "painted, verified" if maps[name].get("verified") else "painted"
    return f"{name} · {maps[name].get('section', 'Local')} — {status}"


names = st.multiselect(
    "Maps to evaluate", list(maps), key="evaluation_maps",
    format_func=map_label,
    help="Compare predictions with Sure hand-painted cells.",
    width="stretch")
if evaluation is not None and names:
    folds = evaluation.plan(bundle, names)
ready = [fold for fold in folds if not fold.get("reason")]
can_evaluate = bool(evaluation is not None and samples and names and set(names).issubset(maps) and ready)
st.caption(f"{len(names)} maps selected · {len(ready)} ready")
if not maps:
    st.info("Paint a map to make it available for evaluation.")
if not samples:
    st.info("No training samples are available. Train the app model from labeled maps before evaluating.")
elif names and not ready:
    st.info("No evaluation is ready. See the evaluation plan for details.")
if names:
    with st.expander("Evaluation plan", expanded=not bool(ready)):
        plan_rows = []
        for fold in folds:
            map_names = ", ".join(fold.get("maps", []))
            left_out = map_names if fold.get("mode") == "leave-one-map-out" else "None"
            plan_rows.append({"Map": map_names,
                              "Training maps": len(fold.get("train_keys", [])),
                              "Left out": left_out, "Status": fold.get("reason") or "Ready"})
        st.dataframe(plan_rows, hide_index=True, width="stretch")
if st.button("Evaluate", icon=":material/fact_check:", type="primary", disabled=not can_evaluate,
             width="stretch") and can_evaluate and evaluation is not None:
    try:
        with st.spinner("Evaluating selected maps..."):
            message = st.empty()
            evaluation.run(bundle, names, progress=message.text)
    except (OSError, ValueError, RuntimeError) as error:
        st.error(f"Evaluation could not complete: {error}")
    else:
        completed_report.clear()
        st.rerun()

if latest is None:
    st.caption("No completed evaluation report yet.")
else:
    report_path, report = latest
    validation = report.get("validation", {})
    aggregate = validation.get("aggregate") or {}
    baseline = validation.get("baseline_aggregate") or {}
    st.subheader("Latest evaluation")
    st.caption(f"Run: {report.get('run_id', 'not reported')} · {report.get('created_at', 'not reported')}. "
               "These saved scores describe the run's selection, not unsaved chooser changes.")
    with st.container(horizontal=True):
        st.metric("Evaluation accuracy", percent(aggregate.get("accuracy")), border=True)
        st.metric("Evaluation macro F1", percent(aggregate.get("macro_f1")), border=True)
        st.metric("Majority baseline accuracy", percent(baseline.get("accuracy")), border=True)
        st.metric("Scored cells", aggregate.get("n", 0), border=True)
    results = validation.get("folds", [])
    if results:
        result_rows = []
        bars = []
        for fold in results:
            map_names = ", ".join(fold.get("maps", fold.get("test_tiles", [])))
            scores = fold.get("metrics") or {}
            baseline_scores = fold.get("baseline") or {}
            result_rows.append({
                "Map": map_names,
                "Training maps": len(fold.get("train_keys", fold.get("train_tiles", []))),
                "Scored cells": scores.get("n"),
                "Accuracy": scores.get("accuracy"), "Macro F1": scores.get("macro_f1"),
                "Baseline accuracy": baseline_scores.get("accuracy"),
                "Baseline macro F1": baseline_scores.get("macro_f1")})
            for method, metrics in (("Evaluation model", scores), ("Majority baseline", baseline_scores)):
                if metrics.get("macro_f1") is not None:
                    bars.append({"Map": map_names or fold["group"],
                                 "Method": method, "Macro F1": metrics["macro_f1"]})
        st.dataframe(result_rows, hide_index=True, width="stretch")
        if bars:
            st.bar_chart(pd.DataFrame(bars), x="Map", y="Macro F1", color="Method",
                         stack=False, width="stretch")
    with st.expander("Scores and coverage"):
        st.caption("Reference painting cells are coarse labels, not independent 5 m truth. "
                   "Class 3 includes crater interiors / floor, not measured permanent shadow or ice.")
        scopes = {"All selected maps": aggregate, "Majority baseline": baseline}
        for fold in results:
            map_names = ", ".join(fold.get("maps", fold.get("test_tiles", [])))
            scopes[map_names or str(fold["group"])] = fold.get("metrics") or {}
        scope = st.selectbox("Score detail", list(scopes), key="evaluation_score_detail", width="stretch")
        metrics = scopes[scope]
        if metrics.get("per_class"):
            st.dataframe([{"Class": c, **values} for c, values in metrics["per_class"].items()],
                         hide_index=True, width="stretch")
        if metrics.get("confusion_matrix") is not None:
            st.caption("Confusion matrix: reference classes in rows, predictions in columns (1, 2, 3).")
            matrix_rows = []
            for reference, counts in enumerate(metrics["confusion_matrix"], start=1):
                row = {"Reference class": str(reference)}
                for predicted, count in enumerate(counts, start=1):
                    row[f"Predicted {predicted}"] = int(count)
                matrix_rows.append(row)
            st.dataframe(matrix_rows, hide_index=True, width="stretch")
        st.json({"selected_maps": report.get("selected_maps", []), "skipped": report.get("skipped", []),
                 "tiles": report.get("tiles", []), "policies": report.get("policies", {}),
                 "limitations": report.get("limitations", [])})
    if report.get("skipped"):
        st.caption(f"{len(report['skipped'])} skipped entries; reasons are listed under Scores and coverage.")

    st.download_button("Download evaluation JSON", json.dumps(report, indent=2, allow_nan=False) + "\n",
                       file_name=Path(report_path).name, mime="application/json",
                       icon=":material/download:", on_click="ignore", width="stretch")
