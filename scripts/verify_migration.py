#!/usr/bin/env python3
"""Verify independent source boundaries and byte-for-byte CPU demo parity."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def checksums(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def _acceptance_projection(contents, name):
    """Remove only the additive P0-3 fields from a successful demo's schema."""
    outcome = {"model_decision", "module_decision", "before_checkpoint",
               "proposed_checkpoint", "accepted_checkpoint"}

    def strip_outcome(row):
        if not outcome <= row.keys(): raise AssertionError("Missing model acceptance fields")
        if (row["model_decision"] != "accept"
            or row["accepted_checkpoint"] != row["proposed_checkpoint"]
            or row["module_decision"] != row.get("decision", row["module_decision"])):
            raise AssertionError("Positive demo's model/module decision changed")
        return {k: v for k, v in row.items() if k not in outcome}

    if name == "events.jsonl":
        return [strip_outcome(row) if row["kind"] == "cycle_complete" else row
                for row in (json.loads(line) for line in contents.splitlines())]
    row = json.loads(contents)
    if name.endswith("/retirement.json"):
        row = strip_outcome(row)
        for key in ("model_acceptance", "capability_preserved", "cost_improved",
                    "retirement_eligible", "retirement_accepted"):
            row.pop(key)
        row["intervals"].pop("C-A")
    elif name.endswith("/state.json"):
        row = strip_outcome(row)
        row["archive"] = [strip_outcome(entry) for entry in row["archive"]]
    elif name == "deployment.json":
        row["archive"] = [strip_outcome(entry) for entry in row["archive"]]
    else:
        raise AssertionError(f"Unexpected demo file changed: {name}")
    return row


def verify(before, after, *, require_removed=True, allow_model_acceptance_fields=False):
    baseline = checksums(before) if before else json.loads(
        (ROOT / "docs/history/migration-baseline/demo-files.sha256.json").read_text())
    actual = checksums(after)
    differences = [name for name, value in baseline.items() if actual.get(name) != value]
    semantic_matches = []
    if allow_model_acceptance_fields:
        if before is None: raise ValueError("Schema comparison requires --before baseline files")
        for name in differences:
            contents = (before / name).read_text()
            expected = ([json.loads(line) for line in contents.splitlines()] if name == "events.jsonl"
                        else json.loads(contents))
            projected = _acceptance_projection((after / name).read_text(), name)
            if projected != expected: raise AssertionError(f"Original demo semantics changed: {name}")
            semantic_matches.append(name)
        differences = []
    if differences: raise AssertionError(f"Original demo output changed: {differences}")
    test_hashes = json.loads((ROOT / "docs/history/migration-baseline/original-tests.sha256.json").read_text())
    for name, expected in test_hashes.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise AssertionError(f"Original test edited: {name}")
    private = {"opid", "agent_system", "gigpo", "meta_harness"}
    forbidden = ["upstream" + "/" + name for name in ("OPID", "meta-harness")]
    forbidden += ["install_opid" + "_patch.py", "opid-integration" + ".patch"]
    hits, files = [], []
    for directory in ("src", "scripts", "configs"):
        for file in sorted((ROOT / directory).rglob("*")):
            if not file.is_file() or "__pycache__" in file.parts: continue
            if not require_removed and file.name == "install_opid" + "_patch.py": continue
            contents = file.read_text()
            if any(value in contents for value in forbidden): hits.append(str(file.relative_to(ROOT)))
            if file.suffix == ".py":
                files.append(file)
                for node in ast.walk(ast.parse(contents, filename=str(file))):
                    imported = []
                    if isinstance(node, ast.Import): imported = [a.name for a in node.names]
                    elif isinstance(node, ast.ImportFrom) and not node.level: imported = [node.module or ""]
                    if any(name.split(".")[0] in private for name in imported):
                        hits.append(str(file.relative_to(ROOT)))
            elif file.suffix == ".json": json.loads(contents)
    if hits: raise AssertionError(f"Private import/path/patch references remain: {hits}")
    if require_removed:
        targets = [ROOT / "upstream" / name for name in ("OPID", "meta-harness")]
        targets += [ROOT / "patches" / ("opid-integration" + ".patch"),
                    ROOT / "scripts" / ("install_opid" + "_patch.py")]
        if any(p.exists() for p in targets): raise AssertionError("Removal targets still exist")
    return {"original_test_files_unchanged": len(test_hashes),
            "original_demo_files_equal": len(baseline), "original_demo_differences": differences,
            "original_demo_files_byte_equal": len(baseline) - len(semantic_matches),
            "acceptance_schema_only_changes": semantic_matches,
            "additional_demo_files": sorted(set(actual)-set(baseline)),
            "python_files_parsed": len(files), "private_import_or_path_hits": hits,
            "removal_required": require_removed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--pre-removal", action="store_true")
    parser.add_argument("--allow-model-acceptance-fields", action="store_true",
                        help="Compare added P0-3 verdict fields separately; require --before")
    args = parser.parse_args()
    print(json.dumps(verify(args.before, args.after, require_removed=not args.pre_removal,
                           allow_model_acceptance_fields=args.allow_model_acceptance_fields), indent=2))


if __name__ == "__main__": main()
