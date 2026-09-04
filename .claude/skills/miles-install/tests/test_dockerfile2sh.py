"""Self-test for dockerfile2sh.py — run with `python3 -m unittest` from this directory.

Exercises the Dockerfile subset semantics the translator promises, end to end: parse,
emit, then actually execute the generated bash against a throwaway context.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import dockerfile2sh as d2s  # noqa: E402

SAMPLE = r"""
# syntax=docker/dockerfile:1
ARG BASE_TAG=v1
FROM example/base:${BASE_TAG} AS base
ARG GREETING="hello world"
ARG TARGETARCH
ARG EMPTY=""
ENV LEGACY a b c
ENV PATH="/opt/x/bin:${PATH}" OTHER=1
WORKDIR /root/
WORKDIR sub
RUN echo "$GREETING" > greeting.txt && \
    # a comment inside the continuation must be dropped
    python3 -c "import sys; x=[1, \
2, 3]; \
print(sum(x))" > sum.txt
RUN ["sh", "-c", "echo exec-form > exec.txt"]
COPY files/*.txt copied/
COPY nothin[g]/ /nowhere/
RUN --mount=type=cache,target=/root/.cache --mount=type=bind,source=files/a.txt,target=/tmp/bound.txt \
    cp /tmp/bound.txt bound-copy.txt
LABEL maintainer=nobody
RUN if [ "${EMPTY}" = "" ]; then echo empty-ok > empty.txt; fi
"""


class ParseTests(unittest.TestCase):
    def test_continuation_joins_and_drops_comments(self):
        ins = d2s.parse_dockerfile(SAMPLE)
        runs = [rest for name, rest, _ in ins if name == "RUN"]
        self.assertIn("x=[1, 2, 3]; print(sum(x))", runs[0])
        self.assertNotIn("comment inside", runs[0])

    def test_kv_forms(self):
        self.assertEqual(d2s._parse_kv_list("K v with spaces", legacy_space_form=True), [("K", "v with spaces")])
        self.assertEqual(d2s._parse_kv_list('A="x y" B=1', legacy_space_form=True), [("A", "x y"), ("B", "1")])
        self.assertEqual(d2s._parse_kv_list("NODEFAULT", legacy_space_form=False), [("NODEFAULT", None)])

    def test_multistage_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            df = Path(td) / "Dockerfile"
            df.write_text("FROM a\nFROM b\nRUN true\n")
            with self.assertRaises(d2s.TranslateError):
                d2s.translate(df, Path(td), {}, "t", "t")
            df.write_text("FROM a\nCOPY --from=x /a /b\n")
            with self.assertRaises(d2s.TranslateError):
                d2s.translate(df, Path(td), {}, "t", "t")


class ExecuteTests(unittest.TestCase):
    def test_generated_script_runs(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = Path(td) / "ctx"
            (ctx / "files").mkdir(parents=True)
            (ctx / "files" / "a.txt").write_text("A\n")
            (ctx / "files" / "b.txt").write_text("B\n")
            (ctx / "Dockerfile").write_text(SAMPLE)
            fake_root = Path(td) / "fakeroot"
            # Rewrite absolute paths in the sample so the test never touches the real /root.
            text = (
                SAMPLE.replace("WORKDIR /root/", f"WORKDIR {fake_root}/")
                .replace("/nowhere/", f"{fake_root}/nowhere/")
                .replace("/tmp/bound.txt", f"{fake_root}/bound.txt")
            )
            (ctx / "Dockerfile").write_text(text)
            script, meta = d2s.translate(ctx / "Dockerfile", ctx, {"GREETING": "from build-arg"}, "t", "t")
            self.assertEqual(meta["steps"], 4)
            self.assertEqual(meta["args"]["GREETING"], "from build-arg")
            self.assertEqual(meta["args"]["EMPTY"], "")
            out = Path(td) / "install.sh"
            out.write_text(script)
            env = dict(os.environ, DF_STATE_DIR=str(Path(td) / "state"), HOME=td)
            r = subprocess.run(["bash", str(out)], env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            wd = fake_root / "sub"
            self.assertEqual((wd / "greeting.txt").read_text().strip(), "from build-arg")
            self.assertEqual((wd / "sum.txt").read_text().strip(), "6")
            self.assertEqual((wd / "exec.txt").read_text().strip(), "exec-form")
            self.assertEqual(sorted(p.name for p in (wd / "copied").iterdir()), ["a.txt", "b.txt"])
            self.assertFalse((fake_root / "nowhere").exists(), "empty wildcard COPY must be a no-op")
            self.assertEqual((wd / "bound-copy.txt").read_text().strip(), "A")
            self.assertFalse((fake_root / "bound.txt").exists(), "bind mount target must be removed after the step")
            self.assertEqual((wd / "empty.txt").read_text().strip(), "empty-ok")
            envsh = (Path(td) / "state" / "env.sh").read_text()
            self.assertIn('export PATH="/opt/x/bin:${PATH}"', envsh)
            self.assertIn('export LEGACY="a b c"', envsh)
            # Second run: every step is skipped via markers.
            r2 = subprocess.run(["bash", str(out)], env=env, capture_output=True, text=True)
            self.assertEqual(r2.returncode, 0, r2.stderr)
            self.assertEqual(r2.stderr.count("done earlier, skip"), 4)
            # Dry run prints and touches nothing.
            r3 = subprocess.run(
                ["bash", str(out)], env=dict(env, DF_DRY_RUN="1", DF_FORCE="1"), capture_output=True, text=True
            )
            self.assertEqual(r3.returncode, 0, r3.stderr)
            self.assertIn("      | echo", r3.stderr)


if __name__ == "__main__":
    unittest.main()
