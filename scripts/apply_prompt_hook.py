#!/usr/bin/env python3
"""D2: put the exact prompt into the trace. Two small anchored edits, safe to re-run.

    python3 scripts/apply_prompt_hook.py --check
    python3 scripts/apply_prompt_hook.py

1. api/app/services/prompt.py : build_messages() records the rendered system+user messages in the active trace
   (every provider - ollama, hosted, stub - builds its prompt here, so one hook covers all of them).
2. api/app/services/answer.py : records the configured LLM model name in the trace.
"""
import argparse
import sys
from pathlib import Path

PLAN = [
    ("api/app/services/prompt.py", "import", 
     "from ..schemas import Answer, Passage\n",
     "from ..obs.tracing import record_prompt\nfrom ..schemas import Answer, Passage\n",
     "from ..obs.tracing import record_prompt"),
    ("api/app/services/prompt.py", "record prompt",
     '    return [{"role": "system", "content": system}, {"role": "user", "content": user}]\n',
     '    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]\n'
     '    # D2: the exact text sent to the model goes into the request\'s trace (no-op outside a traced request)\n'
     '    record_prompt("\\n\\n".join(f"[{m[\'role\']}]\\n{m[\'content\']}" for m in messages))\n'
     '    return messages\n',
     "record_prompt(\"\\n\\n\""),
    ("api/app/services/answer.py", "model name",
     '    trace.rec["llm"]["provider"] = provider.name\n',
     '    trace.rec["llm"]["provider"] = provider.name\n'
     '    trace.rec["llm"]["model"] = getattr(settings, "llm_model", None) or trace.rec["llm"]["model"]\n',
     'getattr(settings, "llm_model"'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    texts, todo, bad = {}, [], []
    for path, name, old, new, marker in PLAN:
        texts.setdefault(path, Path(path).read_text(encoding="utf-8"))
        t = texts[path]
        if marker in t:
            print(f"  already applied : {name} ({path})")
        elif t.count(old) != 1:
            print(f"  ANCHOR PROBLEM  : {name} ({path}) matches={t.count(old)}")
            bad.append(name)
        else:
            print(f"  will apply      : {name} ({path})")
            todo.append((path, old, new))
    if bad:
        sys.exit("Nothing written. Paste the lines above to Claude.")
    if a.check or not todo:
        return print("(check only)" if a.check else "Nothing to do.")
    for path, old, new in todo:
        texts[path] = texts[path].replace(old, new, 1)
    for path, t in texts.items():
        compile(t, path, "exec")
    for path, t in texts.items():
        Path(path).write_text(t, encoding="utf-8")
    print(f"Patched {len(todo)} place(s). Review: git diff api/app/services")


if __name__ == "__main__":
    main()
