#!/usr/bin/env python3
"""
Enroll several people, then measure how well identification actually works.

This is the end-to-end test of the matching algorithm: real presses in, a name
out, scored against the truth. It exercises the same engine the gym system uses
(fingerprint_local, or NBIS if you ever install it), so what it reports is what
the door will do.

Output is plain ASCII, so it survives a Windows console in any code page.

    # 1. enroll 10 people in one sitting (asks for each name)
    py -3 test/test_enroll_identify.py enroll-many --count 10

    #    or one at a time, which also works over ssh / in scripts
    py -3 test/test_enroll_identify.py enroll --name ali
    py -3 test/test_enroll_identify.py enroll --name sara

    # 2. see who is enrolled and whether their prints are good enough
    py -3 test/test_enroll_identify.py list

    # 3. offline accuracy over everyone enrolled - no sensor needed
    py -3 test/test_enroll_identify.py evaluate

    # 4. live identification - press a finger, get a name
    py -3 test/test_enroll_identify.py identify --trials 10
    py -3 test/test_enroll_identify.py identify --expect ali

Enrollments are just PNGs in fingerprint_storage/images named
<person>_sample_<n>.png, the same layout test_local_engine.py reads, so the two
tools share one dataset. `clear` removes a person.

STOP the main gym application first - this needs the image UDP port.

These files are biometric data: keep the directory access-controlled.
"""

import argparse
import collections
import itertools
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fingerprint_local import (  # noqa: E402
    LocalEngine, build_template, parse_xyt, match_templates,
    MIN_ENROLL_MINUTIAE as MIN_MINUTIAE,
)

IMAGES = ROOT / "fingerprint_storage" / "images"
SAFE = re.compile(r"[^A-Za-z0-9_-]")


# --------------------------------------------------------------- gallery ----

def load_gallery(engine, quiet=False):
    """Build templates from every enrolled PNG. Returns {name: [(n, tpl, count)]}."""
    people = collections.defaultdict(list)
    for path in sorted(IMAGES.glob("*_sample_*.png")):
        people[path.name.split("_sample_")[0]].append(path)

    gallery = {}
    for name, paths in sorted(people.items()):
        entries = []
        for p in paths:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            t = engine.extract_template(img)
            if t is None:
                entries.append((p.name, None, 0))
                continue
            entries.append((p.name, build_template(parse_xyt(t["xyt"])), t["minutiae_count"]))
        if entries:
            gallery[name] = entries
    if not quiet:
        usable = sum(1 for v in gallery.values() for _n, t, _c in v if t is not None)
        print(f"[gallery] {len(gallery)} people, {usable} usable templates")
    return gallery


def best_scores(probe, gallery, exclude=None):
    """Score a probe against every person. Returns [(name, score)] best first."""
    out = []
    for name, entries in gallery.items():
        best = 0
        for fname, tpl, _count in entries:
            if tpl is None or (exclude is not None and fname == exclude):
                continue
            s = match_templates(probe, tpl)
            if s > best:
                best = s
        out.append((name, best))
    out.sort(key=lambda r: -r[1])
    return out


# --------------------------------------------------------------- capture ----

def open_receiver(port):
    from fingerprint_gym import UDPFingerprintReceiver
    try:
        return UDPFingerprintReceiver(port)
    except OSError as exc:
        print(f"[ERROR] cannot listen on port {port}: {exc}")
        print("        Stop the main gym application first - it holds that port.")
        return None


def capture(receiver, engine, timeout, prompt):
    """Wait for one press. Returns (image, template, minutiae) or (None, None, 0)."""
    print(f"  {prompt}", flush=True)
    img = receiver.receive_image(timeout=timeout)
    if img is None:
        print("  ...timed out, no image")
        return None, None, 0
    t = engine.extract_template(img)
    if t is None:
        print("  ...no minutiae found in that press")
        return img, None, 0
    return img, build_template(parse_xyt(t["xyt"])), t["minutiae_count"]


# ----------------------------------------------------------------- enroll ---

def cmd_enroll(args, engine):
    name = SAFE.sub("_", args.name)[:40]
    if not name:
        print("[ERROR] --name must contain letters or digits")
        return 2

    IMAGES.mkdir(parents=True, exist_ok=True)
    existing = sorted(IMAGES.glob(f"{name}_sample_*.png"))
    if existing and not args.replace:
        print(f"[ERROR] '{name}' already has {len(existing)} samples. "
              f"Use --replace to re-enroll, or pick another name.")
        return 2
    for p in existing:
        p.unlink()

    receiver = open_receiver(args.image_port)
    if receiver is None:
        return 1

    print(f"\nEnrolling '{name}' - {args.samples} presses")
    print("Lift the finger fully between presses and shift it slightly each time.\n")

    saved, counts, rejected = 0, [], 0
    try:
        while saved < args.samples:
            img, tpl, n = capture(receiver, engine, args.timeout,
                                  f"[{saved+1}/{args.samples}] place finger...")
            if img is None:
                rejected += 1
                if rejected > args.retries:
                    print("\n[FAIL] too many timeouts - is the sensor reachable?")
                    return 1
                continue
            if tpl is None or n < MIN_MINUTIAE:
                rejected += 1
                print(f"  ...REJECTED: {n} minutiae (need {MIN_MINUTIAE}). "
                      f"Press firmer and cover the whole window.")
                if rejected > args.retries:
                    print(f"\n[FAIL] '{name}' could not produce {args.samples} good presses.")
                    print("       Clean the sensor; a dry finger also reads badly.")
                    return 1
                continue

            saved += 1
            counts.append(n)
            path = IMAGES / f"{name}_sample_{saved}.png"
            cv2.imwrite(str(path), img)
            print(f"  ...saved {path.name} ({n} minutiae)")
            if saved < args.samples:
                time.sleep(0.8)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 1
    finally:
        receiver.close()

    mean = float(np.mean(counts))
    print(f"\n[OK] '{name}' enrolled: {saved} samples, "
          f"minutiae min {min(counts)} / mean {mean:.1f} / max {max(counts)}")
    if mean < 15:
        print("[WARN] this finger averages under 15 minutiae. Fingers this faint "
              "match poorly - consider re-enrolling a different finger.")
    total = len({p.name.split('_sample_')[0] for p in IMAGES.glob('*_sample_*.png')})
    print(f"       {total} people enrolled so far")
    return 0


def cmd_enroll_many(args, engine):
    """Enroll several people back to back, asking for each name as you go."""
    if not sys.stdin or not sys.stdin.isatty():
        print("[ERROR] enroll-many needs an interactive terminal to ask for names.")
        print("        Run it in your own shell, or use one 'enroll --name X' per person.")
        return 2

    done = 0
    for slot in range(1, args.count + 1):
        print(f"\n{'='*60}")
        print(f"Person {slot} of {args.count}")
        print(f"{'='*60}")
        try:
            name = input("  name (blank to stop): ").strip()
        except EOFError:
            if slot == 1:
                print()
                print("[ERROR] no terminal available to read names from.")
                print("        Run this in your own shell, or use "
                      "'enroll --name X' once per person.")
                return 2
            break
        if not name:
            print("  stopping")
            break

        one = argparse.Namespace(
            name=name, samples=args.samples, retries=args.retries,
            replace=args.replace, image_port=args.image_port, timeout=args.timeout,
        )
        if cmd_enroll(one, engine) == 0:
            done += 1
        else:
            print("  (not enrolled - try again with a different finger)")

    print(f"\n{done} of {args.count} enrolled.")
    print("Now measure it:  py -3 test/test_enroll_identify.py evaluate")
    return 0


# --------------------------------------------------------------- evaluate ---

def cmd_evaluate(args, engine):
    """Offline accuracy over everyone enrolled. Needs no sensor."""
    gallery = load_gallery(engine)
    if len(gallery) < 2:
        print("[ERROR] enroll at least 2 people first")
        return 2

    print(f"\n{'='*64}\nPer-person capture quality\n{'='*64}")
    print(f"{'person':>22} {'samples':>8} {'minutiae':>9} {'self':>6} {'verdict':>12}")
    genuine_by_person = {}
    for name, entries in gallery.items():
        tpls = [t for _f, t, _c in entries if t is not None]
        counts = [c for _f, _t, c in entries if c]
        gen = [match_templates(a, b) for a, b in itertools.combinations(tpls, 2)]
        g = float(np.mean(gen)) if gen else 0.0
        genuine_by_person[name] = g
        verdict = "good" if g >= 8 else "marginal" if g >= 6 else "WEAK"
        print(f"{name:>22} {len(entries):>8} {np.mean(counts) if counts else 0:>9.1f} "
              f"{g:>6.1f} {verdict:>12}")

    # genuine / impostor over the whole set
    gen, imp = [], []
    for name, entries in gallery.items():
        tpls = [t for _f, t, _c in entries if t is not None]
        gen += [match_templates(a, b) for a, b in itertools.combinations(tpls, 2)]
    for n1, n2 in itertools.combinations(gallery, 2):
        for _f1, t1, _c1 in gallery[n1]:
            for _f2, t2, _c2 in gallery[n2]:
                if t1 is not None and t2 is not None:
                    imp.append(match_templates(t1, t2))
    gen, imp = np.array(gen, float), np.array(imp, float)
    if not len(gen) or not len(imp):
        print("[ERROR] not enough usable templates")
        return 2

    print(f"\n{'='*64}\nSeparation\n{'='*64}")
    print(f"  genuine  ({len(gen):4d} pairs): mean {gen.mean():5.1f}  p5 {np.percentile(gen,5):.0f}")
    print(f"  impostor ({len(imp):4d} pairs): mean {imp.mean():5.1f}  p95 {np.percentile(imp,95):.0f}  max {imp.max():.0f}")
    d = abs(gen.mean()-imp.mean())/np.sqrt((gen.var()+imp.var())/2)
    best = None
    for t in np.unique(np.concatenate([gen, imp])):
        far, frr = (imp >= t).mean(), (gen < t).mean()
        if best is None or abs(far-frr) < abs(best[1]-best[2]):
            best = (t, far, frr)
    print(f"  d' = {d:.2f} (>3 usable)    EER = {(best[1]+best[2])/2*100:.1f}%")

    print(f"\n{'='*64}\nIdentification (leave-one-out: each press vs everyone else)\n{'='*64}")
    correct = total = 0
    confusion = []
    margins = []
    for name, entries in gallery.items():
        for fname, tpl, _c in entries:
            if tpl is None:
                continue
            total += 1
            ranking = best_scores(tpl, gallery, exclude=fname)
            top, score = ranking[0]
            runner = ranking[1][1] if len(ranking) > 1 else 0
            margins.append(score - runner)
            if top == name:
                correct += 1
            else:
                confusion.append((fname, name, top, score, runner))
    print(f"  rank-1 accuracy: {correct}/{total} = {correct/total*100:.1f}%")
    print(f"  mean margin over the runner-up: {np.mean(margins):.1f} matched minutiae")
    if confusion:
        print(f"\n  misidentified ({len(confusion)}):")
        for fname, truth, got, sc, run in confusion[:15]:
            print(f"    {fname:<30} is {truth:<18} -> said {got:<18} ({sc} vs {run})")

    print(f"\n{'='*64}\nOperating points (threshold on matched minutiae)\n{'='*64}")
    print(f"  {'accept at':>10} {'FAR':>8} {'FRR':>8}   meaning")
    for t in range(4, int(gen.max()) + 2):
        far, frr = (imp >= t).mean()*100, (gen < t).mean()*100
        note = ""
        if far == 0 and not note:
            note = "no false accepts in this sample"
        print(f"  {'>= '+str(t):>10} {far:7.2f}% {frr:7.1f}%   {note}")
    print("\n  FAR = strangers let in, FRR = members turned away.")
    print(f"  LOCAL_MATCH_THRESHOLD is currently {engine.threshold}.")
    if len(gallery) < 10:
        print(f"\n[NOTE] only {len(gallery)} people enrolled. FAR is measured against "
              f"{len(imp)} impostor pairs,\n       so it cannot resolve rates below "
              f"about {100/max(len(imp),1):.1f}%. Enroll more people for a firmer number.")
    return 0


# --------------------------------------------------------------- identify ---

def cmd_identify(args, engine):
    gallery = load_gallery(engine)
    if not gallery:
        print("[ERROR] nobody is enrolled yet")
        return 2

    receiver = open_receiver(args.image_port)
    if receiver is None:
        return 1

    print(f"\nIdentifying against {len(gallery)} enrolled people "
          f"(threshold {engine.threshold})")
    if args.expect:
        print(f"Expecting: {args.expect}")
    print()

    hits = misses = rejects = 0
    try:
        for trial in range(1, args.trials + 1):
            img, tpl, n = capture(receiver, engine, args.timeout,
                                  f"[trial {trial}/{args.trials}] place finger...")
            if img is None:
                continue
            if tpl is None:
                rejects += 1
                continue

            t0 = time.time()
            ranking = best_scores(tpl, gallery)
            elapsed = (time.time() - t0) * 1000
            top, score = ranking[0]
            runner_name, runner = ranking[1] if len(ranking) > 1 else ("-", 0)

            if score >= engine.threshold:
                verdict = f"MATCH  {top}"
            else:
                verdict = "NO MATCH"
            line = (f"  -> {verdict:<24} score {score:2d} "
                    f"(runner-up {runner_name} {runner}, {n} minutiae, {elapsed:.0f} ms)")
            if args.expect:
                truth = SAFE.sub("_", args.expect)[:40]
                if score < engine.threshold:
                    rejects += 1
                    line += "  [FALSE REJECT]"
                elif top == truth:
                    hits += 1
                    line += "  [correct]"
                else:
                    misses += 1
                    line += "  [WRONG PERSON]"
            print(line)
            if args.verbose:
                for nm, sc in ranking[:5]:
                    print(f"       {nm:<24} {sc}")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        receiver.close()

    if args.expect:
        done = hits + misses + rejects
        if done:
            print(f"\n{hits}/{done} correct, {misses} wrong person, {rejects} falsely rejected")
    return 0


# ------------------------------------------------------------------- misc ---

def cmd_list(args, engine):
    gallery = load_gallery(engine, quiet=True)
    if not gallery:
        print("nobody enrolled yet")
        return 0
    print(f"{'person':>22} {'samples':>8} {'minutiae (min/mean/max)':>26}")
    for name, entries in gallery.items():
        c = [x for _f, _t, x in entries if x]
        if c:
            print(f"{name:>22} {len(entries):>8} "
                  f"{min(c):>10} /{np.mean(c):6.1f} /{max(c):5}")
        else:
            print(f"{name:>22} {len(entries):>8} {'no usable templates':>26}")
    print(f"\n{len(gallery)} people enrolled")
    return 0


def cmd_clear(args, engine):
    target = SAFE.sub("_", args.name)[:40] if args.name else None
    paths = sorted(IMAGES.glob(f"{target}_sample_*.png" if target else "*_sample_*.png"))
    if not paths:
        print("nothing to remove")
        return 0
    if not args.yes:
        who = target or f"ALL {len({p.name.split('_sample_')[0] for p in paths})} people"
        print(f"would delete {len(paths)} images for {who}")
        print("re-run with --yes to actually delete")
        return 0
    for p in paths:
        p.unlink()
    print(f"deleted {len(paths)} images")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--image-port", type=int, default=8889)
        p.add_argument("--timeout", type=float, default=60.0,
                       help="seconds to wait for each press")

    e = sub.add_parser("enroll", help="record one person's finger")
    e.add_argument("--name", required=True)
    e.add_argument("--samples", type=int, default=5)
    e.add_argument("--retries", type=int, default=6,
                   help="bad presses tolerated before giving up")
    e.add_argument("--replace", action="store_true", help="re-enroll an existing person")
    common(e)

    m = sub.add_parser("enroll-many", help="enroll several people in one sitting")
    m.add_argument("--count", type=int, default=10, help="how many people")
    m.add_argument("--samples", type=int, default=5)
    m.add_argument("--retries", type=int, default=6)
    m.add_argument("--replace", action="store_true")
    common(m)

    i = sub.add_parser("identify", help="live 1:N identification")
    i.add_argument("--trials", type=int, default=1)
    i.add_argument("--expect", help="who it should be, to score correctness")
    i.add_argument("--verbose", action="store_true", help="show the top 5 per trial")
    common(i)

    v = sub.add_parser("evaluate", help="offline accuracy, no sensor needed")
    sub.add_parser("list", help="who is enrolled")

    c = sub.add_parser("clear", help="remove enrollments")
    c.add_argument("--name", help="person to remove (default: everyone)")
    c.add_argument("--yes", action="store_true")

    args = ap.parse_args()
    engine = LocalEngine()
    return {
        "enroll": cmd_enroll, "enroll-many": cmd_enroll_many, "identify": cmd_identify,
        "evaluate": cmd_evaluate, "list": cmd_list, "clear": cmd_clear,
    }[args.cmd](args, engine)


if __name__ == "__main__":
    sys.exit(main())
