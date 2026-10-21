import argparse
import csv
import importlib
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


#maths
def run_trials(samples):
    
    names = [s[0] for s in samples]
    photos = [s[1] for s in samples]
    E = np.array([s[2] for s in samples], dtype=np.float64)

    sq = (E ** 2).sum(axis=1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * E @ E.T, 0.0))  # pairwise distances

    photos_per_person = Counter(names)
    trials = []
    for i in range(len(samples)):
        # Enrolled person
        if photos_per_person[names[i]] >= 2:
            d = D[i].copy()
            d[i] = np.inf
            j = int(np.argmin(d))
            trials.append(("genuine", names[i], photos[i], names[j], float(d[j])))

        # Stranger
        others = [j for j in range(len(samples)) if names[j] != names[i]]
        if others:
            j = others[int(np.argmin(D[i, others]))]
            trials.append(("impostor", names[i], photos[i], names[j], float(D[i, j])))
    return trials


def summarize(trials, t):
    gen = [x for x in trials if x[0] == "genuine"]
    imp = [x for x in trials if x[0] == "impostor"]
    return {
        "n_gen": len(gen),
        "n_imp": len(imp),
        "correct": sum(1 for _, p, _, nn, d in gen if d <= t and nn == p),
        "wrong_name": sum(1 for _, p, _, nn, d in gen if d <= t and nn != p),
        "rejected": sum(1 for _, _, _, _, d in gen if d > t),
        "false_accept": sum(1 for _, _, _, _, d in imp if d <= t),
    }


def recommend(trials, lo=0.20):
    hi = max(d for *_, d in trials) + 0.02
    best, best_cost = [], None
    for t in np.round(np.arange(lo, hi + 1e-9, 0.01), 2):
        s = summarize(trials, float(t))
        cost = 0.0
        if s["n_gen"]:
            cost += (s["wrong_name"] + s["rejected"]) / s["n_gen"]
        if s["n_imp"]:
            cost += s["false_accept"] / s["n_imp"]
        if best_cost is None or cost < best_cost - 1e-12:
            best, best_cost = [float(t)], cost
        elif abs(cost - best_cost) <= 1e-12:
            best.append(float(t))

    runs, run = [], [best[0]]            #unbroken stretch of best tolerances
    for t in best[1:]:
        if t - run[-1] <= 0.0101:
            run.append(t)
        else:
            runs.append(run)
            run = [t]
    runs.append(run)
    run = max(runs, key=len)

    low, high = run[0], run[-1]
    pick = round(low + (high - low) / 3, 2)
    pick = max(low, min(pick, 0.60))
    return pick, low, high, best_cost


#project glue
def find_src():
    here = Path(__file__).resolve()
    for p in (here.parent.parent / "src", Path.cwd() / "src", Path.cwd().parent / "src"):
        if any((p / n).exists() for n in ("detector.py", "Detector.py")):
            return p
    sys.exit("Could not find the src folder (looking for detector.py). Run from the project root.")


def load_class(attr, *module_names):
    for name in module_names:
        try:
            return getattr(importlib.import_module(name), attr)
        except ModuleNotFoundError as e:
            if e.name != name:
                raise
    raise ImportError(f"Could not find {attr} in any of: {module_names}")


def collect_samples(known_dir, detector, encoder):
    samples, skipped = [], []
    for person_dir in sorted(p for p in known_dir.iterdir() if p.is_dir()):
        for img in sorted(person_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            frame = cv2.imread(str(img))
            if frame is None:
                skipped.append((person_dir.name, img.name, "could not read image"))
                continue
            locations = detector.detect_faces(frame)
            if len(locations) != 1:
                skipped.append((person_dir.name, img.name, f"found {len(locations)} faces"))
                continue
            samples.append((person_dir.name, img.name, encoder.encode_faces(frame, locations)[0]))
    return samples, skipped


def pct(n, total):
    return f"{100.0 * n / total:5.1f}%" if total else "   n/a"


def print_report(samples, skipped, trials, thresholds):
    people = Counter(s[0] for s in samples)
    gen = [x for x in trials if x[0] == "genuine"]
    imp = [x for x in trials if x[0] == "impostor"]

    print(f"\nPeople: {len(people)}  |  usable photos: {len(samples)}  |  skipped: {len(skipped)}")
    print("Photos per person:", ", ".join(f"{n} ({c})" for n, c in sorted(people.items())))
    for who, photo, why in skipped:
        print(f"  [skipped] {who}/{photo}: {why}")
    print(f"Enrolled-person trials: {len(gen)}   Stranger trials: {len(imp)}")

    if len(people) < 2:
        print("\nNeed at least 2 people to test strangers. Only the enrolled-person results are valid.")
    if len(gen) < 30 or len(imp) < 30:
        print("\nNote: fewer than 30 trials in a group, so treat these percentages as rough, not precise.")

    head = ["tolerance", "recognised correctly", "wrong name", "rejected as Unknown", "stranger accepted"]
    print("\n" + " | ".join(f"{h:>20}" for h in head))
    print("-" * 112)
    for t in thresholds:
        s = summarize(trials, t)
        row = [f"{t:.2f}",
               f"{pct(s['correct'], s['n_gen'])} ({s['correct']}/{s['n_gen']})",
               f"{pct(s['wrong_name'], s['n_gen'])} ({s['wrong_name']})",
               f"{pct(s['rejected'], s['n_gen'])} ({s['rejected']})",
               f"{pct(s['false_accept'], s['n_imp'])} ({s['false_accept']}/{s['n_imp']})"]
        print(" | ".join(f"{c:>20}" for c in row))

    if gen:
        gd = np.array([d for *_, d in gen])
        print(f"\nDistance to the person's own nearest photo: median {np.median(gd):.2f}, max {gd.max():.2f}")
    if imp:
        idist = np.array([d for *_, d in imp])
        print(f"Distance of a stranger to the nearest enrolled face: median {np.median(idist):.2f}, min {idist.min():.2f}")
    if gen and imp:
        pick, lo, hi, cost = recommend(trials)
        if cost == 0:
            print(f"\nNo errors for tolerances from {lo:.2f} to {hi:.2f}.")
            print(f"Suggested: {pick:.2f}. Only a few strangers were tested and real strangers can look more "
                  "alike than your teammates, so a value in the lower part of that range is safer.")
        else:
            s = summarize(trials, pick)
            print(f"\nNo tolerance gets everything right. Fewest errors: {lo:.2f} to {hi:.2f}; suggested {pick:.2f}.")
            print(f"At {pick:.2f}: {pct(s['correct'], s['n_gen']).strip()} of enrolled people recognised, "
                  f"{pct(s['false_accept'], s['n_imp']).strip()} of strangers accepted.")
            print("More varied, better-lit photos usually help more than changing the tolerance.")

    print("\nHow to read this:")
    print("  recognised correctly = enrolled person, nearest face is theirs and within tolerance")
    print("  wrong name           = enrolled person, but the system named someone else")
    print("  rejected as Unknown  = enrolled person, but too far from their own photos")
    print("  stranger accepted    = someone not enrolled got a name (the dangerous error)")
    print("\nCaution: photos taken in one sitting look alike, which flatters these numbers. For a fairer")
    print("test, add photos taken at another time of day or in different light.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.40, 0.45, 0.50, 0.55, 0.60])
    ap.add_argument("--csv", default=None, help="where to write every trial (default: data/eval_trials.csv)")
    ap.add_argument("--no-csv", action="store_true", help="do not write the CSV")
    args = ap.parse_args()

    src = find_src()
    sys.path.insert(0, str(src))
    base = src.parent
    known_dir = base / "data" / "known_faces"
    if not known_dir.exists():
        sys.exit(f"No folder at {known_dir}. Capture some photos first.")

    detector = load_class("FaceDetector", "detector", "Detector")()
    encoder = load_class("FaceEncoder", "encoder", "Encoder")()

    print("Encoding photos (this can take a minute)...")
    samples, skipped = collect_samples(known_dir, detector, encoder)
    if len(samples) < 2:
        sys.exit("Need at least 2 usable photos. Capture more and run enroll.py's checks first.")

    trials = run_trials(samples)
    print_report(samples, skipped, trials, args.thresholds)

    if not args.no_csv:
        out = Path(args.csv) if args.csv else base / "data" / "eval_trials.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["kind", "person", "photo", "nearest_name", "distance"])
            for kind, person, photo, nearest, d in trials:
                w.writerow([kind, person, photo, nearest, f"{d:.4f}"])
        print(f"\nEvery trial was saved to {out} (handy for a chart in Excel).")


if __name__ == "__main__":
    main()