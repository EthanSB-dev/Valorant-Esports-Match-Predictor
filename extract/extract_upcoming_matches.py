"""
Extract currently-scheduled (not yet played) flagship VCT matches.

Unlike extract_matches.py, this does NOT use the checkpoint system.
Checkpointing only makes sense for progressing forward through matches
that have already happened (tracked by end_at) — an upcoming match has
no end_at yet, and its scheduled time can shift, so the correct behavior
here is to simply re-fetch the full current upcoming schedule every time
this runs, not to track "new since last time."

Once a match here is actually played, the regular extract_matches.py +
load_matches.py flow will naturally pick up its real result later and
overwrite this placeholder row (load_matches.py upserts by match_id).
"""
import json
import os
from datetime import datetime, timezone

from pandascore_client import PandaScoreClient
from vct_filters import is_flagship_vct_serie, VCT_LEAGUE_ID

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def extract_upcoming_matches():
    client = PandaScoreClient()
    upcoming_matches = []
    page = 1

    while True:
        resp = client.get_upcoming_matches_page(page=page, per_page=50)
        matches = resp.json()

        if not matches:
            break

        for m in matches:
            serie_name = m.get("serie", {}).get("full_name", "")
            league_id = m.get("league", {}).get("id")
            if is_flagship_vct_serie(serie_name) and league_id == VCT_LEAGUE_ID:
                upcoming_matches.append(m)

        print(f"Page {page}: scanned {len(matches)} matches, kept {len(upcoming_matches)} so far")
        page += 1

    if not upcoming_matches:
        print("No upcoming flagship VCT matches found. Nothing to write.")
        return

    os.makedirs(RAW_DIR, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(RAW_DIR, f"matches_upcoming_{run_id}.json")

    with open(out_path, "w") as f:
        json.dump(upcoming_matches, f, indent=2)
    print(f"Wrote {len(upcoming_matches)} upcoming matches to {out_path}")


if __name__ == "__main__":
    extract_upcoming_matches()