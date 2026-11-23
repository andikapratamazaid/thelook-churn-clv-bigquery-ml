"""Generate a local copy of the TheLook eCommerce tables.

Provenance
----------
The BigQuery public dataset ``bigquery-public-data.thelook_ecommerce`` is
synthetic. It is produced by an official generator published by Google in the
``GoogleCloudPlatform/public-datasets-pipelines`` repository
(``datasets/thelook_ecommerce/pipelines/_images/run_thelook_kub/fake.py``).
The production Airflow DAG in that same repository runs the generator with
``NUM_OF_USERS=100000`` and ``NUM_OF_GHOST_EVENTS=5`` and loads the resulting
CSVs (users, orders, order_items, events, inventory_items, plus the static
products and distribution_centers tables) into BigQuery.

This driver runs that unmodified upstream generator (kept next to this file
as ``thelook_generator_upstream.py``) with the same production parameters,
with three local adaptations:

1. The Google Cloud Storage upload function is replaced by a local file
   writer, so the CSVs land in ``data/raw_mirror/`` instead of a GCS bucket.
2. Fixed random seeds (Python ``random``, NumPy, and Faker) are set so the
   local copy is reproducible. The live BigQuery tables are a different
   random draw from the same generator, so this copy is statistically
   equivalent but not byte-identical to the live tables.
3. Per-row INFO logging is silenced, and the CSV serializer streams rows to
   disk instead of building one giant in-memory string (byte-identical
   output, lower memory use on the events table).
4. The upstream get_address function is replaced by an equivalent cached
   implementation. The upstream version recomputes the same population
   weights over the 19,492 row location table on every call (roughly 40 ms
   per call, and it is called once per user and once per ghost session).
   Both generator call sites use the default arguments, so the weights are
   constant. The replacement precomputes them once and then consumes the
   random stream in exactly the same way (one random() draw, same bisect
   selection over the same cumulative weights), which an A/B run on a
   small user count confirmed produces identical addresses. Only the
   wasted recomputation is removed; the distribution is unchanged.

The seed files ``products.csv``, ``distribution_centers.csv`` and
``world_pop.csv`` in ``data/raw_mirror/seeds/`` are the exact static inputs
used by the production pipeline, downloaded from the same repository.

Usage:
    python3 analysis/generate_thelook_copy.py
Environment overrides: NUM_OF_USERS, NUM_OF_GHOST_EVENTS, OUTPUT_DIR,
SOURCE_DIR, SEED.
"""

import logging
import os
import random
import shutil
import sys

import numpy as np

# The upstream generator imports google.cloud.storage at module level. Its
# only use is the upload function, which this driver replaces, so stub the
# module out instead of installing the Google Cloud client library.
import types

_google = types.ModuleType("google")
_cloud = types.ModuleType("google.cloud")
_storage = types.ModuleType("google.cloud.storage")
_storage.Client = object
_cloud.storage = _storage
_google.cloud = _cloud
sys.modules.setdefault("google", _google)
sys.modules.setdefault("google.cloud", _cloud)
sys.modules.setdefault("google.cloud.storage", _storage)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.environ.get(
    "SOURCE_DIR", os.path.join(BASE_DIR, "data", "raw_mirror", "seeds")
)
# The upstream module reads SOURCE_DIR at import time (it loads the product
# and location seed tables when the module is first imported).
os.environ["SOURCE_DIR"] = SOURCE_DIR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thelook_generator_upstream as upstream

SEED = int(os.environ.get("SEED", "42"))
NUM_OF_USERS = int(os.environ.get("NUM_OF_USERS", "100000"))
NUM_OF_GHOST_EVENTS = int(os.environ.get("NUM_OF_GHOST_EVENTS", "5"))
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR", os.path.join(BASE_DIR, "data", "raw_mirror")
)
# Same list as the production DAG in public-datasets-pipelines.
EXTRANEOUS_HEADERS = [
    "event_type",
    "ip_address",
    "browser",
    "traffic_source",
    "session_id",
    "sequence_number",
    "uri",
    "is_sold",
]


def install_fast_get_address():
    """Replace upstream get_address with a cached equivalent.

    Identical inputs, identical outputs, identical random stream
    consumption (one random() draw per call, bisect over the same
    cumulative population weights, then fake.street_address()). See the
    module docstring, adaptation 4.
    """
    import bisect as _bisect
    import itertools

    data = upstream.LOCATION_DATA
    pops = [int(loc["population"]) for loc in data]
    total_pop = sum(pops)
    weights = [p / total_pop for p in pops]
    cum_weights = list(itertools.accumulate(weights))
    total = cum_weights[-1] + 0.0
    hi = len(cum_weights) - 1
    fake = upstream.fake

    def fast_get_address(*, country="*", state="*", postal_code="*"):
        if country != "*" or state != "*" or postal_code != "*":
            raise ValueError(
                "cached get_address only supports the default arguments, "
                "which are the only ones the generator uses")
        loc = data[_bisect.bisect(cum_weights, random.random() * total,
                                  0, hi)]
        return {
            "street": fake.street_address(),
            "city": loc["city"],
            "state": loc["state"],
            "postal_code": loc["postal_code"],
            "country": loc["country"],
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
        }

    upstream.get_address = fast_get_address


def local_upload(target_bucket, target_prefix, target_object,
                 source_data=None, source_filepath=None):
    """Drop-in replacement for the upstream GCS upload: write locally."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dest = os.path.join(OUTPUT_DIR, target_object)
    if source_data is not None and str(source_data).startswith("@@FILE@@"):
        # Streaming path: dict_to_csv already wrote the CSV to a temp file.
        shutil.move(str(source_data)[len("@@FILE@@"):], dest)
    elif source_data is not None:
        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(str(source_data))
    else:
        shutil.copyfile(source_filepath, dest)
    print(f"wrote {dest}", flush=True)
    return dest


def streaming_dict_to_csv(data):
    """Memory-light version of the upstream dict_to_csv.

    Produces byte-identical CSV output (same csv.writer settings) but
    streams rows to a temp file instead of building one giant string in
    memory, which matters for the roughly 2.4 million row events table.
    Returns a sentinel that local_upload resolves to the final path.
    """
    import csv as _csv
    import io as _io
    import tempfile

    # Temp file lives next to the final output: /tmp may be a small tmpfs
    # that cannot hold the roughly 500 MB events table.
    fd, tmp_path = tempfile.mkstemp(suffix=".csv", dir=OUTPUT_DIR)
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
        writer = _csv.writer(fh, quoting=_csv.QUOTE_NONNUMERIC)
        header_writer = _csv.DictWriter(fh, fieldnames=data[0].keys())
        header_writer.writeheader()
        for dat in data:
            writer.writerow(dat.values())
    return "@@FILE@@" + tmp_path


def main():
    logging.getLogger().setLevel(logging.WARNING)
    os.environ["SOURCE_DIR"] = SOURCE_DIR
    random.seed(SEED)
    np.random.seed(SEED)
    upstream.fake.seed_instance(SEED)
    upstream.upload_to_bucket = local_upload
    upstream.dict_to_csv = streaming_dict_to_csv
    if os.environ.get("SKIP_FAST_ADDRESS") != "1":
        install_fast_get_address()
    upstream.main(
        num_of_users=NUM_OF_USERS,
        num_of_ghost_events=NUM_OF_GHOST_EVENTS,
        target_gcs_prefix="local",
        target_gcs_bucket="local",
        source_dir=SOURCE_DIR,
        extraneous_headers=EXTRANEOUS_HEADERS,
    )


if __name__ == "__main__":
    main()
