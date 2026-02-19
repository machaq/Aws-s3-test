#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import string
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError, ConnectionClosedError


# ---------------------------
# Percentile
# ---------------------------
def percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


# ---------------------------
# Payload generation (JSON)
# ---------------------------
def make_payload(target_bytes: int, rng: random.Random, seq: int) -> bytes:
    base = {
        "schema": "workload.v1",
        "ts_unix_ms": int(time.time() * 1000),
        "seq": seq,
        "source": "s3-put-workload",
        "meta": {"pid": os.getpid()},
        "data": {"id": f"{seq:012d}", "filler": ""},
    }

    def dumps(obj) -> bytes:
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=True).encode("utf-8")

    payload = dumps(base)
    if len(payload) >= target_bytes:
        return payload[:target_bytes]

    needed = target_bytes - len(payload)
    filler = "".join(rng.choices(string.ascii_letters + string.digits, k=max(0, needed)))
    base["data"]["filler"] = filler
    payload = dumps(base)

    if len(payload) > target_bytes:
        diff = len(payload) - target_bytes
        if diff < len(filler):
            base["data"]["filler"] = filler[:-diff]
            payload = dumps(base)

    # final clamp/pad
    if len(payload) < target_bytes:
        payload += b" " * (target_bytes - len(payload))
    elif len(payload) > target_bytes:
        payload = payload[:target_bytes]
    return payload


# ---------------------------
# Retry policy (app-level)
# ---------------------------
RETRYABLE_EXC = (EndpointConnectionError, ConnectionClosedError)

def is_retryable_client_error(e: ClientError) -> bool:
    code = e.response.get("Error", {}).get("Code", "")
    status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
    return code in {
        "SlowDown",
        "Throttling",
        "ThrottlingException",
        "RequestTimeout",
        "InternalError",
        "ServiceUnavailable",
        "ExpiredTokenException",
    } or status in {429, 500, 502, 503, 504}


@dataclass
class PutResult:
    ok: bool
    latency_s: float
    bytes_sent: int
    attempt: int
    scheduled_t: float
    start_t: float
    end_t: float
    error: Optional[str] = None
    status_code: Optional[int] = None


def put_one(
    s3_client,
    bucket: str,
    key: str,
    body: bytes,
    content_type: str,
    storage_class: Optional[str],
    sse: Optional[str],
    max_attempts: int,
    base_backoff_ms: int,
    jitter_ms: int,
    scheduled_t: float,
) -> PutResult:
    attempt = 0
    start_t = time.perf_counter()
    while True:
        attempt += 1
        t0 = time.perf_counter()
        try:
            extra = {"ContentType": content_type}
            if storage_class:
                extra["StorageClass"] = storage_class
            if sse:
                if sse.lower() in ("aes256", "aes-256"):
                    extra["ServerSideEncryption"] = "AES256"
                elif sse.lower() in ("aws:kms", "kms"):
                    extra["ServerSideEncryption"] = "aws:kms"
                else:
                    extra["ServerSideEncryption"] = sse

            resp = s3_client.put_object(Bucket=bucket, Key=key, Body=body, **extra)
            dt = time.perf_counter() - t0
            end_t = time.perf_counter()
            meta = resp.get("ResponseMetadata", {}) or {}
            return PutResult(
                ok=True,
                latency_s=dt,
                bytes_sent=len(body),
                attempt=attempt,
                scheduled_t=scheduled_t,
                start_t=start_t,
                end_t=end_t,
                status_code=meta.get("HTTPStatusCode"),
            )

        except ClientError as e:
            dt = time.perf_counter() - t0
            status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            err_code = e.response.get("Error", {}).get("Code", "ClientError")
            if attempt < max_attempts and is_retryable_client_error(e):
                backoff = base_backoff_ms * (2 ** (attempt - 1))
                time.sleep((backoff + random.randint(0, jitter_ms)) / 1000.0)
                continue
            end_t = time.perf_counter()
            return PutResult(
                ok=False,
                latency_s=dt,
                bytes_sent=len(body),
                attempt=attempt,
                scheduled_t=scheduled_t,
                start_t=start_t,
                end_t=end_t,
                error=f"{err_code} (http {status})",
                status_code=status,
            )

        except RETRYABLE_EXC as e:
            dt = time.perf_counter() - t0
            if attempt < max_attempts:
                backoff = base_backoff_ms * (2 ** (attempt - 1))
                time.sleep((backoff + random.randint(0, jitter_ms)) / 1000.0)
                continue
            end_t = time.perf_counter()
            return PutResult(
                ok=False,
                latency_s=dt,
                bytes_sent=len(body),
                attempt=attempt,
                scheduled_t=scheduled_t,
                start_t=start_t,
                end_t=end_t,
                error=type(e).__name__,
            )

        except Exception as e:
            dt = time.perf_counter() - t0
            end_t = time.perf_counter()
            return PutResult(
                ok=False,
                latency_s=dt,
                bytes_sent=len(body),
                attempt=attempt,
                scheduled_t=scheduled_t,
                start_t=start_t,
                end_t=end_t,
                error=f"{type(e).__name__}: {e}",
            )


# ---------------------------
# Workload model
# ---------------------------
def poisson_interarrival(rng: random.Random, rps: float) -> float:
    # exponential distribution with mean 1/rps
    return rng.expovariate(rps)

def fixed_interarrival(rps: float) -> float:
    return 1.0 / rps

def sample_size(rng: random.Random, base: int, jitter_pct: float) -> int:
    if jitter_pct <= 0:
        return base
    # uniform +/- jitter_pct
    lo = int(base * (1.0 - jitter_pct))
    hi = int(base * (1.0 + jitter_pct))
    return max(1, rng.randint(lo, hi))

def choose_prefix(rng: random.Random, base_prefix: str, tenants: int, hotspot_pct: float) -> str:
    """
    tenants>1: multi-tenant distribution.
    hotspot_pct: percentage of traffic forced into tenant 0 to simulate hot prefix.
    """
    if tenants <= 1:
        return base_prefix
    if hotspot_pct > 0 and rng.random() < hotspot_pct:
        t = 0
    else:
        t = rng.randrange(tenants)
    return f"{base_prefix}tenant={t:03d}/"


# ---------------------------
# Windowed metrics
# ---------------------------
@dataclass
class WindowStats:
    start_s: float
    end_s: float
    sent: int = 0
    ok: int = 0
    fail: int = 0
    bytes_sent: int = 0
    lats: List[float] = None

    def __post_init__(self):
        if self.lats is None:
            self.lats = []

def summarize_window(w: WindowStats) -> Dict[str, str]:
    dur = w.end_s - w.start_s
    lats_sorted = sorted(w.lats)
    p50 = percentile(lats_sorted, 50)
    p95 = percentile(lats_sorted, 95)
    p99 = percentile(lats_sorted, 99)
    put_s = (w.sent / dur) if dur > 0 else 0.0
    mib_s = ((w.bytes_sent / (1024 * 1024)) / dur) if dur > 0 else 0.0
    return {
        "window_start_s": f"{w.start_s:.3f}",
        "window_end_s": f"{w.end_s:.3f}",
        "sent": str(w.sent),
        "ok": str(w.ok),
        "fail": str(w.fail),
        "put_per_s": f"{put_s:.2f}",
        "mib_per_s": f"{mib_s:.3f}",
        "p50_ms": f"{p50*1000:.2f}" if p50 == p50 else "",
        "p95_ms": f"{p95*1000:.2f}" if p95 == p95 else "",
        "p99_ms": f"{p99*1000:.2f}" if p99 == p99 else "",
    }


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="S3 PUT workload replay (1KB JSON-ish).")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--base-prefix", default="bench/workload/")
    ap.add_argument("--duration", type=int, default=60, help="Test duration seconds")
    ap.add_argument("--rps", type=float, default=200.0, help="Target average requests/sec")
    ap.add_argument("--arrival", choices=["poisson", "fixed"], default="poisson")
    ap.add_argument("--burst-mult", type=float, default=1.0, help="Burst multiplier (e.g., 3.0)")
    ap.add_argument("--burst-every", type=int, default=0, help="Every N seconds apply burst for burst-len (0=off)")
    ap.add_argument("--burst-len", type=int, default=3, help="Burst length seconds")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--size-jitter", type=float, default=0.0, help="Size jitter pct (e.g., 0.2 = +/-20%)")
    ap.add_argument("--tenants", type=int, default=1, help="Tenant count for prefix distribution")
    ap.add_argument("--hotspot", type=float, default=0.0, help="Hotspot ratio [0..1], traffic forced to tenant 0")
    ap.add_argument("--concurrency", type=int, default=128, help="Worker threads")
    ap.add_argument("--region", default=None)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--content-type", default="application/json")
    ap.add_argument("--storage-class", default=None)
    ap.add_argument("--sse", default=None)
    ap.add_argument("--max-attempts", type=int, default=6)
    ap.add_argument("--base-backoff-ms", type=int, default=50)
    ap.add_argument("--jitter-ms", type=int, default=30)
    ap.add_argument("--client-max-pool", type=int, default=256)
    ap.add_argument("--connect-timeout", type=int, default=5)
    ap.add_argument("--read-timeout", type=int, default=60)
    ap.add_argument("--window", type=float, default=1.0, help="Metrics window seconds")
    ap.add_argument("--csv", default=None, help="Write per-window metrics CSV path")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    run_id = str(int(time.time() * 1000))

    session_kwargs = {}
    if args.profile:
        session_kwargs["profile_name"] = args.profile
    session = boto3.Session(**session_kwargs)

    cfg = Config(
        region_name=args.region,
        retries={"max_attempts": 2, "mode": "standard"},  # keep low; app-level retries handle rest
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
        max_pool_connections=args.client_max_pool,
    )
    s3 = session.client("s3", config=cfg)

    # Schedule generation: create (scheduled_time, seq, key, payload) on the fly
    t0 = time.perf_counter()
    t_end = t0 + args.duration
    next_t = t0
    seq = 0

    # Metrics windows
    w_start = t0
    w_end = w_start + args.window
    cur = WindowStats(start_s=0.0, end_s=args.window)  # relative seconds
    windows: List[Dict[str, str]] = []

    def current_rps(at_time: float) -> float:
        # apply periodic burst if configured
        if args.burst_every and args.burst_mult > 1.0:
            elapsed = at_time - t0
            in_cycle = elapsed % args.burst_every
            if in_cycle < args.burst_len:
                return args.rps * args.burst_mult
        return args.rps

    def interarrival(at_time: float) -> float:
        rps_now = max(0.0001, current_rps(at_time))
        if args.arrival == "poisson":
            return poisson_interarrival(rng, rps_now)
        return fixed_interarrival(rps_now)

    # Submit tasks as schedule time arrives
    futures = []
    submitted = 0

    def submit_one(executor, scheduled_t: float, i: int):
        nonlocal submitted
        prefix = choose_prefix(rng, args.base_prefix, args.tenants, args.hotspot)
        key = f"{prefix}run={run_id}/{i:012d}.json"
        size = sample_size(rng, args.size, args.size_jitter)
        body = make_payload(size, rng, i)
        fut = executor.submit(
            put_one,
            s3,
            args.bucket,
            key,
            body,
            args.content_type,
            args.storage_class,
            args.sse,
            args.max_attempts,
            args.base_backoff_ms,
            args.jitter_ms,
            scheduled_t,
        )
        futures.append(fut)
        submitted += 1

    print("=== Workload ===")
    print(f"bucket={args.bucket}")
    print(f"run_id={run_id}")
    print(f"duration={args.duration}s, rps={args.rps}, arrival={args.arrival}")
    if args.burst_every and args.burst_mult > 1.0:
        print(f"burst: every {args.burst_every}s for {args.burst_len}s, mult={args.burst_mult}")
    print(f"payload: base={args.size}B, jitter={args.size_jitter*100:.0f}%")
    print(f"key dist: tenants={args.tenants}, hotspot={args.hotspot*100:.1f}% to tenant0")
    print(f"concurrency={args.concurrency}, window={args.window}s")
    print("Running...\n")

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        # Producer loop: schedule submits until duration end
        while True:
            now = time.perf_counter()
            if now >= t_end:
                break

            # sleep until next schedule time
            if now < next_t:
                time.sleep(min(0.01, next_t - now))
                continue

            # submit this request
            submit_one(ex, scheduled_t=next_t, i=seq)
            seq += 1

            # compute next schedule time from interarrival
            next_t += interarrival(now)

        # Consumer loop: collect all results
        ok = fail = 0
        bytes_sent = 0
        lat_ok_all: List[float] = []
        lat_all: List[float] = []
        attempts_total = 0

        for fut in as_completed(futures):
            r: PutResult = fut.result()
            rel_end = r.end_t - t0
            # advance windows up to rel_end
            while rel_end >= w_end - t0:
                # finalize current window
                windows.append(summarize_window(cur))
                # print a compact line
                last = windows[-1]
                print(
                    f"[{float(last['window_start_s']):6.1f}-{float(last['window_end_s']):6.1f}s] "
                    f"sent={last['sent']:>6} ok={last['ok']:>6} fail={last['fail']:>4} "
                    f"put/s={last['put_per_s']:>7} p95={last['p95_ms']:>8}ms"
                )
                # move to next window
                w_start = w_end
                w_end = w_start + args.window
                cur = WindowStats(start_s=w_start - t0, end_s=w_end - t0)

            # record into current window
            cur.sent += 1
            cur.bytes_sent += r.bytes_sent
            cur.lats.append(r.latency_s)

            attempts_total += r.attempt
            bytes_sent += r.bytes_sent
            lat_all.append(r.latency_s)
            if r.ok:
                ok += 1
                cur.ok += 1
                lat_ok_all.append(r.latency_s)
            else:
                fail += 1
                cur.fail += 1

        # flush remaining window
        windows.append(summarize_window(cur))

    elapsed = time.perf_counter() - t0
    lat_all.sort()
    lat_ok_all.sort()

    def ms(x): return x * 1000.0

    print("\n=== Summary ===")
    print(f"submitted          : {submitted}")
    print(f"ok / fail          : {ok} / {fail}  (success {ok/max(1,submitted)*100:.2f}%)")
    print(f"elapsed            : {elapsed:.3f}s")
    print(f"avg offered rps     : {submitted/elapsed:.2f} req/s (actual scheduled)")
    print(f"data rate          : {(bytes_sent/(1024*1024))/elapsed:.3f} MiB/s")
    print(f"avg attempts        : {attempts_total/max(1,submitted):.2f}")

    print("\nLatency (ALL):")
    print(f"  p50  : {ms(percentile(lat_all, 50)):.2f} ms")
    print(f"  p95  : {ms(percentile(lat_all, 95)):.2f} ms")
    print(f"  p99  : {ms(percentile(lat_all, 99)):.2f} ms")
    print(f"  max  : {ms(lat_all[-1]) if lat_all else float('nan'):.2f} ms")

    print("\nLatency (OK only):")
    print(f"  p50  : {ms(percentile(lat_ok_all, 50)):.2f} ms")
    print(f"  p95  : {ms(percentile(lat_ok_all, 95)):.2f} ms")
    print(f"  p99  : {ms(percentile(lat_ok_all, 99)):.2f} ms")
    print(f"  max  : {ms(lat_ok_all[-1]) if lat_ok_all else float('nan'):.2f} ms")

    if args.csv:
        fieldnames = [
            "window_start_s","window_end_s","sent","ok","fail","put_per_s","mib_per_s","p50_ms","p95_ms","p99_ms"
        ]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in windows:
                w.writerow(row)
        print(f"\nWrote window metrics CSV: {args.csv}")


if __name__ == "__main__":
    main()