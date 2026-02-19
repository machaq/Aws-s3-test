#!/usr/bin/env python3
"""
S3へ「約1KBのJSON」を大量PUTし、性能検証（スループット/レイテンシ/失敗率/リトライ回数など）を行うスクリプト。

この版で実装している要件：
- 失敗率（fail/total）
- 平均リトライ回数（attempt-1 の平均）
- prefix分散（tenant=xxx/ に分散、hotspotで偏りも再現）
- 一定RPS固定（--rps でオファードRPSを固定して投入）

注意（設計上のポイント）：
- --rps は「投入（オファード）RPS」です。S3やネットワークが詰まると実効PUT/sは落ちます。
  ただしそれ自体が“ワークロード再現”では重要です（遅延・失敗・リトライの波形が出る）。
- boto3 / botocore の内部リトライに頼りすぎると制御しづらいので、
  botocore側リトライは控えめにして（retries max_attemptsを低め）、アプリ側で明示リトライを行います。
"""

import argparse
import json
import os
import random
import string
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError, ConnectionClosedError


# ---------------------------
# Utility: percentile（パーセンタイル計算）
# ---------------------------
def percentile(sorted_vals: List[float], p: float) -> float:
    """
    昇順ソート済み配列 sorted_vals に対して pパーセンタイルを返します。
    - p=50 => median
    - p=95 => p95
    連続値として線形補間する一般的な方法。

    注意：
    - 呼び出し側でソートしてから渡す設計（ここで毎回ソートしない）。
    """
    if not sorted_vals:
        return float("nan")
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]

    # 配列長 N に対して 0..N-1 の位置に対応付ける
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)  # floor
    c = min(f + 1, len(sorted_vals) - 1)  # ceil（末尾超えないように）
    if f == c:
        return sorted_vals[f]

    # 線形補間
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


# ---------------------------
# Payload generation (~1KB JSON)
# ---------------------------
def make_payload(target_bytes: int, seed: int, seq: int) -> bytes:
    """
    target_bytes（デフォルト1024B）程度のJSONバイト列を生成します。
    - seq を含めてオブジェクトごとに内容が変わるようにする
    - filler を調整してほぼ指定サイズに合わせる
    - ensure_ascii=True でASCIIに寄せ、文字数とバイト数の乖離を減らす（UTF-8の可変長を避ける）

    注意：
    - 完全にピッタリに合わせるのはJSONエスケープ等で難しいので、最後に pad/trim を行う。
    """
    # オブジェクトごとに再現可能な乱数を得るために seed と seq を混ぜる
    rng = random.Random(seed ^ (seq * 0x9E3779B1))

    # ベース構造：実運用のイベントJSONを想定して、多少階層を持たせる
    base = {
        "schema": "bench.v1",
        "ts_unix_ms": int(time.time() * 1000),
        "seq": seq,
        "source": "s3-put-bench",
        "meta": {
            "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
            "pid": os.getpid(),
        },
        "data": {
            "id": f"{seq:012d}",
            "note": "1KB json payload for s3 put benchmark",
            "filler": ""
        }
    }

    def dumps(obj) -> bytes:
        # separators を詰めて無駄な空白を削り、サイズ調整しやすくする
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=True).encode("utf-8")

    payload = dumps(base)

    # 既に target_bytes を超えていたら、切り詰め（あまり起きないが保険）
    if len(payload) >= target_bytes:
        return payload[:target_bytes]

    # 足りない分だけ filler を追加して再度ダンプ
    needed = target_bytes - len(payload)
    filler = "".join(rng.choices(string.ascii_letters + string.digits, k=max(0, needed)))
    base["data"]["filler"] = filler
    payload = dumps(base)

    # もし少しオーバーしたら filler を削って調整し直す
    if len(payload) > target_bytes:
        diff = len(payload) - target_bytes
        if diff < len(filler):
            base["data"]["filler"] = filler[:-diff]
            payload = dumps(base)

    # 最終調整：それでもズレる場合に備え、パディングor切り詰め
    if len(payload) != target_bytes:
        if len(payload) < target_bytes:
            payload += b" " * (target_bytes - len(payload))
        else:
            payload = payload[:target_bytes]

    return payload


# ---------------------------
# Prefix distribution（prefix分散/ホットスポット再現）
# ---------------------------
def choose_prefix(base_prefix: str, tenants: int, hotspot: float, rng: random.Random) -> str:
    """
    キーのprefixを分散させます。

    - tenants > 1 の場合、tenant=000..(tenants-1) に分散
      例: base_prefix="prod/events/" なら "prod/events/tenant=005/" のようになる
    - hotspot を 0..1 で指定すると、その割合だけ tenant=000 に偏らせる（ホットプレフィックス再現）
      例: hotspot=0.2 => 20%のリクエストは tenant=000 に集中

    S3はprefix偏りがあるとクライアント側/ネットワーク側のボトルネックが露呈しやすく、
    実運用に近い偏りの再現に役立ちます。
    """
    if tenants <= 1:
        return base_prefix

    # hotspotが有効で、確率的にホット側に寄せる
    if hotspot > 0 and rng.random() < hotspot:
        t = 0
    else:
        t = rng.randrange(tenants)

    return f"{base_prefix}tenant={t:03d}/"


# ---------------------------
# Result types（1 PUTの結果）
# ---------------------------
@dataclass
class PutResult:
    """
    1回の put_object の結果を保持する構造体
    - ok: 成功/失敗
    - latency_s: その試行でのレイテンシ（秒）
    - bytes_sent: Bodyサイズ
    - attempt: 何回目で最終結果になったか（=試行回数。1ならリトライ無し）
    - error/status_code/request_id: 失敗時の情報やデバッグ用
    """
    ok: bool
    latency_s: float
    bytes_sent: int
    attempt: int
    error: Optional[str] = None
    status_code: Optional[int] = None
    request_id: Optional[str] = None


# ---------------------------
# Upload worker with retries（アプリ側リトライ）
# ---------------------------
# ネットワーク的に再試行価値が高い例外
RETRYABLE_EXC = (
    EndpointConnectionError,  # エンドポイント接続不可（瞬断など）
    ConnectionClosedError,    # 接続が閉じられた（keep-alive周りなど）
)

def is_retryable_client_error(e: ClientError) -> bool:
    """
    S3のClientErrorのうち、「一時的」なものをリトライ対象として判定します。
    - SlowDown / Throttling / 5xx / 429 などを対象にするのが一般的
    """
    try:
        code = e.response.get("Error", {}).get("Code", "")
        status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
    except Exception:
        return False

    return code in {
        "SlowDown",
        "Throttling",
        "ThrottlingException",
        "RequestTimeout",
        "RequestTimeTooSkewed",
        "InternalError",
        "ServiceUnavailable",
        "ExpiredTokenException",
    } or status in {429, 500, 502, 503, 504}


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
) -> PutResult:
    """
    1オブジェクトをS3へPUTする関数（必要ならリトライする）。
    ThreadPoolExecutor から並列に呼び出される想定。

    リトライ設計：
    - exponential backoff: base_backoff_ms * 2^(attempt-1)
    - jitter（ランダム揺らぎ）を足して、同期的な再送集中を緩和
    """
    attempt = 0
    while True:
        attempt += 1
        t0 = time.perf_counter()

        try:
            # put_object の追加オプション
            extra = {"ContentType": content_type}
            if storage_class:
                extra["StorageClass"] = storage_class

            # SSE指定：AES256 or aws:kms を簡易サポート
            if sse:
                if sse.lower() in ("aes256", "aes-256"):
                    extra["ServerSideEncryption"] = "AES256"
                elif sse.lower() in ("aws:kms", "kms"):
                    extra["ServerSideEncryption"] = "aws:kms"
                else:
                    # それ以外の指定はそのまま渡す（利用者側で正しい値を指定）
                    extra["ServerSideEncryption"] = sse

            # 実際のPUT（この呼び出しがS3へのリクエスト）
            resp = s3_client.put_object(Bucket=bucket, Key=key, Body=body, **extra)

            dt = time.perf_counter() - t0
            meta = resp.get("ResponseMetadata", {}) or {}
            return PutResult(
                ok=True,
                latency_s=dt,
                bytes_sent=len(body),
                attempt=attempt,
                status_code=meta.get("HTTPStatusCode"),
                request_id=meta.get("RequestId") or meta.get("HostId"),
            )

        except ClientError as e:
            # AWS側（S3）が返したエラー
            dt = time.perf_counter() - t0
            retryable = is_retryable_client_error(e)
            status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            err_code = e.response.get("Error", {}).get("Code", "ClientError")

            # リトライ可能かつ max_attempts 未満なら待って再試行
            if attempt < max_attempts and retryable:
                backoff = base_backoff_ms * (2 ** (attempt - 1))
                sleep_ms = backoff + random.randint(0, jitter_ms)
                time.sleep(sleep_ms / 1000.0)
                continue

            # リトライしない（or できない）場合は失敗として返す
            return PutResult(
                ok=False,
                latency_s=dt,
                bytes_sent=len(body),
                attempt=attempt,
                error=f"{err_code} (http {status})",
                status_code=status,
            )

        except RETRYABLE_EXC as e:
            # ネットワーク系（瞬断など）
            dt = time.perf_counter() - t0
            if attempt < max_attempts:
                backoff = base_backoff_ms * (2 ** (attempt - 1))
                sleep_ms = backoff + random.randint(0, jitter_ms)
                time.sleep(sleep_ms / 1000.0)
                continue

            return PutResult(
                ok=False,
                latency_s=dt,
                bytes_sent=len(body),
                attempt=attempt,
                error=type(e).__name__,
            )

        except Exception as e:
            # 想定外（コードバグ・環境問題など）
            dt = time.perf_counter() - t0
            return PutResult(
                ok=False,
                latency_s=dt,
                bytes_sent=len(body),
                attempt=attempt,
                error=f"{type(e).__name__}: {e}",
            )


# ---------------------------
# Main benchmark（CLI引数処理・並列投入・集計）
# ---------------------------
def main():
    ap = argparse.ArgumentParser(
        description="S3 PUT benchmark for ~1KB JSON objects with fixed RPS + prefix distribution."
    )

    # 必須：対象バケット
    ap.add_argument("--bucket", required=True, help="Target S3 bucket name")

    # baseのprefix（ここに tenant=xxx/ が追加される）
    ap.add_argument("--prefix", default="bench/1kb-json/", help="Base key prefix")

    # 総件数
    ap.add_argument("--count", type=int, default=10000, help="Number of objects")

    # JSONサイズ（バイト）
    ap.add_argument("--size", type=int, default=1024, help="Payload size in bytes (default 1024)")

    # ワーカー（スレッド数）。HTTP接続プールやCPUとのバランスで調整。
    ap.add_argument("--concurrency", type=int, default=64, help="Parallel PUT workers")

    # 一定RPS固定（0なら無制限投入＝スレッドが許す限り詰める）
    ap.add_argument("--rps", type=float, default=0.0, help="Fixed offered RPS (0 = no rate limit)")

    # prefix分散：tenant数
    ap.add_argument("--tenants", type=int, default=1, help="Prefix dispersion tenant count (>=1)")

    # hotspot：tenant=000へ偏らせる割合（0..1）
    ap.add_argument("--hotspot", type=float, default=0.0, help="Hot prefix ratio [0..1] to tenant=000/")

    # AWS設定
    ap.add_argument("--region", default=None, help="AWS region (optional)")
    ap.add_argument("--profile", default=None, help="AWS profile (optional)")

    # payload生成のseed（同一seedなら同一パターンで再現可能）
    ap.add_argument("--seed", type=int, default=12345, help="Seed for payload generation")

    # S3オブジェクトの属性
    ap.add_argument("--content-type", default="application/json", help="Content-Type")
    ap.add_argument("--storage-class", default=None, help="e.g. STANDARD, STANDARD_IA")
    ap.add_argument("--sse", default=None, help="SSE: AES256 or aws:kms")

    # リトライ制御（アプリ側）
    ap.add_argument("--max-attempts", type=int, default=6, help="Max attempts per object (includes first)")
    ap.add_argument("--base-backoff-ms", type=int, default=50, help="Retry base backoff (ms)")
    ap.add_argument("--jitter-ms", type=int, default=30, help="Retry jitter (ms)")

    # botocoreのHTTPクライアント設定
    ap.add_argument("--connect-timeout", type=int, default=5, help="Connect timeout seconds")
    ap.add_argument("--read-timeout", type=int, default=60, help="Read timeout seconds")
    ap.add_argument("--client-max-pool", type=int, default=256, help="botocore max_pool_connections")

    # 進捗表示間隔
    ap.add_argument("--print-every", type=int, default=1000, help="Progress print interval")

    # キーのフォーマット
    # {prefix} には choose_prefix() の結果（tenant=xxx/を含む）が入る
    ap.add_argument(
        "--key-format",
        default="{prefix}run={run_id}/{i:08d}.json",
        help="Key format template; {prefix} already includes tenant/"
    )

    # run-id（未指定なら epochms）
    ap.add_argument("--run-id", default=None, help="Run id for keys; default=epochms")

    args = ap.parse_args()

    # 引数バリデーション：変な値で動かさない
    if not (0.0 <= args.hotspot <= 1.0):
        raise SystemExit("--hotspot must be in [0..1]")
    if args.tenants < 1:
        raise SystemExit("--tenants must be >= 1")
    if args.rps < 0:
        raise SystemExit("--rps must be >= 0")

    # run_idはキーに含める。並行実行してもprefixが衝突しにくくなる
    run_id = args.run_id or str(int(time.time() * 1000))

    # prefix選択用の乱数（payload生成のrngと分離）
    # こうしておくと、payloadのseedを変えなくてもprefix分散の再現性が保てる
    rng_prefix = random.Random(args.seed ^ 0xC0FFEE)

    # boto3 session（profileがあれば指定）
    session_kwargs = {}
    if args.profile:
        session_kwargs["profile_name"] = args.profile
    session = boto3.Session(**session_kwargs)

    # botocore Config
    # - retries: botocore側の自動リトライは少なめ（2回程度）にして、
    #   アプリ側の詳細制御を優先する
    cfg = Config(
        region_name=args.region,
        retries={"max_attempts": 2, "mode": "standard"},
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
        max_pool_connections=args.client_max_pool,
    )
    s3 = session.client("s3", config=cfg)

    total = args.count
    conc = args.concurrency

    # レイテンシ蓄積（後でパーセンタイル算出）
    lat_ok: List[float] = []
    lat_all: List[float] = []

    # 成否カウンタ
    ok = 0
    fail = 0

    # 総送信バイト数（MiB/s計算用）
    bytes_total = 0

    # 試行回数合計（attemptは「1回目も含む」）
    attempts_total = 0

    # リトライ回数合計（attempt-1）
    retries_total = 0

    # ベンチマーク開始時刻
    t_start = time.perf_counter()

    def submit_index(i: int):
        """
        1件分のPUTを実行するための関数（executorに渡す）。
        - prefix分散を決める
        - keyを組み立てる
        - payloadを生成する
        - put_one() を呼ぶ
        """
        pfx = choose_prefix(args.prefix, args.tenants, args.hotspot, rng_prefix)
        key = args.key_format.format(prefix=pfx, run_id=run_id, i=i)
        body = make_payload(args.size, args.seed, i)

        return put_one(
            s3_client=s3,
            bucket=args.bucket,
            key=key,
            body=body,
            content_type=args.content_type,
            storage_class=args.storage_class,
            sse=args.sse,
            max_attempts=args.max_attempts,
            base_backoff_ms=args.base_backoff_ms,
            jitter_ms=args.jitter_ms,
        )

    # ThreadPoolExecutorで並列PUT
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futures = {}

        # ここが「一定RPS固定」の肝：
        # i件目は t_start + i*(1/rps) に投入する、というスケジュールを作る
        # ただし実行の完了は並列実行とS3側状況に依存し、遅延しても次の投入は続く
        # => キューが溜まれば実効レイテンシ増・失敗増を再現できる
        interval = (1.0 / args.rps) if args.rps and args.rps > 0 else 0.0

        # Producer: submit するタイミングを制御しながら futures を作る
        for i in range(total):
            if interval > 0:
                # i件目の目標投入時刻（固定RPS）
                target = t_start + i * interval
                now = time.perf_counter()
                sleep_s = target - now
                if sleep_s > 0:
                    time.sleep(sleep_s)

            futures[ex.submit(submit_index, i)] = i

        # Consumer: 完了したfutureから順に結果回収（完了順）
        for n, fut in enumerate(as_completed(futures), start=1):
            r: PutResult = fut.result()

            # 全体レイテンシ統計（成功/失敗どちらも）
            lat_all.append(r.latency_s)

            # 試行回数・リトライ回数統計
            attempts_total += r.attempt
            retries_total += max(0, r.attempt - 1)

            # 送信バイト数
            bytes_total += r.bytes_sent

            # 成否カウント＆成功レイテンシ統計
            if r.ok:
                ok += 1
                lat_ok.append(r.latency_s)
            else:
                fail += 1

            # 進捗表示：一定間隔で、失敗率/平均リトライ/PUT/s などを出す
            if args.print_every and (n % args.print_every == 0 or n == total):
                elapsed = time.perf_counter() - t_start
                put_s = n / elapsed if elapsed > 0 else 0.0
                mib_s = (bytes_total / (1024 * 1024)) / elapsed if elapsed > 0 else 0.0
                fail_rate = (fail / n) * 100.0 if n else 0.0
                avg_retries = (retries_total / n) if n else 0.0

                print(
                    f"[{n}/{total}] ok={ok} fail={fail} "
                    f"fail_rate={fail_rate:.2f}% avg_retries={avg_retries:.3f} "
                    f"throughput={put_s:,.1f} PUT/s, {mib_s:,.2f} MiB/s"
                )

    # 経過時間
    elapsed = time.perf_counter() - t_start

    # パーセンタイル計算のためにソート
    lat_all.sort()
    lat_ok.sort()

    def fmt_ms(x: float) -> str:
        return f"{x*1000:.2f} ms"

    # 最終統計：失敗率と平均リトライ回数
    fail_rate_total = (fail / total) * 100.0 if total else 0.0
    avg_retries_total = (retries_total / total) if total else 0.0

    # サマリ出力
    print("\n=== Summary ===")
    print(f"Bucket            : {args.bucket}")
    print(f"Base Prefix       : {args.prefix}")
    print(f"Tenants/Hotspot   : {args.tenants} / {args.hotspot:.2f}")
    print(f"Run ID            : {run_id}")
    print(f"Objects attempted : {total}")
    print(f"OK / Fail         : {ok} / {fail}")
    print(f"Failure rate      : {fail_rate_total:.2f}%")
    print(f"Avg retries       : {avg_retries_total:.3f}  (retries = attempts-1)")
    print(f"Avg attempts      : {attempts_total/total:.3f}")
    print(f"Concurrency       : {conc}")
    print(f"Offered RPS       : {args.rps if args.rps > 0 else 'unlimited'}")
    print(f"Payload bytes     : {args.size}")
    print(f"Elapsed           : {elapsed:.3f} s")
    print(f"Total bytes       : {bytes_total} ({bytes_total/(1024*1024):.2f} MiB)")
    print(f"Throughput        : {total/elapsed:,.1f} PUT/s")
    print(f"Data rate         : {(bytes_total/(1024*1024))/elapsed:,.2f} MiB/s")

    print("\nLatency (ALL, includes failures):")
    print(f"  p50  : {fmt_ms(percentile(lat_all, 50))}")
    print(f"  p95  : {fmt_ms(percentile(lat_all, 95))}")
    print(f"  p99  : {fmt_ms(percentile(lat_all, 99))}")
    print(f"  max  : {fmt_ms(lat_all[-1]) if lat_all else 'n/a'}")

    print("\nLatency (OK only):")
    print(f"  p50  : {fmt_ms(percentile(lat_ok, 50))}")
    print(f"  p95  : {fmt_ms(percentile(lat_ok, 95))}")
    print(f"  p99  : {fmt_ms(percentile(lat_ok, 99))}")
    print(f"  max  : {fmt_ms(lat_ok[-1]) if lat_ok else 'n/a'}")


if __name__ == "__main__":
    main()
