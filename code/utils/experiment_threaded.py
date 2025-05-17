# code/utils/experiment_threaded.py
import time, random, requests, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore, Lock
from typing import Dict, List
from tqdm import tqdm

from code.utils.experiment import ExperimentRunner

# ========= 설정 =========
QPM_LIMIT = 180     # 90 → 180 (계정별 허용치 확인 후 조정)
MAX_WORKERS = 16    # 10 → 16 (동시 스레드 ↑)
MAX_RETRY   = 5      # 429 재시도 횟수
# ========================

# ----- 토큰 버킷 -----
_interval      = 60 / QPM_LIMIT
_bucket        = Semaphore(QPM_LIMIT)
_last_call     = 0.0
_last_call_lck = Lock()

def _rate_limited_post(url, **kwargs):
    """토큰 버킷 + 최소 간격 조절 후 requests.post 실행"""
    global _last_call
    with _bucket:                     # 토큰 하나 사용
        with _last_call_lck:
            now = time.time()
            sleep_for = _interval - (now - _last_call)
            if sleep_for > 0:
                time.sleep(sleep_for)
            _last_call = time.time()
        return requests.post(url, **kwargs, timeout=60)

# -----------------------

class ThreadedExperimentRunner(ExperimentRunner):
    """다중 스레드 + 토큰버킷 + 지수 백오프"""

    def _call_with_retry(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }

        wait = 1.0
        for _ in range(MAX_RETRY):
            r = _rate_limited_post(self.api_url, headers=headers, json=payload)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            if r.status_code == 429:
                time.sleep(wait + random.uniform(0, 0.5))
                wait *= 2
                continue
            r.raise_for_status()

        # 재시도 초과 — 원문 반환해 손실 최소화
        return prompt.split("잘못:")[-1].strip()

    # -------------------

    def _process_row(self, row) -> Dict:
        prompt    = self._make_prompt(row["err_sentence"])
        corrected = self._call_with_retry(prompt)
        return {"id": row["id"], "cor_sentence": corrected}

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        results: List[Dict] = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(self._process_row, row): idx
                       for idx, row in data.iterrows()}

            with tqdm(total=len(futures), desc="Predict", ncols=80) as bar:
                for fut in as_completed(futures):
                    results.append(fut.result())
                    bar.update(1)

        return pd.DataFrame(results)
