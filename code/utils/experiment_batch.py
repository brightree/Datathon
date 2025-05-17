import re, time, random, requests, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict
from code.utils.experiment import ExperimentRunner

BATCH_SIZE   = 12   # 문장 12개씩 한 프롬프트
MAX_WORKERS  = 6    # 동시 배치 호출 수
MAX_RETRY    = 4
QPM_LIMIT    = 120  # 계정 허용치에 맞게

# ── 토큰 버킷 ───────────────────────────────────────
_interval, _last = 60 / QPM_LIMIT, 0.0
def rate_post(url, **kw):
    global _last
    now = time.time()
    if now - _last < _interval:
        time.sleep(_interval - (now - _last))
    _last = time.time()
    return requests.post(url, timeout=60, **kw)
# ──────────────────────────────────────────────────

# 번호 유니코드
NUM = ["①","②","③","④","⑤","⑥","⑦","⑧","⑨","⑩","⑪","⑫"]

class BatchExperimentRunner(ExperimentRunner):
    """12문장씩 묶어 한 번에 요청 → 속도 10배"""

    # 배치 프롬프트 작성
    def _build_prompt(self, batch_df: pd.DataFrame) -> str:
        lines = []
        for i, err in enumerate(batch_df["err_sentence"]):
            lines.append(f"{NUM[i]} 잘못: {err}")
        inputs = "\n".join(lines)
        # 템플릿 규칙(한 번만) + numbered inputs
        return (
            self.template.split("### 작업")[0]  # 규칙 부분만 재활용
            + "\n\n### 작업\n"
            + inputs
            + "\n\n교정:"
        )

    # 응답 파싱
    def _parse(self, text: str, k: int) -> List[str]:
        outs = [""] * k
        for ln in text.splitlines():
            m = re.match(r"^[①-⑫]\s*(.+)$", ln.strip())
            if m:
                idx = NUM.index(ln[0])
                if idx < k:
                    outs[idx] = m.group(1).strip()
        # 누락 시 원문 그대로
        for i, s in enumerate(outs):
            if not s:
                outs[i] = "<<EMPTY>>"
        return outs

    # 재시도 포함 한 배치 처리
    def _handle_batch(self, batch_df: pd.DataFrame) -> List[str]:
        prompt = self._build_prompt(batch_df)
        hdr = {"Authorization": f"Bearer {self.api_key}",
               "Content-Type": "application/json"}
        payload = {"model": self.model,
                   "messages":[{"role":"user","content":prompt}],
                   "temperature": self.config.temperature,
                   "max_tokens": 512,
                   "top_p": 0.9,
                   "stop": [] }

        wait = 1.0
        for _ in range(MAX_RETRY):
            r = rate_post(self.api_url, headers=hdr, json=payload)
            if r.status_code == 200:
                return self._parse(
                    r.json()["choices"][0]["message"]["content"],
                    len(batch_df)
                )
            if r.status_code == 429:
                time.sleep(wait + random.random())
                wait *= 2
                continue
            r.raise_for_status()
        # 실패 시 원문 반환
        return list(batch_df["err_sentence"])

    # 오버라이드 run
    # def run(self, data: pd.DataFrame) -> pd.DataFrame:
    #     batches = [data.iloc[i:i+BATCH_SIZE]
    #                for i in range(0, len(data), BATCH_SIZE)]
    #     results = []
    #     with ThreadPoolExecutor(MAX_WORKERS) as pool, \
    #          tqdm(total=len(batches), desc="Batch") as bar:
    #         futs = {pool.submit(self._handle_batch, b): b for b in batches}
    #         for fut in as_completed(futs):
    #             batch_df = futs[fut]
    #             for _id, fixed in zip(batch_df["id"], fut.result()):
    #                 results.append({"id": _id, "cor_sentence": fixed})
    #             bar.update(1)
    #     return pd.DataFrame(results)
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        # 1) 원본 index, id 보존
        data = data.reset_index()            # index 컬럼 새로 생김
        all_fixed = [""] * len(data)         # 자리 고정 리스트

        # 2) 배치 생성
        batches = [data.iloc[i:i+BATCH_SIZE]
                for i in range(0, len(data), BATCH_SIZE)]

        with ThreadPoolExecutor(MAX_WORKERS) as pool, \
            tqdm(total=len(batches), desc="Batch", ncols=80) as bar:

            futs = {pool.submit(self._handle_batch, b): b for b in batches}

            for fut in as_completed(futs):
                batch_df = futs[fut]
                fixed_list = fut.result()
                # 3) 각 문장을 'index' 위치에 삽입
                for orig_idx, sent in zip(batch_df["index"], fixed_list):
                    all_fixed[orig_idx] = sent
                bar.update(1)

        return pd.DataFrame({
            "id": data["id"],
            "cor_sentence": all_fixed
        })
