# recommender — Claude 작업 메모

배치 추천 파이프라인. 실시간 서빙은 `recsys-serving` 담당. 이 repo는 글로벌 풀 빌드, bandit reconcile, user vector EMA, 메트릭 rollup 만 한다.

## 한 줄 요약

`recommendation_global` (글로벌 풀) + `member_category_bandit` (Beta TS state) + Qdrant `user_profile` (Phase 2) 를 매일 오버라이트로 갱신한다. 서빙은 이 셋만 읽는다.

---

## Phase 로드맵

### Phase 1 — Onboarding-Prior + Per-Category Thompson Sampling (현재)

- 글로벌 풀: `freshness × quality × popularity`. 카테고리 라벨 포함.
- Bandit: 카테고리 단위 Beta(α, β). onboarding 선택 카테고리는 `Beta(4, 1)`, 미선택은 `Beta(1, 2)`.
- 서빙: TS sampling → 카테고리 quota 분배 → 글로벌 풀에서 카테고리별 top-K 추출.
- Reward: `like/bookmark/click → α += 1`, `share → α += 2`, `uninterest → β += 2`, `impression-no-engagement-24h → β += 0.1`.
- 배치 reconcile은 daily ground-truth overwrite. 서빙 측 실시간 update는 incremental.

### Phase 2 — Embedding-based User Vector + Within-Category Personalization

- Qdrant `user_profile` collection (1024d, Qwen3-Embedding 차원). `point_id = member_id`.
- 클릭한 article 임베딩의 시간 가중 EMA로 user vector 빌드. Decay = `0.95^days_ago`.
- 서빙: 카테고리 quota는 TS 그대로 → 각 카테고리 안에서 `user_profile` 코사인 유사도로 within-category rerank.
- 클릭 이력 0인 유저는 user_vector 없음 → 글로벌 score fallback (자동 전이).
- 실시간 update: `POST /recommend/feedback` 경로에서 클릭한 article 임베딩으로 EMA push.

### Phase 3 — Contextual Bandit (미래)

- LinUCB 또는 neural reranker. Feature: user_vec + article_vec + category one-hot + freshness + popularity.
- 데이터 충분히 쌓일 때 도입. 지금은 schema/계획만.

---

## 데이터 contract (서빙과 공유)

### 입력 (recommender가 읽는 것)
- `article` — published_at, category_id, quality_score, like/share/bookmark count
- `user_events` — bandit reconcile, popularity 신호, EMA 입력
- `recommendation_impression` — bandit β reconcile (impression 후 click 없음)
- `member_interest` — bandit prior 초기화
- Qdrant `bite-vectordb` — article 임베딩 (Phase 2 user vector 빌드)

### 출력 (서빙이 읽는 것)
- `recommendation_global` — 글로벌 풀, 매일 atomic swap (TRUNCATE + INSERT)
- `member_category_bandit` — daily ground-truth overwrite (실시간 incremental과 race 무시 — 다음 배치가 다시 ground-truth)
- Qdrant `user_profile` collection — Phase 2 user vector
- `recommendation_metric_daily` — 일별 KPI (CTR, per-category, diversity, freshness)
- `bandit_state_snapshot` — 일별 bandit α/β trajectory

---

## 다이어트 결정 (2026-04-30)

이전 segment-based 파이프라인은 **유저 0인 상태에서 동작 불가** (user_events / behavior embedding 의존). 카테고리 prior + bandit 으로 재설계하면서 다음을 제거했다:

- `data/seg_routing.py`, `data/candidate_builder.py`, `data/candidate_merge.py`
- `data/pipeline/initial_embedding.py` (FastText 한국어 mini)
- `data/pipeline/behavior_embedding.py`, `data/pipeline/mix_user_embedding.py`
- `data/pipeline/user_features.py`, `data/pipeline/fetch_data.py`
- `ranker/scoring.py` (sim 0.7 + freshness 0.15 + popularity 0.10 + diversity 0.05 가중치 — 카테고리 quota로 대체)
- `metrics/recall_calculator.py`, `metric_load.py` (Phase 1은 CTR로 평가, recall@k는 Phase 3에서 재도입 가능)
- `utils/candidate_save.py` (per-member `recommendation` 테이블 → `recommendation_global` 글로벌 풀로 교체)

**유지 (재활용)**: `common/db.py`, `utils/{config_loader,logger,metric_sink}.py`, `data/pipeline/{popularity,article_metrics,engagement_aggregator}.py`.

---

## 실행 위치 (2026-04-30 결정)

**Phase 1+2 정식 실행 위치는 Mac mini의 docker compose batch profile.** 매일 cron 트리거.

```
docker compose --profile batch run --rm recommender
```

이 워크로드는 SQL 집계 + numpy 가중평균 + Qdrant retrieve/upsert 만 한다. GPU 불필요. CPU only 의존성 (numpy, polars, sqlalchemy, qdrant-client) 유지.

GPU 서버에 있던 `~/recommender` conda + cron 운영 흔적은 deprecate. 매일 돌아야 하는 배치를 WoL 켰다 끄는 GPU 서버에 의존시키는 건 stale 위험.

### Phase 3 (종착지) 와의 분리 원칙

종착지: **two-tower candidate gen + ranking model**. 그러나 이 학습 컴포넌트는 **이 repo 안에 같이 두지 않는다**. 다음 분리를 유지:

- **training (GPU 서버, 주간/일간)**: two_tower train, item/user embedding refresh, ranker train. 미래 별도 디렉토리 `training/` 또는 신규 repo. PyTorch + CUDA 의존성은 여기에만.
- **daily orchestration (이 repo, Mac mini)**: 글로벌 풀, bandit reconcile, EMA, metric rollup. CPU 의존성만.
- **online serving (recsys-serving, Mac mini)**: 학습 산출물(Qdrant 임베딩, ONNX weight)만 읽음. GPU 의존성 0.

학습 산출물 배포 채널:
- 임베딩 (item/user vec): Qdrant 에 직접 upsert (별도 채널 X).
- 모델 weight (ranker): ONNX export → docker image 안에 빌드 시 동봉 (현 reranker 패턴).
- 얇은 hyperparameter (bandit prior, blend ratio): Doppler config.

이 분리 덕분에 학습 실패가 daily 배치를 막지 않고, recsys-serving 은 GPU 서버 가용성과 무관하게 계속 동작한다.

### Qdrant collection 명명 규약 (Phase 3 대비)

- `bite-vectordb` — harvest_post 가 채우는 article 본문 임베딩 (Qwen3, 1024d). Phase 2 EMA 입력.
- `user_profile` — Phase 2 user vector EMA (recommender 배치 + recsys-serving 실시간 push). 1024d.
- (미래) `bite-tt-item` / `bite-tt-user` — two-tower 학습 결과. 차원은 학습 시점 결정.

recsys-serving 의 retrieval strategy 는 collection 우선순위 (`bite-tt-*` 있으면 우선, 없으면 `bite-vectordb` + `user_profile` fallback) 로 두면 Phase 3 도입 시 서빙 코드 변경 최소화.

## Pipeline stage (main.py)

순서대로:
1. **`global_ranking.build_pool`** — article 테이블 + popularity + quality → recommendation_global 통째 swap.
2. **`engagement_aggregator.aggregate_engagement`** — user_events → user_article_engagement upsert (기존 유지).
3. **`bandit_reconcile.reconcile`** — user_events ↔ recommendation_impression join → member_category_bandit overwrite.
4. **`user_vector.build_profiles`** (Phase 2) — 클릭한 article 임베딩 EMA → Qdrant user_profile upsert.
5. **`metric_rollup.rollup`** — recommendation_metric_daily + bandit_state_snapshot upsert.

각 stage는 자기 시간/에러를 `MetricSink`로 `recommender_run_metric`에 기록한다 (기존 유지).

---

## ⚠️ 함정

- **`rank` 컬럼명 금지** — MySQL 8.0 reserved word. `rank_global` 사용.
- **bandit reconcile은 ground-truth overwrite**. 서빙 incremental update와 race 가능하지만 무시 — 다음 배치가 정정. 만약 batch 도중 race로 데이터 망가지면 batch 트랜잭션 안에서 `LOCK TABLES` 까지 갈 수도 있는데 현재는 over-engineering.
- **글로벌 풀 swap 은 TRUNCATE + INSERT 단일 트랜잭션**. 서빙이 빈 풀을 보면 안 됨.
- **Qdrant `user_profile` collection 차원**은 article 임베딩과 같아야 함 (현재 1024). 다르면 Phase 2 within-category rerank가 cosine 못 함.
- **유저 0인 환경**: popularity / engagement / user_vector 모두 빈 값 → 글로벌 풀은 `freshness × quality` 만으로 동작. Phase 2 stage도 빈 결과 반환 (정상).
- **카테고리 NULL 글**: 글로벌 풀에는 안 들어감 (`category_id IS NOT NULL` 필터). bandit이 카테고리 단위라 매칭 불가.

---

## 검증 / 운영

- 한 사이클 수동 실행: `doppler run -- uv run python main.py` (cwd = repo root, config 상대경로 때문).
- 빠른 sanity: `SELECT COUNT(*), MIN(rank_global), MAX(rank_global) FROM recommendation_global;`
- bandit 분포: `SELECT category_id, AVG(alpha/(alpha+beta)) FROM member_category_bandit GROUP BY category_id;`
- 일별 KPI: `SELECT * FROM recommendation_metric_daily ORDER BY metric_date DESC LIMIT 7;`

## Doppler

- 프로젝트: `recommender`. config는 `prd` / `dev`.
- 비밀: RDS_*, QDRANT_ENDPOINT, QDRANT_API.

## 한 줄 정리

이 repo는 **상태 만드는 곳**, 서빙은 **상태 읽는 곳**. cross-service 변경 시 contract (recommendation_global / member_category_bandit / user_profile) 가 깨지지 않게.
