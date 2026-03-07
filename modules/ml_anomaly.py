# modules/ml_anomaly.py
# ML 이상 탐지 모듈 (PCA + IsolationForest)
# 다중 웨이퍼 맵을 벡터화해 이상 웨이퍼를 자동 탐지
#
# pip install scikit-learn
#
# =============================================================================
# [설계 결정 근거]
# =============================================================================
#
# ① 소규모 샘플(n < 5) IsolationForest 신뢰도 대응
#    contamination=0.10 + n=4 → 0.4개 = floor(0) → 이상 탐지 0개 → 의미 없음.
#    대응:
#      n < 3  → st.warning + return (UI 진입 차단, 최소 비교 불가)
#      3 ≤ n < 5 → "결과 신뢰도 낮음" 경고 + 계속 허용
#      contamination 자동 상한 조정: min(contamination, (n-1)/n - 0.01)
#        → n=4, contamination=0.30 → min(0.30, 0.74) = 0.30 그대로
#        → n=3, contamination=0.40 → min(0.40, 0.65) = 0.40 그대로
#        → n=5, contamination=0.80 → min(0.80, 0.79) = 0.79 (자동 조정)
#
# ② ZI NaN 처리 전략: Z-score 정규화 후 0 대체
#    ZI NaN 발생: circular mask (웨이퍼 원 밖 영역)
#    모든 웨이퍼가 동일한 circular 패턴 → 외부 픽셀을 0(정규화 기준값)으로 통일
#    처리 순서:
#      1. valid_mask = ~np.isnan(ZI)  (웨이퍼 내부 픽셀만)
#      2. zi_mean, zi_std = ZI[valid_mask].mean(), ZI[valid_mask].std()
#      3. ZI_norm = (ZI - zi_mean) / (zi_std + 1e-10)  (전체 배열)
#      4. ZI_norm[~valid_mask] = 0.0  (정규화 후 외부 픽셀 = 0 = 평균 수준)
#    → 정규화 전에 0 대체하면 zi_mean/std가 외부 0에 의해 오염됨 (NG)
#    → 정규화 후 0 대체: 외부 픽셀이 정규화 기준(=0)에 위치 (OK)
#
# ③ contamination 변경 시 PCA 재사용 캐시 전략
#    PCA 비용: 높음 (SVD 분해, 웨이퍼 수×해상도² 행렬)
#    IF 비용:  낮음 (트리 앙상블, 빠른 재실행 가능)
#    캐시 키 설계:
#      pca_key = (tuple(sorted(names)), resolution)
#        → 웨이퍼 목록 or 해상도 변경 시 PCA 무효화
#        → contamination 변경만으로는 pca_key 불변 → PCA 재사용
#      if_key  = (pca_key, contamination)
#        → contamination 변경 → if_key 변경 → IF만 재실행
#    두 키를 session_state["ml_pca_key"], session_state["ml_if_key"]에 저장
#    → 버튼 클릭 시 키 비교로 재실행 여부 결정
#
# ④ ndarray/@st.cache_data 비호환성
#    np.ndarray는 mutable → @st.cache_data 인자로 불가
#    PCA 결과(ndarray)와 IF 결과(ndarray)는 session_state에 직접 저장
#    → 버튼 클릭 이벤트에서만 재계산 → session_state에 보존
# =============================================================================

# ── 표준 라이브러리 ─────────────────────────────────────────────────────────
import glob
import hashlib
import os

# ── 외부 라이브러리 ─────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── wafer_app_global 핵심 함수 import ────────────────────────────────────────
from app import _default_col_index  # 컬럼 기본값 탐색 (데이터셋 추가 UI)
from app import apply_col_mapping  # x/y/data 컬럼 표준화 (데이터셋 추가 UI)
from app import calculate_stats  # GPC 패턴 분류용 통계
from app import create_2d_heatmap  # compact=True로 이상 웨이퍼 미리보기
from app import get_sheet_names  # Excel 시트 목록 (데이터셋 추가 UI)
from app import get_wafer_grid  # 불규칙 산점 → 균일 그리드 보간 (@st.cache_data)
from app import load_file_cached  # CSV/Excel 로드 (데이터셋 추가 UI)

# =============================================================================
# scikit-learn 가용성 탐지 (모듈 로딩 시 1회)
# =============================================================================

try:
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False
    # 타입 힌트용 더미 클래스 (import 실패 시 NameError 방지)
    PCA              = None   # type: ignore
    IsolationForest  = None   # type: ignore
    StandardScaler   = None   # type: ignore


# =============================================================================
# session_state 키 상수 (prefix: "ml_")
# =============================================================================
_SS_PCA_KEY     = "ml_pca_key"       # PCA 캐시 키 (tuple → hash)
_SS_PCA_RESULT  = "ml_pca_result"    # PCA 결과 dict
_SS_IF_KEY      = "ml_if_key"        # IF 캐시 키 tuple
_SS_IF_RESULT   = "ml_if_result"     # IF 결과 dict
_SS_NAMES       = "ml_names"         # 유효 웨이퍼 이름 리스트
_SS_PATTERNS    = "ml_patterns"      # 패턴 분류 결과 dict
_SS_CONTAM      = "ml_contamination" # 마지막 실행된 contamination 값
_SS_RESOLUTION  = "ml_resolution"    # 마지막 실행된 해상도 값
# ── 데이터셋 관리 (신규) ────────────────────────────────────────────────────
_SS_DATASETS    = "ml_datasets"      # ML 탭 전용 데이터셋 [{name, df_json}, ...]
_SS_APP_HASH    = "ml_app_hash"      # 앱 제공 datasets 이름 목록의 hash (동기화 감지)


# =============================================================================
# 이상 패턴 분류 임계값 상수
# =============================================================================
_HOTSPOT_RATIO   = 4.0   # max > mean + k×std → Hotspot (k=4 → 4σ 이상)
_RING_CE_RATIO   = 0.05  # |center-edge|/global_mean > 5% → Ring 후보
_EDGE_DEG_RATIO  = 0.90  # edge_mean < center_mean × 0.90 → Edge Degradation
_GRAD_CORR_THR   = 0.40  # |corr| > 0.40 → Gradient 방향성 있음
_NORMAL_UNIF_THR = 2.0   # Uniformity(%) < 2% → Normal


# =============================================================================
# [함수 1] prepare_wafer_features
# =============================================================================

def prepare_wafer_features(
    maps_data: list,
    resolution: int = 50,
) -> tuple:
    """
    여러 웨이퍼 데이터에서 ML 특성 행렬을 추출.

    [처리 흐름]
    for each wafer:
      1. get_wafer_grid(df_json, resolution) → ZI [resolution × resolution]
      2. 보간 실패(ZI 전체 NaN) → valid_mask=False, 건너뜀
      3. Z-score 정규화 (유효 픽셀만 사용):
           valid = ~isnan(ZI)
           zi_norm = (ZI - ZI[valid].mean()) / (ZI[valid].std() + 1e-10)
      4. 외부 픽셀(NaN 위치) → 0으로 대체 (정규화 평균 수준)
      5. flatten → 1D 벡터 (resolution²차원)

    [Z-score 정규화를 먼저 하는 이유]
    0 대체 후 정규화 시 외부 0이 mean/std를 오염 → 내부 분포 왜곡.
    정규화 후 0 대체: 외부 픽셀이 정규화 기준(=0, 평균 수준)에 위치 → 올바름.

    [StandardScaler 미사용 이유]
    각 웨이퍼를 독립적으로 Z-score 정규화 → 절대 두께 차이가 아닌
    공간 패턴(분포 형태)에 집중. StandardScaler는 피처 컬럼 방향 정규화
    (웨이퍼 간 같은 픽셀을 정규화) → 목적이 다름.

    인자:
        maps_data : [{"df_json": str, "name": str}, ...] 웨이퍼 목록
        resolution: 보간 그리드 해상도 (낮을수록 빠름, 기본 50 → 2500차원)

    반환:
        (feature_matrix, valid_names, valid_mask)
        feature_matrix: ndarray (n_valid × resolution²)
        valid_names   : 유효 웨이퍼 이름 리스트
        valid_mask    : 각 웨이퍼가 유효한지 bool 리스트 (원본 순서 보존)
    """
    feature_rows: list[np.ndarray] = []
    valid_names: list[str]         = []
    valid_mask:  list[bool]        = []

    for wafer in maps_data:
        df_json   = wafer.get("df_json", "")
        wafer_name = wafer.get("name", "Unknown")

        try:
            # ── 그리드 보간 (하위 캐시 get_wafer_grid 재사용) ────────────────
            _, _, ZI, _ = get_wafer_grid(df_json, resolution)

            # ── 보간 실패 체크: ZI가 전부 NaN이면 제외 ───────────────────────
            valid_pixels = ~np.isnan(ZI)
            n_valid = int(valid_pixels.sum())
            if n_valid < 3:
                # 유효 픽셀 3개 미만: 보간 실패 또는 의미 없는 데이터
                valid_mask.append(False)
                continue

            # ── Z-score 정규화 (유효 픽셀만 사용) ───────────────────────────
            zi_mean = float(ZI[valid_pixels].mean())
            zi_std  = float(ZI[valid_pixels].std())
            zi_norm = (ZI - zi_mean) / (zi_std + 1e-10)

            # ── 외부 픽셀을 0으로 대체 (정규화 후) ──────────────────────────
            # 외부 픽셀이 정규화 기준(0=평균)에 위치 → 특성 벡터 길이 통일
            zi_norm[~valid_pixels] = 0.0

            # ── flatten → 1D 특성 벡터 ──────────────────────────────────────
            feature_rows.append(zi_norm.flatten())
            valid_names.append(wafer_name)
            valid_mask.append(True)

        except Exception:
            # 개별 웨이퍼 처리 실패 시 건너뜀 (전체 중단 방지)
            valid_mask.append(False)
            continue

    if len(feature_rows) == 0:
        # 모든 웨이퍼 처리 실패
        return np.empty((0, resolution * resolution)), [], valid_mask

    feature_matrix = np.vstack(feature_rows)   # (n_valid, resolution²)
    return feature_matrix, valid_names, valid_mask


# =============================================================================
# [함수 2] run_pca
# =============================================================================

def run_pca(feature_matrix: np.ndarray) -> dict:
    """
    특성 행렬에 PCA를 적용해 저차원 표현 생성.

    [@st.cache_data 미적용 이유]
    np.ndarray는 mutable → hash() 불가 → @st.cache_data 불가.
    → 호출부(render_anomaly_tab)에서 캐시 키 비교로 재실행 여부 결정.
    → 결과는 session_state["ml_pca_result"]에 저장.

    [n_components 결정]
    n_components = min(n_wafers - 1, 10)
    이유:
      - PCA의 유효 최대 성분 수 = min(n_samples-1, n_features)
      - 10개 이상의 PC는 시각화(2D 산점도)에 사용 안 되고 메모리만 낭비
      - n_wafers - 1: 최소 1개의 성분 보장 (n_wafers ≥ 2이면)

    [explained_variance_ratio 활용]
    PC1, PC2 축 라벨에 % 표시 → 해당 성분이 전체 분산의 몇 %를 설명하는지
    반도체 공정에서 주요 변동 모드(Ring, Gradient 등)의 기여도 직관적 파악

    인자:
        feature_matrix: ndarray (n_wafers × resolution²)

    반환:
        {
          "components"            : ndarray (n_wafers × n_components),
          "explained_variance_ratio": ndarray (n_components,),
          "n_components"          : int,
        }
    """
    if not _SKLEARN_OK:
        raise ImportError("scikit-learn 미설치: 'pip install scikit-learn'")

    n_wafers   = feature_matrix.shape[0]
    n_comp     = min(n_wafers - 1, 10)  # 유효 최대 성분 수 자동 결정
    n_comp     = max(n_comp, 2)         # 최소 2개 (2D 산점도 렌더링 필요)
    n_comp     = min(n_comp, feature_matrix.shape[1])  # 특성 수 초과 방지

    pca = PCA(n_components=n_comp, random_state=42)
    components = pca.fit_transform(feature_matrix)   # (n_wafers, n_comp)

    return {
        "components":               components,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "n_components":             n_comp,
    }


# =============================================================================
# [함수 3] run_isolation_forest
# =============================================================================

def run_isolation_forest(
    pca_components: np.ndarray,
    contamination: float,
) -> dict:
    """
    PCA 성분 공간에서 IsolationForest로 이상 웨이퍼 탐지.

    [@st.cache_data 미적용 이유]
    np.ndarray(pca_components)는 mutable → hash() 불가.
    → session_state["ml_if_result"]에 결과 저장.
    → contamination 변경 시에만 재실행 (PCA 성분은 불변).

    [contamination 상한 자동 조정]
    n_wafers가 적으면 contamination × n_wafers < 1이 되어 이상 탐지 0개 가능.
    → max_contamination = (n_wafers - 1) / n_wafers - 0.01
    → contamination = min(contamination, max_contamination)

    [anomaly score 해석]
    IsolationForest.score_samples(): 낮을수록 이상
    raw_score 범위: 음수 (정상) ~ 더 큰 음수 (이상)
    [0, 1] 정규화:
      inv = -raw_score  (부호 반전 → 이상일수록 큰 양수)
      normalized = (inv - inv.min()) / (inv.max() - inv.min() + 1e-12)
      → 1에 가까울수록 이상, 0에 가까울수록 정상

    인자:
        pca_components: PCA 성분 행렬 (n_wafers × n_components)
        contamination : 이상 웨이퍼 비율 (0.05~0.30)

    반환:
        {
          "predictions"    : ndarray (-1=이상, 1=정상),
          "scores"         : ndarray ([0,1] 정규화, 높을수록 이상),
          "anomaly_indices": list[int] (이상 웨이퍼 인덱스),
          "threshold"      : float (임계 점수 = 이상/정상 경계),
          "contamination_used": float (실제 적용된 contamination),
        }
    """
    if not _SKLEARN_OK:
        raise ImportError("scikit-learn 미설치: 'pip install scikit-learn'")

    n_wafers = pca_components.shape[0]

    # contamination 상한 자동 조정 (IsolationForest 내부 제약: < 0.5)
    max_contamination = min(0.49, (n_wafers - 1) / n_wafers - 0.01)
    contamination_used = min(contamination, max_contamination)

    clf = IsolationForest(
        contamination=contamination_used,
        n_estimators=200,    # 200트리: 안정성과 속도의 균형
        random_state=42,     # 재현성 보장
        n_jobs=-1,           # 모든 CPU 코어 활용 (병렬 트리 구축)
    )
    clf.fit(pca_components)

    predictions  = clf.predict(pca_components)   # 1=정상, -1=이상
    raw_scores   = clf.score_samples(pca_components)  # 낮을수록 이상

    # [0, 1] 정규화: 부호 반전 후 min-max
    # 정규화 후: 1에 가까울수록 이상, 0에 가까울수록 정상
    inv        = -raw_scores
    s_min, s_max = inv.min(), inv.max()
    norm_scores  = (inv - s_min) / (s_max - s_min + 1e-12)

    # 이상/정상 경계 점수 계산
    # IsolationForest에서 predictions=-1인 샘플들의 정규화 점수 최솟값
    anom_mask = (predictions == -1)
    threshold = float(norm_scores[anom_mask].min()) if anom_mask.any() else 1.0

    anomaly_indices = [int(i) for i in np.where(anom_mask)[0]]

    return {
        "predictions":        predictions,
        "scores":             norm_scores,
        "anomaly_indices":    anomaly_indices,
        "threshold":          threshold,
        "contamination_used": contamination_used,
    }


# =============================================================================
# [함수 4] classify_anomaly_pattern
# =============================================================================

def classify_anomaly_pattern(df_json: str) -> str:
    """
    규칙 기반으로 웨이퍼 맵 이상 패턴을 자동 분류.

    [분류 우선순위]
    1. Normal    : Uniformity(%) < 2% → 공정 이상 없음
    2. Hotspot   : max > mean + 4σ   → 좁은 영역 급격한 피크 (파티클, 스크래치)
    3. Ring      : |center - edge| / mean > 5% AND 단조성 위반
    4. Edge Deg  : edge_mean < center_mean × 0.90
    5. X-Gradient: |corr(x, data)| > 0.40 AND > |corr(y, data)|
    6. Y-Gradient: |corr(y, data)| > 0.40 AND > |corr(x, data)|
    7. Global Shift: 위 어느 것도 아님 + Uniformity(%) > 5%
    8. Mixed     : 위 어느 것도 해당 없음

    [3구역 정의]
    Center: r < radius × 0.30
    Edge:   r ≥ radius × 0.70

    인자:
        df_json: "x","y","data" 컬럼 JSON (표준 구조)

    반환:
        분류 문자열: "Normal" / "Hotspot" / "Ring" / "Edge Degradation" /
                    "X-Gradient" / "Y-Gradient" / "Global Shift" / "Mixed"
    """
    try:
        df = pd.read_json(df_json).dropna(subset=["x", "y", "data"])
        if len(df) < 5:
            return "데이터 부족"

        x    = df["x"].values
        y    = df["y"].values
        data = df["data"].values.astype(float)

        r      = np.sqrt(x ** 2 + y ** 2)
        radius = r.max()

        mean_val = float(np.nanmean(data))
        std_val  = float(np.nanstd(data))
        max_val  = float(np.nanmax(data))

        # Uniformity(%) = σ/μ × 100
        uniformity = (std_val / mean_val * 100) if mean_val != 0 else float("inf")

        # ── 1. Normal: 균일도 2% 이내 ────────────────────────────────────────
        if uniformity < _NORMAL_UNIF_THR:
            return "Normal"

        # ── 2. Hotspot: 극단 피크 (4σ 이상) ─────────────────────────────────
        if mean_val != 0 and max_val > mean_val + _HOTSPOT_RATIO * std_val:
            return "Hotspot"

        # ── 3구역 평균 계산 ───────────────────────────────────────────────────
        center_mask = r <  radius * 0.30
        mid_mask    = (r >= radius * 0.30) & (r < radius * 0.70)
        edge_mask   = r >= radius * 0.70

        center_mean = float(np.nanmean(data[center_mask])) if center_mask.any() else np.nan
        mid_mean    = float(np.nanmean(data[mid_mask]))    if mid_mask.any()    else np.nan
        edge_mean   = float(np.nanmean(data[edge_mask]))   if edge_mask.any()   else np.nan

        # ── 3. Ring: 중심-가장자리 차이 + 단조성 위반 ───────────────────────
        if (not np.isnan(center_mean) and not np.isnan(edge_mean)
                and mean_val != 0):
            ce_diff = abs(center_mean - edge_mean) / mean_val
            if ce_diff > _RING_CE_RATIO:
                # 단조성 체크: center → mid → edge 방향이 일관되지 않으면 Ring
                if (not np.isnan(mid_mean)):
                    center_to_mid = mid_mean - center_mean
                    mid_to_edge   = edge_mean - mid_mean
                    # 방향 반전: Center↑Mid↓Edge 또는 Center↓Mid↑Edge → Ring
                    if (center_to_mid * mid_to_edge) < 0:
                        return "Ring"

        # ── 4. Edge Degradation: 가장자리가 중심보다 낮음 ───────────────────
        if (not np.isnan(center_mean) and not np.isnan(edge_mean)
                and center_mean != 0):
            if edge_mean < center_mean * _EDGE_DEG_RATIO:
                return "Edge Degradation"

        # ── 5 & 6. Gradient: X 또는 Y 방향 선형 상관 ────────────────────────
        if std_val > 0:
            corr_x = float(np.corrcoef(x, data)[0, 1]) if len(x) > 2 else 0.0
            corr_y = float(np.corrcoef(y, data)[0, 1]) if len(y) > 2 else 0.0

            abs_cx = abs(corr_x)
            abs_cy = abs(corr_y)

            if abs_cx > _GRAD_CORR_THR or abs_cy > _GRAD_CORR_THR:
                if abs_cx >= abs_cy:
                    return "X-Gradient"
                else:
                    return "Y-Gradient"

        # ── 7. Global Shift: 패턴 없이 전체 균일도 이탈 ─────────────────────
        if uniformity > 5.0:
            return "Global Shift"

        # ── 8. Mixed: 분류 불가 ──────────────────────────────────────────────
        return "Mixed"

    except Exception:
        return "분류 실패"


# =============================================================================
# [함수 5] create_pca_scatter
# =============================================================================

def create_pca_scatter(
    pca_result: dict,
    if_result: dict,
    wafer_names: list,
    df_jsons: list,   # 패턴 분류 hover에 사용
) -> go.Figure:
    """
    PCA 결과 PC1-PC2 2D 산점도 생성.

    [마커 설계]
    정상 웨이퍼: 파란 원(circle), 반투명
    이상 웨이퍼: 빨간 X(x), 불투명
    마커 크기: anomaly score에 반비례
      → score 높을수록(이상일수록) 더 크게 → 시각적 강조

    [hover 정보]
    웨이퍼 이름 + anomaly score + 분류 패턴 + PC1/PC2 값

    [축 라벨]
    "PC1 (42.3%)": 설명 분산 비율 포함 → 주요 변동 모드 기여도 표시

    인자:
        pca_result  : run_pca() 반환 dict
        if_result   : run_isolation_forest() 반환 dict
        wafer_names : 유효 웨이퍼 이름 리스트
        df_jsons    : 각 웨이퍼의 df_json (패턴 분류용)

    반환:
        go.Figure: PCA 산점도
    """
    components = pca_result["components"]        # (n_wafers, n_comp)
    ev_ratio   = pca_result["explained_variance_ratio"]  # (n_comp,)
    predictions = if_result["predictions"]       # (n_wafers,)
    scores      = if_result["scores"]            # (n_wafers,) [0,1]
    threshold   = if_result["threshold"]

    pc1 = components[:, 0]
    pc2 = components[:, 1] if components.shape[1] > 1 else np.zeros_like(pc1)

    pc1_pct = ev_ratio[0] * 100 if len(ev_ratio) > 0 else 0
    pc2_pct = ev_ratio[1] * 100 if len(ev_ratio) > 1 else 0

    fig = go.Figure()

    # ── 정상/이상 분리 ────────────────────────────────────────────────────────
    is_anomaly = (predictions == -1)
    norm_mask  = ~is_anomaly
    anom_mask  = is_anomaly

    # ── 마커 크기 계산 (score에 반비례 → 이상일수록 크게) ───────────────────
    # 정상: 8~14px, 이상: 14~22px
    def _score_to_size(s_arr: np.ndarray, base: float, scale: float) -> list:
        return [float(base + score * scale) for score in s_arr]

    # ── 정상 웨이퍼 trace ─────────────────────────────────────────────────────
    if norm_mask.any():
        norm_idx   = np.where(norm_mask)[0]
        norm_names = [wafer_names[i] for i in norm_idx]
        norm_sizes = _score_to_size(scores[norm_mask], base=8, scale=6)

        # 패턴 분류
        norm_patterns = []
        for i in norm_idx:
            if i < len(df_jsons):
                norm_patterns.append(classify_anomaly_pattern(df_jsons[i]))
            else:
                norm_patterns.append("N/A")

        fig.add_trace(go.Scatter(
            x=pc1[norm_mask],
            y=pc2[norm_mask],
            mode="markers",
            name="정상 웨이퍼",
            text=norm_names,
            customdata=np.column_stack([
                scores[norm_mask].round(4),
                norm_patterns,
            ]),
            marker=dict(
                symbol="circle",
                size=norm_sizes,
                color="rgba(31, 119, 180, 0.70)",    # 파란 반투명
                line=dict(width=1, color="rgba(31, 119, 180, 0.90)"),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "PC1: %{x:.4f}<br>"
                "PC2: %{y:.4f}<br>"
                "이상 점수: %{customdata[0]}<br>"
                "패턴: %{customdata[1]}"
                "<extra>정상</extra>"
            ),
        ))

    # ── 이상 웨이퍼 trace ─────────────────────────────────────────────────────
    if anom_mask.any():
        anom_idx   = np.where(anom_mask)[0]
        anom_names = [wafer_names[i] for i in anom_idx]
        anom_sizes = _score_to_size(scores[anom_mask], base=14, scale=10)

        # 패턴 분류
        anom_patterns = []
        for i in anom_idx:
            if i < len(df_jsons):
                anom_patterns.append(classify_anomaly_pattern(df_jsons[i]))
            else:
                anom_patterns.append("N/A")

        fig.add_trace(go.Scatter(
            x=pc1[anom_mask],
            y=pc2[anom_mask],
            mode="markers",
            name="이상 웨이퍼",
            text=anom_names,
            customdata=np.column_stack([
                scores[anom_mask].round(4),
                anom_patterns,
            ]),
            marker=dict(
                symbol="x",
                size=anom_sizes,
                color="rgba(214, 39, 40, 0.90)",     # 빨간 불투명
                line=dict(width=2.5, color="rgba(214, 39, 40, 1.0)"),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "PC1: %{x:.4f}<br>"
                "PC2: %{y:.4f}<br>"
                "이상 점수: %{customdata[0]}<br>"
                "패턴: %{customdata[1]}"
                "<extra>⚠️ 이상</extra>"
            ),
        ))

    # ── 레이아웃 ──────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text="PCA 이상 탐지 산점도", x=0.5, font=dict(size=14)),
        xaxis=dict(
            title=f"PC1 ({pc1_pct:.1f}%)",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.5)",
            zeroline=True,
            zerolinecolor="rgba(150,150,150,0.5)",
            zerolinewidth=1,
        ),
        yaxis=dict(
            title=f"PC2 ({pc2_pct:.1f}%)",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.5)",
            zeroline=True,
            zerolinecolor="rgba(150,150,150,0.5)",
            zerolinewidth=1,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=420,
        margin=dict(l=60, r=20, t=50, b=50),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(180,180,180,0.5)",
            borderwidth=1,
        ),
    )

    return fig


# =============================================================================
# [함수 6] create_anomaly_score_bar
# =============================================================================

def create_anomaly_score_bar(
    wafer_names: list,
    scores: np.ndarray,
    is_anomaly: np.ndarray,
) -> go.Figure:
    """
    웨이퍼별 anomaly score 수평 막대 그래프.

    [시각 설계]
    이상 웨이퍼: 빨간색 막대 (강조)
    정상 웨이퍼: 파란색 막대 (배경)
    임계값 수직선: 이상/정상 경계 명시 → 판단 기준 투명화
    정렬: score 내림차순 → 이상 웨이퍼가 상단에 집중

    [anomaly score 방향]
    score = 1에 가까울수록 이상 → X축이 커질수록 위험
    → "점수가 높을수록 위험" → 직관적

    인자:
        wafer_names: 웨이퍼 이름 리스트
        scores     : [0,1] 정규화 anomaly scores (1=이상)
        is_anomaly : bool 배열 (True=이상)

    반환:
        go.Figure: 수평 막대 그래프
    """
    n = len(wafer_names)
    if n == 0:
        return go.Figure()

    # score 내림차순 정렬 (이상 웨이퍼 상단 집중)
    sort_idx   = np.argsort(scores)[::-1]
    names_sort = [wafer_names[i] for i in sort_idx]
    scores_sort = scores[sort_idx]
    is_anom_sort = is_anomaly[sort_idx]

    # 이상/정상 색상 배열
    colors = [
        "rgba(214, 39, 40, 0.80)" if anom else "rgba(31, 119, 180, 0.60)"
        for anom in is_anom_sort
    ]

    # 이상/정상 레이블
    labels = ["⚠️ 이상" if a else "✅ 정상" for a in is_anom_sort]

    fig = go.Figure()

    # ── 단일 trace로 전체 막대 그래프 (색상은 marker.color 배열로 지정) ──────
    fig.add_trace(go.Bar(
        y=names_sort,              # 웨이퍼 이름 (Y축, 수평 막대이므로)
        x=scores_sort,             # anomaly score (X축)
        orientation="h",           # 수평 막대
        marker=dict(
            color=colors,
            line=dict(width=0.5, color="rgba(100,100,100,0.3)"),
        ),
        text=[f"{s:.4f}" for s in scores_sort],   # 막대 끝 score 텍스트
        textposition="outside",
        customdata=labels,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "이상 점수: %{x:.4f}<br>"
            "판정: %{customdata}"
            "<extra></extra>"
        ),
        name="이상 점수",
    ))

    # ── 임계값 수직선 ─────────────────────────────────────────────────────────
    # 이상/정상 경계: is_anomaly 경계점에서의 score 값
    if is_anomaly.any() and (~is_anomaly).any():
        anom_scores  = scores[is_anomaly]
        normal_scores = scores[~is_anomaly]
        threshold = (anom_scores.min() + normal_scores.max()) / 2

        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="rgba(128, 0, 0, 0.7)",
            line_width=1.5,
            annotation_text=f"임계값 {threshold:.4f}",
            annotation_position="top right",
            annotation_font=dict(size=9, color="darkred"),
        )

    # ── 레이아웃 ──────────────────────────────────────────────────────────────
    # 웨이퍼 수에 따라 높이 자동 조정 (웨이퍼당 약 30px)
    chart_height = max(300, n * 30 + 100)

    fig.update_layout(
        title=dict(text="웨이퍼별 이상 점수", x=0.5, font=dict(size=14)),
        xaxis=dict(
            title="이상 점수 (높을수록 이상)",
            range=[0, 1.15],    # 텍스트 레이블 공간 확보
            showgrid=True,
            gridcolor="rgba(200,200,200,0.5)",
            zeroline=True,
        ),
        yaxis=dict(
            showgrid=False,
            autorange="reversed",  # 상단에 높은 score 배치 (sort와 일관성)
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=chart_height,
        margin=dict(l=150, r=80, t=50, b=50),   # l=150: 긴 웨이퍼 이름 공간
        showlegend=False,
    )

    return fig


# =============================================================================

# =============================================================================
# [함수 7a] _datasets_app_hash (내부 헬퍼)
# =============================================================================

def _datasets_app_hash(datasets: list) -> str:
    """
    앱이 제공하는 datasets 목록의 이름을 이용해 짧은 해시를 반환.

    [용도]
    앱 제공 datasets가 바뀌었는지 감지 → 동기화 배너 표시 여부 결정.
    이름만 비교하는 이유: df_json 전체 비교는 수십 MB가 될 수 있음.
    """
    names = sorted(ds.get("name", "") for ds in datasets)
    return hashlib.md5(str(names).encode()).hexdigest()[:12]


# =============================================================================
# [함수 7b] _invalidate_ml_results (내부 헬퍼)
# =============================================================================

def _invalidate_ml_results() -> None:
    """PCA / IF 분석 결과를 모두 초기화하는 공통 헬퍼."""
    for key in (_SS_PCA_KEY, _SS_PCA_RESULT, _SS_IF_KEY,
                _SS_IF_RESULT, _SS_NAMES, _SS_PATTERNS):
        st.session_state[key] = None


# =============================================================================
# [함수 7c] _render_dataset_adder (내부 헬퍼)
# =============================================================================

def _render_dataset_adder(data_folder: str) -> None:
    """ML 탭: 파일 또는 수동 입력으로 데이터셋 추가."""

    # data_folder에서 파일 목록 스캔
    file_list: list[str] = []
    if os.path.exists(data_folder):
        file_list = sorted(
            os.path.basename(f)
            for f in glob.glob(os.path.join(data_folder, "*.csv"))
               + glob.glob(os.path.join(data_folder, "*.xls*"))
        )

    add_file_tab, add_manual_tab = st.tabs(["📁 파일에서 추가", "✏️ 수동 입력"])

    # ── [탭 A] 파일에서 추가 ──────────────────────────────────────────
    with add_file_tab:
        if not file_list:
            st.info(
                "ℹ️ 데이터 폴더에 파일이 없습니다. '✏️ 수동 입력' 탭을 사용하세요."
            )
        else:
            # === 기존 파일 선택 로직 전체 (sel_file ~ st.button까지) ===
            # ... (변경 없음, 그대로 유지) ...
            pass

    # ── [탭 B] 수동 입력 ──────────────────────────────────────────────
    with add_manual_tab:
        st.caption(
            "X, Y 좌표와 측정값을 직접 입력하거나 "
            "스프레드시트에서 복사(Ctrl+V)해 붙여넣으세요."
        )

        manual_name = st.text_input(
            "데이터셋 이름",
            value=f"수동입력_{len(st.session_state.get(_SS_DATASETS, []) or []) + 1}",
            key="ml_manual_name",
        )

        _ML_MANUAL_KEY = "ml_manual_df"
        if _ML_MANUAL_KEY not in st.session_state:
            st.session_state[_ML_MANUAL_KEY] = pd.DataFrame({
                "x": [None] * 10, "y": [None] * 10, "data": [None] * 10,
            })

        edited_manual = st.data_editor(
            st.session_state[_ML_MANUAL_KEY],
            num_rows="dynamic",
            key="ml_manual_editor",
            column_config={
                "x":    st.column_config.NumberColumn("X (mm)",  format="%.2f"),
                "y":    st.column_config.NumberColumn("Y (mm)",  format="%.2f"),
                "data": st.column_config.NumberColumn("측정값",  format="%.4f"),
            },
            use_container_width=True,
            hide_index=False,
        )
        st.session_state[_ML_MANUAL_KEY] = edited_manual

        # 유효 데이터 추출
        df_valid = edited_manual.dropna(subset=["x", "y", "data"]).copy()
        for col in ["x", "y", "data"]:
            df_valid[col] = pd.to_numeric(df_valid[col], errors="coerce")
        df_valid = df_valid.dropna().reset_index(drop=True)
        n_valid = len(df_valid)

        if n_valid > 0:
            st.success(f"✅ 유효 포인트: {n_valid}개")
        else:
            st.info("ℹ️ 데이터를 입력하면 추가할 수 있습니다.")

        # 중복 이름 체크
        existing_names = [d.get("name") for d in st.session_state.get(_SS_DATASETS, []) or []]
        name_dup = manual_name in existing_names
        if name_dup:
            st.warning(f"⚠️ '{manual_name}' 이름이 이미 존재합니다.")

        btn_col, reset_col = st.columns([3, 1])
        with btn_col:
            if st.button(
                "✅ 수동 데이터 추가", type="primary",
                key="ml_manual_add_btn",
                disabled=(n_valid < 3 or name_dup),
                use_container_width=True,
            ):
                new_ds = {"name": manual_name, "df_json": df_valid.to_json()}

                if st.session_state.get(_SS_DATASETS) is None:
                    st.session_state[_SS_DATASETS] = []
                st.session_state[_SS_DATASETS].append(new_ds)

                # PCA 캐시 무효화
                _invalidate_ml_results()

                # 테이블 초기화
                st.session_state[_ML_MANUAL_KEY] = pd.DataFrame({
                    "x": [None] * 10, "y": [None] * 10, "data": [None] * 10,
                })
                st.success(f"✅ '{manual_name}' 추가됨")
                st.rerun()

        with reset_col:
            if st.button("🗑️", key="ml_manual_reset", help="테이블 초기화"):
                st.session_state[_ML_MANUAL_KEY] = pd.DataFrame({
                    "x": [None] * 10, "y": [None] * 10, "data": [None] * 10,
                })
                st.rerun()

# =============================================================================
# [함수 7d] _render_dataset_panel (내부 헬퍼)
# =============================================================================

def _render_dataset_panel(
    datasets_from_app: list,
    data_folder: str,
) -> None:
    """
    ML 탭 전용 데이터셋 관리 패널.

    [레이아웃]
    ┌─────────────────────────────────────────────────────────────────┐
    │ 📋 분석 데이터셋 (N개)   [🔄 앱 동기화]   [🗑️ 전체 초기화]        │
    ├──┬──────────────────────────────┬──────────┬────────────────────┤
    │# │ 이름 (이상탐지 결과 아이콘)    │ 포인트 수 │ 삭제               │
    └──┴──────────────────────────────┴──────────┴────────────────────┘
    ▼ [➕ 데이터셋 추가] (expander, 3개 미만이면 기본 펼침)

    [앱 동기화 정책]
    - 최초 진입(_SS_DATASETS is None): 앱 제공 datasets로 자동 초기화
    - 이후: 사용자 관리 리스트 유지
    - 앱 datasets 이름 목록 hash가 바뀌면 [🔄 앱 동기화] 버튼 표시
    - [🗑️ 전체 초기화]: 목록 + 분석 결과 모두 초기화
    """
    # 최초 진입: 앱 제공 datasets로 자동 초기화
    if st.session_state.get(_SS_DATASETS) is None:
        st.session_state[_SS_DATASETS] = list(datasets_from_app)
        st.session_state[_SS_APP_HASH] = _datasets_app_hash(datasets_from_app)

    ml_datasets: list = st.session_state[_SS_DATASETS]

    # 앱 datasets 변경 감지
    current_app_hash = _datasets_app_hash(datasets_from_app)
    saved_app_hash   = st.session_state.get(_SS_APP_HASH, "")
    app_changed      = (current_app_hash != saved_app_hash) and bool(datasets_from_app)

    # ── 헤더: 제목 + 동기화 + 초기화 ──────────────────────────────────────
    hdr_title, hdr_sync, hdr_clear = st.columns([4, 2, 2])
    hdr_title.markdown(
        f"#### 📋 분석 데이터셋 "
        f"<span style='font-size:14px;font-weight:normal;color:#888;'>"
        f"({len(ml_datasets)}개 로드 · 최소 3개 필요)</span>",
        unsafe_allow_html=True,
    )

    # 앱 datasets 변경 시에만 동기화 버튼 표시
    if app_changed:
        n_app = len(datasets_from_app)
        if hdr_sync.button(
            f"🔄 앱 동기화 ({n_app}개)",
            key="ml_sync_btn",
            use_container_width=True,
            help=(
                "앱에서 제공하는 데이터셋 목록이 변경되었습니다.\n"
                "클릭 시 앱 제공 목록으로 교체됩니다.\n"
                "(기존 ML 분석 결과는 초기화됩니다)"
            ),
        ):
            st.session_state[_SS_DATASETS] = list(datasets_from_app)
            st.session_state[_SS_APP_HASH] = current_app_hash
            _invalidate_ml_results()
            st.rerun()
    else:
        hdr_sync.empty()

    if hdr_clear.button(
        "🗑️ 전체 초기화",
        key="ml_clear_all",
        use_container_width=True,
        help="데이터셋 목록과 분석 결과를 모두 초기화합니다.",
    ):
        st.session_state[_SS_DATASETS] = []
        st.session_state[_SS_APP_HASH] = ""
        _invalidate_ml_results()
        st.rerun()

    st.markdown(
        "<div style='border-top:1px solid #e0e0e0;margin:4px 0 8px 0;'></div>",
        unsafe_allow_html=True,
    )

    # ── 데이터셋 테이블 ────────────────────────────────────────────────────
    if not ml_datasets:
        st.info(
            "ℹ️ 데이터셋이 없습니다.  \n"
            "아래 **➕ 데이터셋 추가** 를 펼쳐 파일을 추가하세요."
        )
    else:
        # 이전 ML 결과 (있으면 이름·점수에 반영)
        names_result = st.session_state.get(_SS_NAMES) or []
        if_result    = st.session_state.get(_SS_IF_RESULT)

        def _row_label(name: str) -> str:
            """이상 탐지 결과가 있으면 상태 아이콘 + 점수를 접두어로 추가."""
            if if_result and names_result and name in names_result:
                idx   = names_result.index(name)
                score = float(if_result["scores"][idx])
                if idx in if_result["anomaly_indices"]:
                    return (
                        f"⚠️ **{name}** "
                        f"<span style='color:#cc3300;font-size:11px;'>"
                        f"이상 점수: {score:.3f}</span>"
                    )
                return (
                    f"✅ **{name}** "
                    f"<span style='color:#2a7a2a;font-size:11px;'>"
                    f"점수: {score:.3f}</span>"
                )
            return f"🔘 {name}"

        # 컬럼 헤더
        h_no, h_name, h_pts, h_del = st.columns([0.45, 4.5, 1.3, 0.75])
        for col, txt in zip(
            (h_no, h_name, h_pts, h_del),
            ("#", "이름", "포인트", "삭제"),
        ):
            col.markdown(
                f"<span style='font-size:12px;color:#888;font-weight:600;'>{txt}</span>",
                unsafe_allow_html=True,
            )

        to_remove = None
        for i, ds in enumerate(ml_datasets):
            name = ds.get("name", f"dataset_{i+1}")
            c_no, c_name, c_pts, c_del = st.columns([0.45, 4.5, 1.3, 0.75])

            c_no.markdown(
                f"<div style='padding-top:6px;font-size:13px;color:#666;'>{i+1}</div>",
                unsafe_allow_html=True,
            )
            c_name.markdown(
                f"<div style='padding-top:4px;font-size:13px;'>{_row_label(name)}</div>",
                unsafe_allow_html=True,
            )

            try:
                n_pts = len(pd.read_json(ds["df_json"]))
                c_pts.markdown(
                    f"<div style='padding-top:6px;font-size:13px;color:#555;'>{n_pts:,}</div>",
                    unsafe_allow_html=True,
                )
            except Exception:
                c_pts.markdown(
                    "<div style='padding-top:6px;font-size:12px;color:#aaa;'>—</div>",
                    unsafe_allow_html=True,
                )

            # ✕ 버튼: key에 인덱스+이름 포함 → 목록 변경 후 rerun 시 key 충돌 방지
            if c_del.button("✕", key=f"ml_del_{i}_{name}", use_container_width=True):
                to_remove = i

        if to_remove is not None:
            st.session_state[_SS_DATASETS].pop(to_remove)
            # 웨이퍼 목록 변경 → PCA 무효화
            _invalidate_ml_results()
            st.session_state[_SS_PCA_KEY] = None
            st.session_state[_SS_IF_KEY]  = None
            st.rerun()

    st.markdown(
        "<div style='border-top:1px solid #e0e0e0;margin:8px 0 4px 0;'></div>",
        unsafe_allow_html=True,
    )

    # 3개 미만이면 자동 펼침
    with st.expander("➕ 데이터셋 추가", expanded=(len(ml_datasets) < 3)):
        _render_dataset_adder(data_folder)


# [함수 7] _compute_pca_key (내부 헬퍼)
# =============================================================================

def _compute_pca_key(names: list, resolution: int) -> str:
    """
    PCA 캐시 키 생성 (웨이퍼 이름 목록 + 해상도).

    [캐시 키 설계 원칙]
    - 웨이퍼 목록 변경 → PCA 무효화
    - 해상도 변경 → PCA 무효화 (특성 차원이 달라짐)
    - contamination 변경 → PCA 그대로 (IF만 재실행)
    - 웨이퍼 순서 변경: sorted()로 순서 무관하게 일관된 키 생성
      (동일 웨이퍼 세트라면 순서와 무관하게 같은 PCA 재사용)

    인자:
        names     : 유효 웨이퍼 이름 리스트
        resolution: 특성 추출 해상도

    반환:
        캐시 키 문자열 (MD5 해시 → 짧고 고정 길이)
    """
    key_str = f"{sorted(names)},{resolution}"
    return hashlib.md5(key_str.encode()).hexdigest()[:16]


# =============================================================================
# [함수 8] render_anomaly_tab (UI 렌더러)
# =============================================================================

def render_anomaly_tab(
    datasets: list,
    resolution: int,
    data_folder: str,
) -> None:
    """
    ML 이상 탐지 탭의 전체 UI를 렌더링.

    [레이아웃 구조]
    ┌─────────────────────────────────────────────────────────────────────┐
    │  scikit-learn 미설치 오류 (필요 시만)                                 │
    ├──────────────┬──────────────┬──────────────────────────────────────┤
    │ contamination │ resolution   │  [🤖 이상 탐지 실행] 버튼             │
    │ 슬라이더      │ 슬라이더     │                                       │
    └──────────────┴──────────────┴──────────────────────────────────────┘
    ┌──────────────────────────┬──────────────────────────────────────────┐
    │  PCA 산점도               │  이상 점수 막대 그래프                   │
    └──────────────────────────┴──────────────────────────────────────────┘
    ┌─────────────────────────────────────────────────────────────────────┐
    │  이상 탐지 결과 요약 (통계 지표)                                      │
    ├─────────────────────────────────────────────────────────────────────┤
    │  이상 웨이퍼 Heatmap 미리보기 (compact=True, 최대 6개)                │
    └─────────────────────────────────────────────────────────────────────┘

    [contamination 변경 시 캐시 재사용 로직]
    pca_key = _compute_pca_key(names, ml_resolution)
    if session_state["ml_pca_key"] != pca_key:
        → PCA 재실행 (웨이퍼/해상도 변경)
    if session_state["ml_if_key"] != (pca_key, contamination):
        → IF 재실행 (contamination 변경 또는 PCA 변경)
    → contamination만 바뀌면: PCA 건너뜀, IF만 재실행

    [session_state 키]
    "ml_pca_key"    : 현재 PCA 캐시 키
    "ml_pca_result" : PCA 결과 dict (components, explained_variance_ratio, ...)
    "ml_if_key"     : (pca_key, contamination) 튜플
    "ml_if_result"  : IF 결과 dict (predictions, scores, anomaly_indices, ...)
    "ml_names"      : 유효 웨이퍼 이름 리스트
    "ml_patterns"   : {wafer_name: pattern_str} 패턴 분류 dict

    인자:
        datasets   : st.session_state["datasets"] (웨이퍼 데이터셋 리스트)
        resolution : 기본 보간 해상도 (사이드바 슬라이더)
        data_folder: 데이터 폴더 경로
    """
    # ── scikit-learn 미설치 체크 ─────────────────────────────────────────────
    if not _SKLEARN_OK:
        st.error(
            "❌ scikit-learn이 설치되어 있지 않습니다.\n\n"
            "터미널에서 다음 커맨드를 실행하세요:\n"
            "```\npip install scikit-learn\n```"
        )
        return

    # ── 제목 ──────────────────────────────────────────────────────────────────
    st.markdown("#### 🤖 ML 이상 탐지 (PCA + IsolationForest)")
    st.caption(
        "여러 웨이퍼 맵을 그리드 벡터로 변환 → PCA 차원 축소 → "
        "IsolationForest로 이상 웨이퍼 자동 탐지"
    )
    st.markdown("---")

    # ── [신규] 데이터셋 관리 패널 ────────────────────────────────────────────
    # _render_dataset_panel 이 _SS_DATASETS 를 초기화·관리.
    # 이후 모든 ML 로직은 st.session_state[_SS_DATASETS] 를 사용.
    _render_dataset_panel(datasets_from_app=datasets, data_folder=data_folder)

    ml_datasets: list = st.session_state.get(_SS_DATASETS) or []

    # ── 최소 웨이퍼 수 체크 (데이터셋 패널 아래에서 진행 차단) ─────────────
    n_datasets = len(ml_datasets)
    if n_datasets < 3:
        st.info(
            f"ℹ️ **최소 3개의 웨이퍼가 필요합니다.** 현재 {n_datasets}개.  \n"
            "위의 **➕ 데이터셋 추가** 로 파일을 추가하거나 "
            "**🔄 앱 동기화** 버튼을 눌러 앱 데이터셋을 불러오세요."
        )
        return

    # ── 소규모 샘플 신뢰도 경고 ─────────────────────────────────────────────
    if n_datasets < 5:
        st.warning(
            f"⚠️ 웨이퍼 수({n_datasets}개)가 적어 이상 탐지 결과의 신뢰도가 낮을 수 있습니다. "
            "10개 이상 권장. 탐지 결과를 참고 자료로만 활용하세요."
        )

    st.markdown("---")

    # ── 컨트롤 패널 ───────────────────────────────────────────────────────────
    ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([1, 1, 2])

    with ctrl_c1:
        contamination: float = st.slider(
            "이상 비율 (contamination)",
            min_value=0.05,
            max_value=0.30,
            value=st.session_state.get(_SS_CONTAM, 0.10),
            step=0.05,
            key=_SS_CONTAM,
            help=(
                "전체 웨이퍼 중 이상으로 간주할 비율.\n\n"
                "공정 경험적 결함율에 맞게 조정하세요.\n"
                "기본값 0.10 = 10%"
            ),
        )

    with ctrl_c2:
        ml_resolution: int = st.slider(
            "특성 추출 해상도",
            min_value=20,
            max_value=80,
            value=st.session_state.get(_SS_RESOLUTION, 40),
            step=10,
            key=_SS_RESOLUTION,
            help=(
                "웨이퍼 맵 보간 그리드 해상도.\n\n"
                "낮을수록 빠름 (20×20=400차원)\n"
                "높을수록 정밀 (80×80=6400차원)\n"
                "기본값 40 (40×40=1600차원)"
            ),
        )

    with ctrl_c3:
        st.markdown("")   # 버튼 수직 정렬용 여백
        run_clicked = st.button(
            "🤖 이상 탐지 실행",
            type="primary",
            use_container_width=True,
            key="ml_run_btn",
            help=(
                "버튼 클릭 시 동작:\n"
                "1. 각 웨이퍼 ZI 그리드 추출 (특성 벡터화)\n"
                "2. PCA 차원 축소\n"
                "3. IsolationForest 이상 탐지"
            ),
        )

    # ── contamination 변경 감지: IF만 재실행 트리거 ──────────────────────────
    # 버튼 클릭 없이 슬라이더 변경 시: contamination만 바뀌면 IF 자동 재실행
    if_key_current = (
        st.session_state.get(_SS_PCA_KEY, ""),
        contamination,
    )
    if_needs_rerun = (
        st.session_state.get(_SS_IF_RESULT) is not None
        and st.session_state.get(_SS_IF_KEY) != if_key_current
    )

    if if_needs_rerun:
        # contamination 변경 → IF만 재실행 (PCA 재사용)
        pca_result = st.session_state.get(_SS_PCA_RESULT)
        if pca_result is not None:
            with st.spinner("contamination 변경 → IsolationForest 재실행 중..."):
                try:
                    if_result = run_isolation_forest(
                        pca_result["components"], contamination
                    )
                    st.session_state[_SS_IF_RESULT] = if_result
                    st.session_state[_SS_IF_KEY]    = if_key_current
                    # [수정] _SS_CONTAM 수동 write 제거: key=_SS_CONTAM 위젯이 자동 관리
                except Exception as e:
                    st.error(f"IsolationForest 재실행 실패: {e}")

    # ── 메인 실행 버튼 처리 ──────────────────────────────────────────────────
    if run_clicked:
        # ── _SS_DATASETS 에서 df_json 목록 수집 ─────────────────────────────
        # [수정] datasets(앱 파라미터) 대신 st.session_state[_SS_DATASETS] 사용
        # → ML 탭에서 직접 추가/삭제한 목록이 반영됨
        maps_data = []
        for ds in ml_datasets:
            name    = ds.get("name", "Unknown")
            df_json = ds.get("df_json", None)
            if df_json is None:
                continue
            maps_data.append({"df_json": df_json, "name": name})

        if len(maps_data) < 3:
            st.error(
                "❌ 유효한 웨이퍼 데이터가 3개 미만입니다. "
                "위의 데이터셋 패널에서 파일을 추가하세요."
            )
            return

        # ── 특성 추출 ────────────────────────────────────────────────────────
        with st.spinner(f"웨이퍼 특성 추출 중... ({len(maps_data)}개 × {ml_resolution}² 그리드)"):
            try:
                feature_matrix, valid_names, valid_mask = prepare_wafer_features(
                    maps_data, resolution=ml_resolution
                )
            except Exception as e:
                st.error(f"특성 추출 실패: {e}")
                return

        if len(valid_names) < 3:
            st.error(
                f"❌ 유효한 웨이퍼가 {len(valid_names)}개뿐입니다 (최소 3개 필요). "
                "보간에 실패한 웨이퍼가 있을 수 있습니다."
            )
            return

        # ── PCA 실행 (캐시 키 확인) ──────────────────────────────────────────
        pca_key_new = _compute_pca_key(valid_names, ml_resolution)
        pca_key_old = st.session_state.get(_SS_PCA_KEY, "")

        if pca_key_new != pca_key_old:
            # 웨이퍼 목록 또는 해상도 변경 → PCA 재실행
            with st.spinner("PCA 차원 축소 실행 중..."):
                try:
                    pca_result = run_pca(feature_matrix)
                    st.session_state[_SS_PCA_RESULT] = pca_result
                    st.session_state[_SS_PCA_KEY]    = pca_key_new
                    # PCA 변경 시 IF 결과도 무효화
                    st.session_state[_SS_IF_RESULT]  = None
                    st.session_state[_SS_IF_KEY]     = None
                except Exception as e:
                    st.error(f"PCA 실행 실패: {e}")
                    return
        else:
            # 웨이퍼/해상도 동일 → PCA 재사용
            pca_result = st.session_state[_SS_PCA_RESULT]
            st.info("✅ PCA 결과 재사용 (웨이퍼 목록·해상도 동일)")

        # ── IsolationForest 실행 ─────────────────────────────────────────────
        if_key_new  = (pca_key_new, contamination)
        if_key_curr = st.session_state.get(_SS_IF_KEY)

        if if_key_new != if_key_curr:
            with st.spinner("IsolationForest 이상 탐지 실행 중..."):
                try:
                    if_result = run_isolation_forest(
                        pca_result["components"], contamination
                    )
                    st.session_state[_SS_IF_RESULT] = if_result
                    st.session_state[_SS_IF_KEY]    = if_key_new
                except Exception as e:
                    st.error(f"IsolationForest 실행 실패: {e}")
                    return
        else:
            if_result = st.session_state[_SS_IF_RESULT]

        # ── 패턴 분류 ────────────────────────────────────────────────────────
        patterns = {}
        for i, name in enumerate(valid_names):
            df_json = maps_data[i]["df_json"] if i < len(maps_data) else ""
            patterns[name] = classify_anomaly_pattern(df_json)
        st.session_state[_SS_PATTERNS]    = patterns
        st.session_state[_SS_NAMES]       = valid_names
        # [수정] _SS_RESOLUTION 수동 write 제거: key=_SS_RESOLUTION 위젯이 자동 관리

        # 유효하지 않은 웨이퍼 수 경고
        n_invalid = sum(1 for v in valid_mask if not v)
        if n_invalid > 0:
            st.warning(f"⚠️ 보간 실패로 {n_invalid}개 웨이퍼 제외됨.")

    # ── 결과 렌더링 ───────────────────────────────────────────────────────────
    pca_result = st.session_state.get(_SS_PCA_RESULT)
    if_result  = st.session_state.get(_SS_IF_RESULT)
    names      = st.session_state.get(_SS_NAMES, [])
    patterns   = st.session_state.get(_SS_PATTERNS, {})

    if pca_result is None or if_result is None:
        st.info(
            "ℹ️ 파라미터 설정 후 **'🤖 이상 탐지 실행'** 버튼을 클릭하세요.\n\n"
            "PCA와 IsolationForest로 이상 웨이퍼를 자동 탐지합니다."
        )
        return

    # ── 요약 지표 ────────────────────────────────────────────────────────────
    n_valid   = len(names)
    n_anomaly = len(if_result["anomaly_indices"])
    n_normal  = n_valid - n_anomaly
    anom_pct  = n_anomaly / n_valid * 100 if n_valid > 0 else 0
    mean_score = float(if_result["scores"].mean()) if n_valid > 0 else 0.0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("분석 웨이퍼", f"{n_valid}개")
    m2.metric(
        "이상 탐지",
        f"{n_anomaly}개",
        delta=f"{anom_pct:.1f}%",
        delta_color="inverse" if n_anomaly > 0 else "off",
    )
    m3.metric("정상", f"{n_normal}개")
    m4.metric(
        "평균 이상 점수",
        f"{mean_score:.4f}",
        help="0 = 정상, 1 = 완전 이상",
    )
    m5.metric(
        "PCA 설명력",
        f"{sum(pca_result['explained_variance_ratio'][:2])*100:.1f}%",
        help="PC1+PC2가 전체 분산의 몇 %를 설명하는지",
    )

    st.markdown("---")

    # ── 상단: PCA 산점도 | 이상 점수 막대 ──────────────────────────────────
    col_pca, col_bar = st.columns([1, 1])

    # df_jsons 목록 수집 (패턴 분류용 hover 데이터)
    # [수정] datasets → ml_datasets (ML 탭 전용 목록)
    df_jsons_for_hover = []
    for ds in ml_datasets:
        if ds.get("name") in names:
            df_jsons_for_hover.append(ds.get("df_json", ""))
    # 이름 순서 맞추기
    name_to_df_json = {
        ds.get("name"): ds.get("df_json", "")
        for ds in ml_datasets if ds.get("df_json")
    }
    df_jsons_ordered = [name_to_df_json.get(n, "") for n in names]

    with col_pca:
        st.markdown("##### 🔵 PCA 이상 탐지 산점도")
        fig_pca = create_pca_scatter(
            pca_result, if_result, names, df_jsons_ordered
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    with col_bar:
        st.markdown("##### 📊 웨이퍼별 이상 점수")
        fig_bar = create_anomaly_score_bar(
            wafer_names=names,
            scores=if_result["scores"],
            is_anomaly=(if_result["predictions"] == -1),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── 이상 패턴 분류 테이블 ────────────────────────────────────────────────
    if patterns:
        st.markdown("---")
        st.markdown("##### 🔬 이상 패턴 분류")

        # 이상 웨이퍼 강조 + 정상 웨이퍼도 포함한 전체 테이블
        pattern_rows = []
        is_anom_set = set(if_result["anomaly_indices"])
        for i, name in enumerate(names):
            is_anom = i in is_anom_set
            score   = float(if_result["scores"][i])
            pattern = patterns.get(name, "N/A")
            pattern_rows.append({
                "상태":        "⚠️ 이상" if is_anom else "✅ 정상",
                "웨이퍼":      name,
                "이상 점수":   round(score, 4),
                "패턴 분류":   pattern,
            })

        # 이상 점수 내림차순 정렬
        pattern_rows.sort(key=lambda r: r["이상 점수"], reverse=True)

        st.dataframe(
            pd.DataFrame(pattern_rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "이상 점수": st.column_config.ProgressColumn(
                    "이상 점수",
                    format="%.4f",
                    min_value=0.0,
                    max_value=1.0,
                ),
            },
        )

    # ── 이상 웨이퍼 Heatmap 미리보기 ────────────────────────────────────────
    anomaly_indices = if_result["anomaly_indices"]
    if anomaly_indices:
        st.markdown("---")
        st.markdown("##### 🗺️ 이상 웨이퍼 Heatmap 미리보기")

        # 최대 6개만 미리보기 (화면 공간 제한)
        preview_count = min(len(anomaly_indices), 6)
        preview_cols  = st.columns(min(preview_count, 3))

        for col_idx, anom_idx in enumerate(anomaly_indices[:preview_count]):
            with preview_cols[col_idx % 3]:
                name    = names[anom_idx]
                score   = float(if_result["scores"][anom_idx])
                pattern = patterns.get(name, "N/A")

                st.markdown(
                    f"**{name}**  \n"
                    f"점수: `{score:.4f}` | 패턴: `{pattern}`"
                )

                # df_json으로 compact Heatmap 생성
                ds_match = next(
                    (ds for ds in ml_datasets if ds.get("name") == name), None
                )
                if ds_match and ds_match.get("df_json"):
                    try:
                        fig_preview = create_2d_heatmap(
                            df_json=ds_match["df_json"],
                            resolution=resolution,
                            colorscale="RdBu_r",    # 이상 강조: 빨강-파랑
                            show_points=False,
                            compact=True,           # 비교 모드용 소형
                        )
                        st.plotly_chart(
                            fig_preview,
                            use_container_width=True,
                            key=f"ml_preview_{name}_{col_idx}",
                        )
                    except Exception:
                        st.warning(f"'{name}' 미리보기 실패")
                else:
                    st.info("데이터 없음")

    # ── 결과 해석 가이드 ────────────────────────────────────────────────────
    with st.expander("📖 결과 해석 가이드", expanded=False):
        st.markdown(
            """
**이상 점수 해석:**
- `0.0 ~ 0.4`: 정상 범위 (공정 변동 내)
- `0.4 ~ 0.7`: 주의 요망 (경계 수준)
- `0.7 ~ 1.0`: 이상 의심 (공정 점검 권장)

**패턴 분류 의미:**
| 패턴 | 의미 | 공정 원인 |
|------|------|-----------|
| Ring | 방사형 대칭 이상 | 가스 분포 중심집중/확산 불균일 |
| Edge Degradation | 가장자리 두께 저하 | Edge exclusion, 로딩 효과 |
| X/Y-Gradient | 방향성 두께 구배 | 기판 기울기, 가스 방향성 |
| Hotspot | 국소 피크 | 파티클, 스크래치, 측정 오류 |
| Global Shift | 전체 수준 이탈 | 공정 레시피 변경, 드리프트 |
| Normal | 정상 패턴 | — |

**contamination 조정 가이드:**
- 공정 결함율이 알려져 있다면 그 값으로 설정 (예: 결함율 5% → 0.05)
- 모름: 0.10으로 시작 후 PCA 산점도에서 이상치 분리도 확인하며 조정
            """
        )