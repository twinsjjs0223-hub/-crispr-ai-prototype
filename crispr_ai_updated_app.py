# -*- coding: utf-8 -*-
"""CRISPR-AI: 생물학적 특징을 보완한 공개/가상 데이터 분석 프로토타입

이 앱은 업로드된 표 형식 자료에서 온타겟 편집 활성과 관련된 통계적 패턴을
탐색하는 교육·연구용 프로토타입입니다. 실제 유전자 편집 실험을 수행하거나
실험 결과, 임상적 유효성, 오프타겟 안전성을 보증하지 않습니다.
"""
from __future__ import annotations

import hashlib
import io
import itertools
import re
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

APP_TITLE = "CRISPR-AI | 생물학적 특징 기반 효율 데이터 분석"
RANDOM_STATE = 42
DNA_ALPHABET = "ACGTN"
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_ALLOWED = set(AA_ALPHABET + "BXZJUO")
POSITION_LIMIT = 40  # 가이드/표적 서열의 앞쪽 위치 특징을 고정 길이로 생성
KMER_SIZES = (2, 3)


# -----------------------------------------------------------------------------
# 파일 입력 및 데모 데이터
# -----------------------------------------------------------------------------
def read_csv_safely(file_bytes: bytes) -> pd.DataFrame:
    errors: list[str] = []
    frame = None
    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            raw = file_bytes.decode(encoding)
            frame = pd.read_csv(io.StringIO(raw), sep=None, engine="python")
            break
        except Exception as exc:
            errors.append(f"{encoding}: {exc}")
    if frame is None:
        raise ValueError(
            "CSV를 읽지 못했습니다. UTF-8 또는 CP949로 저장했는지 확인해 주세요.\n"
            + "\n".join(errors)
        )

    cleaned: list[str] = []
    used: dict[str, int] = {}
    for raw_name in frame.columns:
        name = str(raw_name).replace("\ufeff", "").strip() or "이름없는열"
        used[name] = used.get(name, 0) + 1
        cleaned.append(name if used[name] == 1 else f"{name}_{used[name]}")
    frame.columns = cleaned
    frame = frame.dropna(how="all").copy()
    if frame.empty:
        raise ValueError("CSV에 분석할 행이 없습니다.")
    return frame


def make_synthetic_data(n_rows: int = 240) -> pd.DataFrame:
    """화면·함수 작동 점검용 가상 자료. 생물학적 측정값을 모사하지 않는다."""
    rng = np.random.default_rng(20261101)
    bases = np.array(list("ACGT"))
    sequences = ["".join(rng.choice(bases, size=20)) for _ in range(n_rows)]
    variants = np.array(["SpCas9", "SpCas9-HF1", "eSpCas9(1.1)", "SaCas9"])
    cas9 = rng.choice(variants, size=n_rows, p=[0.35, 0.25, 0.25, 0.15])
    offsets = {"SpCas9": 8.0, "SpCas9-HF1": -5.0, "eSpCas9(1.1)": 2.0, "SaCas9": -8.0}
    interactions = {"SpCas9": 10.0, "SpCas9-HF1": -6.0, "eSpCas9(1.1)": 3.0, "SaCas9": -4.0}
    gc = np.array([100 * (s.count("G") + s.count("C")) / len(s) for s in sequences])
    y = (
        47 + 0.75 * (gc - 50)
        + np.array([offsets[x] for x in cas9])
        + np.array([interactions[x] for x in cas9]) * (gc - 50) / 20
        + rng.normal(0, 7, n_rows)
    )
    y = np.clip(y, 0, 100).round(2)
    return pd.DataFrame({
        "record_id": [f"SYN-{i+1:03d}" for i in range(n_rows)],
        "guide_sequence": sequences,
        "cas9_variant": cas9,
        "synthetic_activity_score": y,
        "data_status": "synthetic",
    })


# -----------------------------------------------------------------------------
# 서열 정리 및 생물학적 특징
# -----------------------------------------------------------------------------
def clean_dna_sequence(value: Any) -> str | None:
    if pd.isna(value):
        return None
    seq = re.sub(r"\s+", "", str(value)).upper().replace("U", "T")
    return seq if seq and re.fullmatch(r"[ACGTN]+", seq) else None


def clean_protein_sequence(value: Any) -> str | None:
    if pd.isna(value):
        return None
    seq = re.sub(r"\s+", "", str(value)).upper().replace("*", "")
    return seq if seq and set(seq).issubset(AA_ALLOWED) else None


def dna_sequence_features(series: pd.Series, prefix: str) -> pd.DataFrame:
    """조성, 고정 위치별 염기, 2/3-mer 빈도 특징을 만든다.

    POSITION_LIMIT보다 긴 서열은 위치별 특징에서 앞쪽 40개 위치만 사용한다.
    전체 서열은 GC/조성과 k-mer 계산에 계속 사용한다.
    """
    rows: list[dict[str, float]] = []
    all_kmers = {
        k: ["".join(x) for x in itertools.product("ACGT", repeat=k)]
        for k in KMER_SIZES
    }
    for value in series:
        seq = clean_dna_sequence(value)
        rec: dict[str, float] = {}
        if seq is None:
            seq = ""
            missing = 1.0
        else:
            missing = 0.0
        length = len(seq)
        rec[f"{prefix}_서열_결측"] = missing
        rec[f"{prefix}_길이"] = float(length)
        rec[f"{prefix}_GC비율(%)"] = (
            100.0 * (seq.count("G") + seq.count("C")) / length if length else 0.0
        )
        for base in DNA_ALPHABET:
            rec[f"{prefix}_{base}_비율"] = seq.count(base) / length if length else 0.0

        for pos in range(POSITION_LIMIT):
            base = seq[pos] if pos < length else "-"
            for symbol in DNA_ALPHABET:
                rec[f"{prefix}_위치{pos+1}_{symbol}"] = float(base == symbol)
            rec[f"{prefix}_위치{pos+1}_서열끝"] = float(base == "-")

        for k, kmers in all_kmers.items():
            possible = [seq[i:i+k] for i in range(max(0, length-k+1))]
            possible = [x for x in possible if len(x) == k and set(x).issubset(set("ACGT"))]
            denom = len(possible)
            counts = {x: 0 for x in kmers}
            for item in possible:
                counts[item] += 1
            for item in kmers:
                rec[f"{prefix}_{k}mer_{item}"] = counts[item] / denom if denom else 0.0
        rows.append(rec)
    return pd.DataFrame(rows, index=series.index).astype(float)


def protein_sequence_features(series: pd.Series) -> pd.DataFrame:
    """아미노산 길이·조성 특징. 정렬 없이 특정 위치/구조를 추정하지 않는다."""
    rows: list[dict[str, float]] = []
    for value in series:
        seq = clean_protein_sequence(value) or ""
        n = len(seq)
        rec = {"Cas9_단백질_길이": float(n), "Cas9_단백질_결측": float(n == 0)}
        for aa in AA_ALPHABET:
            rec[f"Cas9_아미노산_{aa}_비율"] = seq.count(aa) / n if n else 0.0
        rec["Cas9_모호한_잔기_비율"] = (
            sum(seq.count(x) for x in "BXZJUO") / n if n else 0.0
        )
        rows.append(rec)
    return pd.DataFrame(rows, index=series.index).astype(float)


def numeric_context(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def validate_and_prepare(
    frame: pd.DataFrame,
    target_col: str,
    sequence_col: str | None,
    cas9_col: str | None,
    protein_col: str | None,
    target_context_col: str | None = None,
    pam_col: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = frame.copy()
    y = pd.to_numeric(work[target_col], errors="coerce")
    target_ok = y.notna() & np.isfinite(y)
    stats: dict[str, Any] = {
        "전체 행": int(len(work)),
        "목표값 숫자 변환 실패/결측": int((~target_ok).sum()),
        "gRNA 서열 검사": "검사하지 않음" if not sequence_col else "검사함",
        "잘못된 gRNA 서열": "검사하지 않음" if not sequence_col else 0,
        "Cas9 결측 검사": "검사하지 않음" if not cas9_col else "검사함",
        "Cas9 값 결측": "검사하지 않음" if not cas9_col else 0,
        "Cas9 단백질 서열 검사": "검사하지 않음" if not protein_col else "검사함",
        "잘못된 Cas9 단백질 서열": "검사하지 않음" if not protein_col else 0,
        "표적 주변 서열의 결측/형식 오류": "검사하지 않음" if not target_context_col else 0,
        "PAM 결측": "검사하지 않음" if not pam_col else 0,
    }
    work["__target_numeric__"] = y
    mask = target_ok.copy()

    if sequence_col:
        cleaned = work[sequence_col].map(clean_dna_sequence)
        bad = cleaned.isna()
        stats["잘못된 gRNA 서열"] = int(bad.sum())
        work["__sequence_clean__"] = cleaned
        mask &= ~bad

    if cas9_col:
        cas = work[cas9_col].astype("string").str.strip()
        missing = cas.isna() | cas.eq("")
        stats["Cas9 값 결측"] = int(missing.sum())
        work["__cas9_clean__"] = cas.fillna("")
        mask &= ~missing

    if protein_col:
        protein = work[protein_col].map(clean_protein_sequence)
        bad = protein.isna()
        stats["잘못된 Cas9 단백질 서열"] = int(bad.sum())
        work["__protein_clean__"] = protein
        mask &= ~bad

    if target_context_col:
        context_seq = work[target_context_col].map(clean_dna_sequence)
        stats["표적 주변 서열의 결측/형식 오류"] = int(context_seq.isna().sum())
        work["__target_context_clean__"] = context_seq
    if pam_col:
        pam_values = work[pam_col].astype("string").str.strip()
        stats["PAM 결측"] = int((pam_values.isna() | pam_values.eq("")).sum())

    valid = work.loc[mask].copy()
    stats["최종 유효 행"] = int(len(valid))
    stats["정확히 중복된 행"] = int(frame.duplicated().sum())
    if sequence_col and len(valid):
        lengths = valid["__sequence_clean__"].str.len()
        stats["gRNA 길이 범위"] = f"{int(lengths.min())}~{int(lengths.max())} nt"
        stats["gRNA 길이 종류 수"] = int(lengths.nunique())
    if cas9_col and len(valid):
        stats["Cas9 고유 범주 수"] = int(valid["__cas9_clean__"].nunique())
    if protein_col and len(valid):
        protein_lengths = valid["__protein_clean__"].str.len()
        stats["Cas9 단백질 길이 범위"] = f"{int(protein_lengths.min())}~{int(protein_lengths.max())} aa"
        if protein_lengths.median() < 500:
            stats["Cas9 단백질 서열 주의"] = "중앙 길이가 500 aa 미만입니다. 전체 단백질이 아니라 일부 서열 조각인지 확인하세요."
    return valid, stats


# -----------------------------------------------------------------------------
# 특징 블록과 머신러닝
# -----------------------------------------------------------------------------
def build_feature_blocks(valid: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    blocks: dict[str, pd.DataFrame] = {}
    idx = valid.index
    if cfg.get("sequence_col"):
        blocks["grna"] = dna_sequence_features(valid["__sequence_clean__"], "gRNA")
    if cfg.get("cas9_col"):
        blocks["cas9_variant"] = pd.DataFrame(
            {"Cas9_변이체": valid["__cas9_clean__"].astype(str)}, index=idx
        )
    if cfg.get("protein_col"):
        blocks["cas9_protein"] = protein_sequence_features(valid["__protein_clean__"])

    common_parts: list[pd.DataFrame] = []
    if cfg.get("pam_col"):
        common_parts.append(pd.DataFrame(
            {"공통_PAM": valid[cfg["pam_col"]].astype("string").fillna("미기록").astype(str)},
            index=idx,
        ))
    if cfg.get("target_context_col"):
        context_seq = valid["__target_context_clean__"] if "__target_context_clean__" in valid else valid[cfg["target_context_col"]].map(clean_dna_sequence)
        common_parts.append(dna_sequence_features(context_seq, "표적주변"))
    for col in cfg.get("context_cols", []):
        name = f"실험맥락_{col}"
        s = valid[col]
        if numeric_context(s):
            common_parts.append(pd.DataFrame({name: pd.to_numeric(s, errors="coerce")}, index=idx))
        else:
            common_parts.append(pd.DataFrame({name: s.astype("string").fillna("미기록").astype(str)}, index=idx))
    blocks["common"] = pd.concat(common_parts, axis=1) if common_parts else pd.DataFrame(index=idx)
    return blocks


def split_feature_types(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    categorical_cols = [c for c in frame.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_model(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    transformers = []
    if numeric_cols:
        transformers.append((
            "numeric",
            Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]),
            numeric_cols,
        ))
    if categorical_cols:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_cols,
        ))
    if not transformers:
        raise ValueError("모델에 사용할 입력 특징이 없습니다. 열 역할 지정을 확인해 주세요.")
    prep = ColumnTransformer(transformers=transformers, remainder="drop")
    forest = RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", prep), ("model", forest)])


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    corr = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R²": float(r2_score(y_true, y_pred)),
        "Spearman": None if pd.isna(corr) else float(corr),
    }


def evaluate_models(
    valid: pd.DataFrame,
    cfg: dict[str, Any],
    split_mode: str,
    group_col: str | None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, str, dict[str, Any], dict[str, list[str]]]:
    y = valid["__target_numeric__"].to_numpy(dtype=float)
    blocks = build_feature_blocks(valid, cfg)
    grna_parts = [blocks[k] for k in ("grna",) if k in blocks]
    cas_parts = [blocks[k] for k in ("cas9_variant", "cas9_protein") if k in blocks]
    common = blocks["common"]

    grna_x = pd.concat(grna_parts + ([common] if not common.empty else []), axis=1) if grna_parts else None
    cas_x = pd.concat(cas_parts + ([common] if not common.empty else []), axis=1) if cas_parts else None
    combined_x = pd.concat(
        grna_parts + cas_parts + ([common] if not common.empty else []), axis=1
    ) if grna_parts and cas_parts else None

    if split_mode == "그룹 기준 분할":
        if not group_col:
            raise ValueError("그룹 기준 분할을 선택했다면 그룹 열을 지정해야 합니다.")
        groups = valid[group_col].astype("string").fillna("__결측그룹__")
        if groups.nunique() < 2:
            raise ValueError("그룹이 2개 미만이라 그룹 분할을 수행할 수 없습니다.")
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
        train_idx, test_idx = next(splitter.split(valid, y, groups=groups))
        train_groups = set(groups.iloc[train_idx].astype(str))
        test_groups = set(groups.iloc[test_idx].astype(str))
        overlap = train_groups.intersection(test_groups)
        split_text = (
            f"그룹 기준 분할: {group_col} / 학습 {len(train_idx):,}행, 평가 {len(test_idx):,}행 "
            f"(학습 그룹 {len(train_groups):,}개, 평가 그룹 {len(test_groups):,}개)"
        )
        split_info = {"학습 그룹": len(train_groups), "평가 그룹": len(test_groups), "겹치는 그룹": len(overlap)}
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(valid)), test_size=0.25, random_state=RANDOM_STATE
        )
        split_text = f"무작위 75:25 분할 / 학습 {len(train_idx):,}행, 평가 {len(test_idx):,}행"
        split_info = {"학습 그룹": None, "평가 그룹": None, "겹치는 그룹": "그룹 분할 아님"}

    model_specs: list[tuple[str, pd.DataFrame]] = []
    if grna_x is not None:
        model_specs.append(("gRNA 단독", grna_x))
    if cas_x is not None:
        model_specs.append(("Cas9 단독", cas_x))
    if combined_x is not None:
        model_specs.append(("gRNA + Cas9 통합", combined_x))
    if not model_specs:
        raise ValueError("gRNA 서열 또는 Cas9 변이체/단백질 서열 중 하나 이상을 선택해 주세요.")

    models: dict[str, Any] = {}
    metrics_rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(np.zeros((len(train_idx), 1)), y[train_idx])
    dummy_pred = dummy.predict(np.zeros((len(test_idx), 1)))
    metrics_rows.append({"모델": "평균값 기준선", **calculate_metrics(y[test_idx], dummy_pred)})

    model_columns: dict[str, list[str]] = {}
    for name, X in model_specs:
        if X.shape[1] == 0:
            continue
        numeric_cols, categorical_cols = split_feature_types(X)
        model = build_model(numeric_cols, categorical_cols)
        model.fit(X.iloc[train_idx], y[train_idx])
        pred = model.predict(X.iloc[test_idx])
        models[name] = model
        predictions[name] = pred
        model_columns[name] = list(X.columns)
        metrics_rows.append({"모델": name, **calculate_metrics(y[test_idx], pred)})

    test = valid.iloc[test_idx].drop(
        columns=[c for c in valid.columns if c.startswith("__")], errors="ignore"
    ).copy()
    test["실제_목표값"] = y[test_idx]
    for name, pred in predictions.items():
        test[f"예측값_{name}"] = pred
    return models, pd.DataFrame(metrics_rows), test, split_text, split_info, model_columns


# -----------------------------------------------------------------------------
# 해석 문구
# -----------------------------------------------------------------------------
def fmt_num(value: Any, digits: int = 3) -> str:
    return "계산 불가" if value is None or pd.isna(value) else f"{float(value):.{digits}f}"


def make_interpretations(metrics: pd.DataFrame, cfg: dict[str, Any], split_mode: str) -> list[str]:
    rows = {row["모델"]: row for _, row in metrics.iterrows()}
    baseline = rows.get("평균값 기준선")
    messages: list[str] = []
    for name in ("gRNA 단독", "Cas9 단독", "gRNA + Cas9 통합"):
        if name not in rows or baseline is None:
            continue
        row = rows[name]
        reduction = (baseline["MAE"] - row["MAE"]) / baseline["MAE"] * 100 if baseline["MAE"] else 0
        direction = "낮아졌습니다" if reduction > 0 else "높아졌습니다" if reduction < 0 else "같습니다"
        messages.append(
            f"**{name}:** 평균값 기준선보다 MAE가 {abs(reduction):.1f}% {direction}. "
            f"R²는 {fmt_num(row['R²'])}, Spearman은 {fmt_num(row['Spearman'])}입니다."
        )
    if "gRNA 단독" in rows and "gRNA + Cas9 통합" in rows:
        g = rows["gRNA 단독"]["MAE"]
        c = rows["gRNA + Cas9 통합"]["MAE"]
        if g:
            delta = (g - c) / g * 100
            if delta > 0:
                messages.append(
                    f"**Cas9 정보 추가 비교:** 이 평가 분할에서는 통합 모델의 MAE가 gRNA 단독보다 "
                    f"{delta:.1f}% 낮았습니다. 이는 해당 분할에서 추가 입력이 예측에 기여했을 가능성을 뜻하며, "
                    "Cas9의 인과적·분자적 작용을 입증하는 결과는 아닙니다."
                )
            elif delta < 0:
                messages.append(
                    f"**Cas9 정보 추가 비교:** 이 평가 분할에서는 통합 모델의 MAE가 gRNA 단독보다 "
                    f"{abs(delta):.1f}% 높았습니다. 현재 특징과 데이터에서는 Cas9 정보를 추가해 성능 개선이 확인되지 않았습니다."
                )
            else:
                messages.append("**Cas9 정보 추가 비교:** 두 모델의 MAE가 같습니다. 추가 정보의 이점이 확인되지 않았습니다.")
    if split_mode == "무작위 분할":
        messages.append(
            "**일반화 한계:** 무작위 분할은 같은 gRNA, 표적 또는 연구 출처가 학습·평가에 함께 들어갈 수 있습니다. "
            "새로운 연구나 새로운 gRNA에 대한 성능으로 확대 해석하지 마세요."
        )
    else:
        messages.append(
            "**일반화 해석:** 그룹 기준 분할을 사용했습니다. 어떤 그룹 열을 기준으로 나눴는지 함께 보고해야 하며, "
            "그룹 분할 결과도 외부 검증을 완전히 대체하지는 않습니다."
        )
    messages.append(
        "**생물학적 한계:** 위치별 염기와 k-mer는 서열 패턴 특징이지 RNA 구조, DNA 접근성, 단백질 3차원 구조를 직접 계산한 값이 아닙니다. "
        "특징 중요도나 성능 향상만으로 생물학적 인과관계를 결론 내릴 수 없습니다."
    )
    return messages


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🧬", layout="wide")
st.title(APP_TITLE)
st.caption("gRNA·Cas9·실험 맥락 특징을 이용한 회귀 분석 및 결과 해석 프로토타입")
st.warning(
    "연구상 주의: 모델은 업로드한 데이터의 통계적 패턴을 학습합니다. 실제 유전자 편집 실험, "
    "생물학적 인과관계, 임상적 유효성 또는 오프타겟 안전성을 입증하지 않습니다."
)

with st.sidebar:
    st.header("데이터 선택")
    source_mode = st.radio("분석할 데이터", ["CSV 직접 업로드", "가상 데이터 데모"], index=0)
    st.caption("실제 연구 데이터와 가상 테스트 데이터의 결과를 섞어 해석하지 마세요.")

if source_mode == "가상 데이터 데모":
    df = make_synthetic_data()
    source_label = "가상 테스트 데이터"
    st.info("240행 가상 자료입니다. 임의 규칙으로 생성했으며 실제 생물학적 측정값이 아닙니다.")
    st.download_button(
        "가상 데이터 CSV 내려받기", data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="CRISPR_AI_synthetic_demo.csv", mime="text/csv"
    )
else:
    upload = st.file_uploader("실제 또는 연구용 CSV 업로드", type=["csv"])
    if upload is None:
        st.info("분석할 CSV 파일을 업로드해 주세요.")
        st.stop()
    try:
        df = read_csv_safely(upload.getvalue())
    except Exception as exc:
        st.error(f"파일을 읽지 못했습니다: {exc}")
        st.stop()
    source_label = "사용자 업로드 CSV(출처·정의 별도 확인 필요)"

st.subheader("1. 데이터 미리보기")
st.caption(f"자료 유형: {source_label} · 행 {len(df):,}개 · 열 {len(df.columns)}개")
if "data_status" in df.columns:
    status_values = df["data_status"].astype(str).str.lower().unique().tolist()
    if any(("synthetic" in x) or ("virtual" in x) or ("가상" in x) for x in status_values):
        st.warning("data_status 열에서 가상 자료 표시를 찾았습니다. 실제 생물학적 성능으로 해석하지 마세요.")
st.dataframe(df.head(10), use_container_width=True)

columns = list(df.columns)
none_option = "(사용 안 함)"


def guess_column(candidates: list[str]) -> str | None:
    low_map = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in low_map:
            return low_map[candidate.lower()]
    return None


def pick_column(label: str, default: str | None = None) -> str:
    options = [none_option] + columns
    idx = options.index(default) if default in options else 0
    return st.selectbox(label, options, index=idx, key=f"map_{label}")


def optional(value: str) -> str | None:
    return None if value == none_option else value

with st.expander("2. 열 역할 지정 및 생물학적 정의", expanded=True):
    st.markdown("**필수 입력 열**")
    c1, c2 = st.columns(2)
    with c1:
        seq_selected = pick_column(
            "gRNA / spacer 서열 열",
            guess_column(["guide_sequence", "grna_sequence", "spacer_sequence", "guide", "spacer"]),
        )
        target_default = "synthetic_activity_score" if "synthetic_activity_score" in columns else None
        target_selected = pick_column("예측할 목표값 열", target_default)
    with c2:
        cas_selected = pick_column(
            "Cas9 변이체 이름/종류 열",
            guess_column(["cas9_variant", "cas9", "nuclease", "cas9_name"]),
        )
        group_selected = pick_column(
            "그룹 분할용 식별자 열(예: study_id, guide_id)",
            guess_column(["study_id", "source_id", "guide_id", "target_id"]),
        )

    st.markdown("**선택적 생물학적·실험 맥락 열**")
    c3, c4 = st.columns(2)
    with c3:
        protein_selected = pick_column(
            "Cas9 아미노산 서열 열(실제 서열이 있을 때만)",
            guess_column(["cas9_sequence", "cas9_protein_sequence", "protein_sequence", "aa_sequence"]),
        )
        pam_selected = pick_column("PAM 서열 열", guess_column(["pam", "pam_sequence"]))
    with c4:
        target_context_selected = pick_column(
            "표적 DNA 주변 서열 열(실제 서열이 있을 때만)",
            guess_column(["target_context", "target_sequence", "flanking_sequence", "context_sequence"]),
        )

    excluded = {x for x in [target_selected, seq_selected, cas_selected, group_selected,
                            protein_selected, pam_selected, target_context_selected, none_option] if x}
    possible_contexts = [c for c in columns if c not in excluded]
    context_cols = st.multiselect(
        "추가 실험 맥락 열(예: cell_type, organism, assay_type)",
        options=possible_contexts,
        default=[],
        help="실제 의미를 확인한 세포 유형·생물종·측정법 등만 선택하세요. ID, 결과값, 효율 점수, 사후 계산 변수를 선택하지 마세요.",
    )

    sequence_col = optional(seq_selected)
    target_col = optional(target_selected)
    cas9_col = optional(cas_selected)
    group_col = optional(group_selected)
    protein_col = optional(protein_selected)
    pam_col = optional(pam_selected)
    target_context_col = optional(target_context_selected)
    if target_col is None:
        st.error("예측할 목표값 열을 선택해야 합니다.")
        st.stop()
    if not any([sequence_col, cas9_col, protein_col]):
        st.error("gRNA 서열 또는 Cas9 변이체/단백질 서열 중 하나 이상을 선택해야 합니다.")
        st.stop()

    selected_roles = [x for x in [target_col, sequence_col, cas9_col, group_col,
                                  protein_col, pam_col, target_context_col] if x]
    if len(selected_roles) != len(set(selected_roles)):
        st.error("같은 CSV 열을 서로 다른 역할에 중복 지정했습니다. 열 역할을 확인해 주세요.")
        st.stop()
    if any(x in {target_col, sequence_col, cas9_col, protein_col, pam_col, target_context_col, group_col}
           for x in context_cols):
        st.error("실험 맥락 열에 이미 다른 역할로 지정한 열이 포함되어 있습니다.")
        st.stop()

    target_label = st.text_input(
        "목표값의 정확한 의미(논문·데이터 사전 기준)",
        value="가상 활성 점수" if source_mode == "가상 데이터 데모" else "",
        placeholder="예: 표적 위치 indel frequency / normalized editing score",
    )
    c5, c6 = st.columns(2)
    with c5:
        target_unit = st.selectbox(
            "목표값 단위",
            ["단위 미확인", "퍼센트(0~100)", "비율(0~1)", "점수/기타 단위"],
            index=0,
        )
    with c6:
        definition_status = st.selectbox(
            "목표값 정의 확인 상태",
            ["아직 확인하지 않음", "논문·공식 데이터 설명에서 확인", "데이터 제공자 설명만 확인"],
            index=0 if source_mode == "CSV 직접 업로드" else 1,
        )
    split_mode = st.radio(
        "평가 데이터 분할 방식", ["무작위 분할", "그룹 기준 분할"],
        horizontal=True,
        help="guide_id 또는 study_id 등 평가 목적에 맞는 그룹 열을 선택하면 같은 그룹이 학습·평가에 나뉘는 문제를 줄일 수 있습니다.",
    )

cfg = {
    "sequence_col": sequence_col, "cas9_col": cas9_col, "protein_col": protein_col,
    "pam_col": pam_col, "target_context_col": target_context_col,
    "context_cols": context_cols,
}

# 결과가 바뀌는 설정이면 이전 학습 결과를 폐기한다.
signature = hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()
current_signature = (source_mode, signature, target_col, tuple(sorted((k, str(v)) for k, v in cfg.items())),
                     group_col, split_mode, target_label, target_unit, definition_status)
if st.session_state.get("crispr_signature") != current_signature:
    for key in ["crispr_models", "crispr_metrics", "crispr_test", "crispr_valid",
                "crispr_stats", "crispr_split", "crispr_split_info", "crispr_cfg",
                "crispr_target_info", "crispr_model_columns"]:
        st.session_state.pop(key, None)

if source_mode == "가상 데이터 데모":
    st.caption("가상 목표값은 임의의 규칙으로 생성한 0~100 점수입니다. 실제 indel frequency가 아닙니다.")

if st.button("데이터 검증 및 모델 학습", type="primary", use_container_width=True):
    try:
        valid, validation_stats = validate_and_prepare(df, target_col, sequence_col, cas9_col, protein_col, target_context_col, pam_col)
        if len(valid) < 20:
            raise ValueError(f"유효 행이 {len(valid)}개입니다. 현재 프로토타입의 최소 실행 기준은 20행입니다. 이는 과학적으로 충분한 표본 수를 보증하지 않습니다.")
        if split_mode == "그룹 기준 분할":
            if not group_col:
                raise ValueError("그룹 기준 분할을 선택했다면 그룹 열을 지정해야 합니다.")
            if group_col in {target_col, sequence_col, cas9_col, protein_col, pam_col, target_context_col} | set(context_cols):
                raise ValueError("그룹 분할용 열은 목표값이나 모델 입력 특징과 겹치지 않는 식별자 열이어야 합니다.")
        if cas9_col and valid["__cas9_clean__"].nunique() == 1:
            st.warning("Cas9 범주가 한 종류뿐입니다. Cas9 변이체 사이의 성능 차이를 평가할 수 없으므로 Cas9 단독·통합 모델의 해석에 주의하세요.")
        if target_unit == "퍼센트(0~100)":
            outside = (~valid["__target_numeric__"].between(0, 100)).sum()
            if outside:
                st.warning(f"목표값 {outside:,}개가 0~100 범위를 벗어납니다. 단위 선택 또는 원자료 정의를 다시 확인하세요.")
        elif target_unit == "비율(0~1)":
            outside = (~valid["__target_numeric__"].between(0, 1)).sum()
            if outside:
                st.warning(f"목표값 {outside:,}개가 0~1 범위를 벗어납니다. 단위 선택 또는 원자료 정의를 다시 확인하세요.")
        target_name_lower = str(target_col).lower()
        if re.search(r"off.?target|oof|offtarget", target_name_lower):
            st.warning("선택한 목표값 열 이름에 Off-target 관련 표현이 포함되어 있습니다. 이 모델은 선택한 Off-target 지표를 예측하는 것이며 On-target 편집 효율 예측과 구분해야 합니다.")
        if pam_col:
            st.info("PAM은 선택한 Cas9 종류와 원자료 정의에 맞는지 별도로 확인해야 합니다. 현재 앱은 PAM 문자열을 범주형 특징으로 처리하며, PAM 적합성을 생물학적으로 판정하지 않습니다.")
        if not target_label.strip() and source_mode == "CSV 직접 업로드":
            st.warning("목표값의 의미가 입력되지 않았습니다. 결과 지표는 계산되지만 생물학적 의미를 확정할 수 없습니다.")
        suspicious_context = [c for c in context_cols if re.search(r"id|activity|efficien|indel|score|outcome|result|frequency|percent|편집|효율|결과|빈도", c, re.I)]
        if suspicious_context:
            st.warning("다음 추가 맥락 열은 ID 또는 결과값에서 파생된 변수일 수 있습니다. 데이터 사전으로 입력 누수 여부를 확인하세요: " + ", ".join(suspicious_context))

        models, metrics, test_result, split_text, split_info, model_columns = evaluate_models(
            valid, cfg, split_mode, group_col
        )
        st.session_state["crispr_signature"] = current_signature
        st.session_state["crispr_models"] = models
        st.session_state["crispr_metrics"] = metrics
        st.session_state["crispr_test"] = test_result
        st.session_state["crispr_valid"] = valid
        st.session_state["crispr_stats"] = validation_stats
        st.session_state["crispr_split"] = split_text
        st.session_state["crispr_split_info"] = split_info
        st.session_state["crispr_cfg"] = cfg
        st.session_state["crispr_target_info"] = {
            "column": target_col, "label": target_label, "unit": target_unit,
            "definition_status": definition_status, "source_label": source_label,
        }
        st.session_state["crispr_model_columns"] = model_columns
        st.success("데이터 검증과 모델 평가가 완료되었습니다.")
    except Exception as exc:
        st.error(f"분석을 진행하지 못했습니다: {exc}")


# -----------------------------------------------------------------------------
# 결과 화면
# -----------------------------------------------------------------------------
if "crispr_metrics" in st.session_state:
    st.subheader("3. 데이터 검증 결과")
    stats = st.session_state["crispr_stats"]
    required_keys = ["전체 행", "목표값 숫자 변환 실패/결측", "최종 유효 행", "정확히 중복된 행"]
    cols = st.columns(len(required_keys))
    for col, key in zip(cols, required_keys):
        col.metric(key, f"{stats.get(key, '확인 불가'):,}" if isinstance(stats.get(key), int) else str(stats.get(key)))
    st.dataframe(pd.DataFrame([{"검사 항목": k, "결과": v} for k, v in stats.items()]), hide_index=True, use_container_width=True)
    if stats.get("gRNA 서열 검사") == "검사하지 않음":
        st.warning("gRNA 열을 선택하지 않아 gRNA 서열의 유효성을 검사하지 않았습니다. '잘못된 서열 0건'으로 해석하면 안 됩니다.")
    if stats.get("Cas9 결측 검사") == "검사하지 않음":
        st.info("Cas9 열을 선택하지 않아 Cas9 결측 여부를 검사하지 않았습니다.")
    st.caption(st.session_state["crispr_split"])
    split_info = st.session_state["crispr_split_info"]
    if st.session_state["crispr_split"].startswith("그룹 기준"):
        st.caption(f"학습 그룹 {split_info['학습 그룹']}개 · 평가 그룹 {split_info['평가 그룹']}개 · 겹치는 그룹 {split_info['겹치는 그룹']}개")

    st.subheader("4. 모델 성능 비교")
    cfg_used = st.session_state["crispr_cfg"]
    shared_roles = [x for x in [cfg_used.get("pam_col"), cfg_used.get("target_context_col")] if x]
    shared_roles += list(cfg_used.get("context_cols", []))
    if shared_roles:
        st.info("공통 실험 맥락 열(" + ", ".join(shared_roles) + ")은 gRNA·Cas9·통합 모델에 동일하게 포함됩니다. 따라서 비교는 '공통 맥락을 통제한 상태에서 gRNA 정보 또는 Cas9 정보가 추가로 제공하는 예측 차이'로 해석하세요.")
    metrics = st.session_state["crispr_metrics"].copy()
    display = metrics.copy()
    for col in ["MAE", "RMSE", "R²", "Spearman"]:
        display[col] = display[col].map(fmt_num)
    st.dataframe(display, use_container_width=True, hide_index=True)
    target_info = st.session_state["crispr_target_info"]
    if target_info["unit"] == "퍼센트(0~100)":
        unit_text = "퍼센트포인트(pp)"
    elif target_info["unit"] == "비율(0~1)":
        unit_text = "비율 단위(0~1)"
    else:
        unit_text = "목표값의 원래 단위(확인 필요)"
    st.markdown(
        f"**예측 목표:** `{target_info['column']}` / {target_info['label'] or '정의 미입력'}  \n"
        f"**단위:** {unit_text} · **정의 확인:** {target_info['definition_status']}"
    )
    if source_mode == "가상 데이터 데모":
        st.warning("가상 자료 결과입니다. 실제 CRISPR 편집 효율 또는 생물학적 관계를 입증하는 근거로 사용하면 안 됩니다.")
    elif target_info["definition_status"] == "아직 확인하지 않음" or not target_info["label"].strip() or target_info["unit"] == "단위 미확인":
        st.warning("실제 CSV 결과입니다. 다만 목표값의 정의·단위가 충분히 확인되지 않았으므로 MAE/RMSE의 생물학적 해석을 보류하세요.")
    else:
        st.info("목표값의 의미는 사용자가 선택한 자료에 근거한 입력입니다. 앱이 논문 원문이나 측정법을 자동 검증한 것은 아닙니다.")

    st.subheader("5. 결과 자동 해석")
    for message in make_interpretations(metrics, target_info, st.session_state["crispr_split"].split(" / ")[0]):
        st.markdown(f"- {message}")

    st.subheader("6. 실제값과 예측값 비교")
    test = st.session_state["crispr_test"]
    available_prediction_cols = [c for c in test.columns if c.startswith("예측값_")]
    if available_prediction_cols:
        selected_pred_col = st.selectbox("그래프로 확인할 모델", available_prediction_cols)
        scatter = pd.DataFrame({"실제 목표값": test["실제_목표값"], "AI 예측값": test[selected_pred_col]})
        st.scatter_chart(scatter, x="실제 목표값", y="AI 예측값", use_container_width=True)
        st.caption("점들이 대각선 방향으로 모일수록 실제값과 예측값이 비슷한 경향입니다. 이 그래프만으로 인과관계를 판단하지는 않습니다.")
        residuals = test["실제_목표값"].to_numpy() - test[selected_pred_col].to_numpy()
        st.metric("선택 모델의 평균 잔차(실제값 - 예측값)", f"{np.mean(residuals):.3f}")
        st.dataframe(test[["실제_목표값", selected_pred_col]].head(20), use_container_width=True)
    st.download_button(
        "평가 데이터의 실제값·예측값 CSV 내려받기",
        data=test.to_csv(index=False).encode("utf-8-sig"),
        file_name="CRISPR_AI_test_predictions.csv", mime="text/csv",
    )
    st.download_button(
        "모델 성능 지표 CSV 내려받기",
        data=metrics.to_csv(index=False).encode("utf-8-sig"),
        file_name="CRISPR_AI_model_metrics.csv", mime="text/csv",
    )

    st.subheader("7. 모델이 사용한 특징 확인")
    model_names = list(st.session_state["crispr_models"].keys())
    selected_model_name = st.selectbox("특징 중요도를 확인할 모델", model_names)
    selected_model = st.session_state["crispr_models"][selected_model_name]
    try:
        feature_names = selected_model.named_steps["preprocess"].get_feature_names_out()
        importances = selected_model.named_steps["model"].feature_importances_
        importance_df = pd.DataFrame({"특징": feature_names, "상대적 중요도": importances})
        importance_df = importance_df.sort_values("상대적 중요도", ascending=False).head(15)
        st.dataframe(importance_df, hide_index=True, use_container_width=True)
        st.caption("특징 중요도는 해당 모델의 예측에 사용된 상대적 기여 지표입니다. 생물학적 원인이나 인과관계를 뜻하지 않습니다.")
    except Exception as exc:
        st.info(f"특징 중요도를 표시하지 못했습니다: {exc}")


# -----------------------------------------------------------------------------
# 신규 데이터 예측
# -----------------------------------------------------------------------------
st.divider()
st.subheader("8. 신규 데이터 예측")
if "crispr_models" not in st.session_state:
    st.info("먼저 CSV를 업로드하고 모델을 학습해 주세요.")
else:
    new_upload = st.file_uploader("목표값이 없는 신규 CSV 업로드", type=["csv"], key="new_prediction_upload")
    if new_upload is not None:
        try:
            new_df = read_csv_safely(new_upload.getvalue())
            saved_cfg = st.session_state["crispr_cfg"]
            saved_models = st.session_state["crispr_models"]
            selected_model = st.selectbox("예측에 사용할 모델", list(saved_models.keys()))
            new_cols = list(new_df.columns)
            new_mapping: dict[str, Any] = {}
            roles = [
                ("sequence_col", "신규 파일의 gRNA 서열 열"),
                ("cas9_col", "신규 파일의 Cas9 변이체 열"),
                ("protein_col", "신규 파일의 Cas9 단백질 서열 열"),
                ("pam_col", "신규 파일의 PAM 열"),
                ("target_context_col", "신규 파일의 표적 주변 서열 열"),
            ]
            for role, label in roles:
                if saved_cfg.get(role):
                    chosen = st.selectbox(label, new_cols, key=f"new_{role}")
                    new_mapping[role] = chosen
                else:
                    new_mapping[role] = None
            new_contexts = []
            for i, old_col in enumerate(saved_cfg.get("context_cols", [])):
                chosen = st.selectbox(f"신규 파일의 실험 맥락 열: {old_col}", new_cols, key=f"new_context_{i}")
                new_contexts.append(chosen)
            new_mapping["context_cols"] = new_contexts

            if st.button("신규 데이터 예측 실행"):
                features = pd.DataFrame(index=new_df.index)
                if new_mapping.get("sequence_col"):
                    clean = new_df[new_mapping["sequence_col"]].map(clean_dna_sequence)
                    if clean.isna().any():
                        raise ValueError("신규 파일에 비어 있거나 형식이 잘못된 gRNA 서열이 있습니다.")
                    features = pd.concat([features, dna_sequence_features(clean, "gRNA")], axis=1)
                if new_mapping.get("cas9_col"):
                    cas = new_df[new_mapping["cas9_col"]].astype("string").str.strip()
                    if cas.isna().any() or cas.eq("").any():
                        raise ValueError("신규 파일에 Cas9 변이체 결측값이 있습니다.")
                    features = pd.concat([features, pd.DataFrame({"Cas9_변이체": cas.astype(str)}, index=new_df.index)], axis=1)
                if new_mapping.get("protein_col"):
                    prot = new_df[new_mapping["protein_col"]].map(clean_protein_sequence)
                    if prot.isna().any():
                        raise ValueError("신규 파일의 Cas9 단백질 서열에 결측 또는 허용되지 않은 문자가 있습니다.")
                    features = pd.concat([features, protein_sequence_features(prot)], axis=1)
                common_parts = []
                if new_mapping.get("pam_col"):
                    common_parts.append(pd.DataFrame({"공통_PAM": new_df[new_mapping["pam_col"]].astype("string").fillna("미기록").astype(str)}, index=new_df.index))
                if new_mapping.get("target_context_col"):
                    ctx = new_df[new_mapping["target_context_col"]].map(clean_dna_sequence)
                    common_parts.append(dna_sequence_features(ctx, "표적주변"))
                for i, new_col in enumerate(new_contexts):
                    old_name = saved_cfg["context_cols"][i]
                    feature_name = f"실험맥락_{old_name}"
                    s = new_df[new_col]
                    if feature_name in st.session_state["crispr_models"][selected_model].feature_names_in_:
                        if pd.api.types.is_numeric_dtype(st.session_state["crispr_valid"][old_name]):
                            common_parts.append(pd.DataFrame({feature_name: pd.to_numeric(s, errors="coerce")}, index=new_df.index))
                        else:
                            common_parts.append(pd.DataFrame({feature_name: s.astype("string").fillna("미기록").astype(str)}, index=new_df.index))
                if common_parts:
                    features = pd.concat([features, pd.concat(common_parts, axis=1)], axis=1)
                model = saved_models[selected_model]
                expected = list(model.feature_names_in_)
                unknown = [c for c in features.columns if c == "Cas9_변이체"]
                if unknown and "Cas9_변이체" in features:
                    trained = set(st.session_state["crispr_valid"]["__cas9_clean__"].astype(str)) if "__cas9_clean__" in st.session_state["crispr_valid"] else set()
                    unseen = sorted(set(features["Cas9_변이체"].astype(str)) - trained)
                    if unseen:
                        st.warning("학습에서 보지 못한 Cas9 범주가 있습니다: " + ", ".join(unseen))
                features = features.reindex(columns=expected)
                pred = model.predict(features)
                output = new_df.copy()
                output["모델_예측값"] = pred
                ti = st.session_state["crispr_target_info"]
                output["예측값_목표_정의"] = ti["label"] or ti["column"]
                output["예측값_단위"] = ti["unit"]
                output["예측값_상태"] = "모델 예측값(실측값 아님)"
                st.success(f"{len(output):,}개 행의 예측을 완료했습니다.")
                st.dataframe(output.head(20), use_container_width=True)
                st.download_button(
                    "신규 데이터 예측 결과 CSV 내려받기",
                    data=output.to_csv(index=False).encode("utf-8-sig"),
                    file_name="CRISPR_AI_new_data_predictions.csv", mime="text/csv",
                )
        except Exception as exc:
            st.error(f"신규 데이터 예측을 진행하지 못했습니다: {exc}")

st.divider()
st.caption(
    "현재 버전은 CSV 기반 통계적 예측 프로토타입입니다. 위치별 염기·k-mer는 서열 패턴 특징이며, "
    "RNA 2차 구조·Cas9 3차원 구조·실제 오프타겟 절단을 직접 계산하지 않습니다. "
    "XLSX, 자동 논문 출처 검증, 실제 시퀀싱 read 기반 indel 계산, 오프타겟 유전체 검색, 영구 모델 저장은 구현되어 있지 않습니다."
)
