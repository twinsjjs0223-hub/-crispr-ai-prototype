# -*- coding: utf-8 -*-
"""CRISPR-AI: 공개/가상 데이터 기반 효율 예측 프로토타입

주의: 이 앱은 업로드한 표 형식 데이터를 분석하는 교육·연구용 프로토타입입니다.
실제 유전자 편집 실험을 수행하지 않으며, 예측값은 실험 결과나 안전성 보증이 아닙니다.
"""

from __future__ import annotations

import io
import re
import hashlib
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


APP_TITLE = "CRISPR-AI | 편집 효율 데이터 분석 프로토타입"
RANDOM_STATE = 42


# -----------------------------------------------------------------------------
# 데이터 불러오기 및 가상 데이터 생성
# -----------------------------------------------------------------------------
def read_csv_safely(file_bytes: bytes) -> pd.DataFrame:
    """UTF-8 계열 또는 한국어 CSV 인코딩을 순서대로 시도한다."""
    errors: list[str] = []
    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            text = file_bytes.decode(encoding)
            frame = pd.read_csv(io.StringIO(text), sep=None, engine="python")
            break
        except Exception as exc:  # 인코딩/구분자 문제를 다음 후보에서 재시도
            errors.append(f"{encoding}: {exc}")
    else:
        raise ValueError(
            "CSV를 읽지 못했습니다. UTF-8 또는 CP949로 저장했는지 확인해 주세요.\n"
            + "\n".join(errors)
        )

    # 열 이름의 앞뒤 공백/BOM 정리 및 중복 열 이름 보정
    cleaned: list[str] = []
    used: dict[str, int] = {}
    for raw_name in frame.columns:
        name = str(raw_name).replace("\ufeff", "").strip()
        if not name:
            name = "이름없는열"
        used[name] = used.get(name, 0) + 1
        cleaned.append(name if used[name] == 1 else f"{name}_{used[name]}")
    frame.columns = cleaned
    frame = frame.dropna(how="all").copy()
    if frame.empty:
        raise ValueError("CSV에 분석할 행이 없습니다.")
    return frame


def make_synthetic_data(n_rows: int = 240) -> pd.DataFrame:
    """기능 점검 전용 가상 데이터. 생물학적 측정값을 모사한 자료가 아니다."""
    rng = np.random.default_rng(20261101)
    bases = np.array(list("ACGT"))
    sequences = ["".join(rng.choice(bases, size=20)) for _ in range(n_rows)]

    variants = np.array(["SpCas9", "SpCas9-HF1", "eSpCas9(1.1)", "SaCas9"])
    cas9 = rng.choice(variants, size=n_rows, p=[0.35, 0.25, 0.25, 0.15])

    # 아래 계수는 화면·학습·평가 흐름을 시험하려고 임의로 정한 규칙이다.
    # 실제 Cas9 또는 sgRNA의 생물학적 효과를 나타내지 않는다.
    variant_offset = {
        "SpCas9": 8.0,
        "SpCas9-HF1": -5.0,
        "eSpCas9(1.1)": 2.0,
        "SaCas9": -8.0,
    }
    variant_interaction = {
        "SpCas9": 10.0,
        "SpCas9-HF1": -6.0,
        "eSpCas9(1.1)": 3.0,
        "SaCas9": -4.0,
    }
    gc_pct = np.array(
        [100.0 * (seq.count("G") + seq.count("C")) / len(seq) for seq in sequences]
    )
    y = (
        47.0
        + 0.75 * (gc_pct - 50.0)
        + np.array([variant_offset[item] for item in cas9])
        + np.array([variant_interaction[item] for item in cas9]) * (gc_pct - 50.0) / 20.0
        + rng.normal(0.0, 7.0, n_rows)
    )
    y = np.clip(y, 0.0, 100.0).round(2)

    return pd.DataFrame(
        {
            "record_id": [f"SYN-{i + 1:03d}" for i in range(n_rows)],
            "guide_sequence": sequences,
            "cas9_variant": cas9,
            "synthetic_activity_score": y,
            "data_status": "synthetic",
        }
    )


# -----------------------------------------------------------------------------
# 입력 검증 및 특징 추출
# -----------------------------------------------------------------------------
def clean_dna_sequence(value: Any) -> str | None:
    if pd.isna(value):
        return None
    seq = re.sub(r"\s+", "", str(value)).upper().replace("U", "T")
    if not seq or re.fullmatch(r"[ACGTN]+", seq) is None:
        return None
    return seq


def sequence_features(series: pd.Series) -> pd.DataFrame:
    sequences = series.map(clean_dna_sequence)
    if sequences.isna().any():
        raise ValueError("서열 특징 계산 전에 잘못된 서열이 포함되어 있습니다.")
    records = []
    for seq in sequences:
        length = len(seq)
        records.append(
            {
                "서열_길이": length,
                "GC_비율(%)": 100.0 * (seq.count("G") + seq.count("C")) / length,
                "A_개수": seq.count("A"),
                "C_개수": seq.count("C"),
                "G_개수": seq.count("G"),
                "T_개수": seq.count("T"),
                "N_개수": seq.count("N"),
            }
        )
    return pd.DataFrame(records, index=series.index)


def validate_and_prepare(
    frame: pd.DataFrame,
    target_col: str,
    sequence_col: str | None,
    cas9_col: str | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """모델 입력에 필요한 값을 확인하고 유효 행만 남긴다."""
    work = frame.copy()
    y = pd.to_numeric(work[target_col], errors="coerce")
    target_ok = y.notna() & np.isfinite(y)
    stats = {
        "전체 행": len(work),
        "목표값 숫자 변환 실패/결측": int((~target_ok).sum()),
        "잘못된 gRNA 서열": 0,
        "Cas9 값 결측": 0,
        "최종 유효 행": 0,
    }
    work["__target_numeric__"] = y
    mask = target_ok.copy()

    if sequence_col:
        cleaned = work[sequence_col].map(clean_dna_sequence)
        invalid = cleaned.isna()
        stats["잘못된 gRNA 서열"] = int(invalid.sum())
        work["__sequence_clean__"] = cleaned
        mask &= ~invalid

    if cas9_col:
        cas_values = work[cas9_col].astype("string").str.strip()
        missing = cas_values.isna() | cas_values.eq("")
        stats["Cas9 값 결측"] = int(missing.sum())
        work["__cas9_clean__"] = cas_values.fillna("")
        mask &= ~missing

    valid = work.loc[mask].copy()
    stats["최종 유효 행"] = len(valid)
    return valid, stats


# -----------------------------------------------------------------------------
# 모델 학습 및 평가
# -----------------------------------------------------------------------------
def build_model(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    transformers = []
    if numeric_cols:
        numeric_pipe = Pipeline(
            [("imputer", SimpleImputer(strategy="median"))]
        )
        transformers.append(("numeric", numeric_pipe, numeric_cols))
    if categorical_cols:
        categorical_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipe, categorical_cols))

    preprocess = ColumnTransformer(transformers=transformers, remainder="drop")
    forest = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocess), ("model", forest)])


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    spearman = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R²": float(r2_score(y_true, y_pred)),
        "Spearman": None if pd.isna(spearman) else float(spearman),
    }


def evaluate_selected_models(
    valid: pd.DataFrame,
    target_col: str,
    sequence_col: str | None,
    cas9_col: str | None,
    split_mode: str,
    group_col: str | None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, str]:
    if not sequence_col and not cas9_col:
        raise ValueError("gRNA 서열 열 또는 Cas9 변이체 열 중 하나 이상을 선택해야 합니다.")

    y = valid["__target_numeric__"].to_numpy(dtype=float)
    numeric = sequence_features(valid["__sequence_clean__"]) if sequence_col else None
    categorical = (
        pd.DataFrame({"Cas9_변이체": valid["__cas9_clean__"].astype(str)}, index=valid.index)
        if cas9_col
        else None
    )

    if split_mode == "그룹 기준 분할":
        if not group_col:
            raise ValueError("그룹 기준 분할을 선택했다면 그룹 열도 지정해야 합니다.")
        groups = valid[group_col].astype("string").fillna("__결측그룹__")
        if groups.nunique() < 2:
            raise ValueError("그룹이 2개 미만이라 그룹 분할을 수행할 수 없습니다.")
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
        train_idx, test_idx = next(splitter.split(valid, y, groups=groups))
        split_description = f"그룹 기준 분할: {group_col} / 학습 {len(train_idx)}행, 평가 {len(test_idx)}행"
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(valid)), test_size=0.25, random_state=RANDOM_STATE
        )
        split_description = f"무작위 75:25 분할 / 학습 {len(train_idx)}행, 평가 {len(test_idx)}행"

    model_specs: list[tuple[str, pd.DataFrame, list[str], list[str]]] = []
    if numeric is not None:
        model_specs.append(("gRNA 단독", numeric, list(numeric.columns), []))
    if categorical is not None:
        model_specs.append(("Cas9 단독", categorical, [], list(categorical.columns)))
    if numeric is not None and categorical is not None:
        combined = pd.concat([numeric, categorical], axis=1)
        model_specs.append(
            ("gRNA + Cas9 통합", combined, list(numeric.columns), list(categorical.columns))
        )

    models: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    metric_rows: list[dict[str, Any]] = []

    # 평균값만 예측하는 기준선: 모델이 단순 평균보다 나은지 확인한다.
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(np.zeros((len(train_idx), 1)), y[train_idx])
    dummy_pred = dummy.predict(np.zeros((len(test_idx), 1)))
    dummy_metrics = calculate_metrics(y[test_idx], dummy_pred)
    metric_rows.append({"모델": "평균값 기준선", **dummy_metrics})

    for model_name, X, numeric_cols, categorical_cols in model_specs:
        model = build_model(numeric_cols, categorical_cols)
        model.fit(X.iloc[train_idx], y[train_idx])
        pred = model.predict(X.iloc[test_idx])
        models[model_name] = model
        predictions[model_name] = pred
        metric_rows.append({"모델": model_name, **calculate_metrics(y[test_idx], pred)})

    internal_cols = [col for col in valid.columns if col.startswith("__")]
    test_result = valid.iloc[test_idx].drop(columns=internal_cols, errors="ignore").copy()
    test_result["평가용_기준값"] = y[test_idx]
    for name, pred in predictions.items():
        test_result[f"예측값_{name}"] = pred

    metrics = pd.DataFrame(metric_rows)
    return models, metrics, test_result, split_description


# -----------------------------------------------------------------------------
# 화면
# -----------------------------------------------------------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🧬", layout="wide")
st.title(APP_TITLE)
st.caption("CSV 데이터의 구조 점검, 회귀 모델 비교, 예측 결과 다운로드를 위한 한국어 프로토타입")
st.warning(
    "연구상 주의: 가상 데이터 점수는 프로그램 작동 확인용입니다. 실제 생물학적 효율이나 안전성, "
    "오프타겟 위험을 입증하지 않습니다."
)

with st.sidebar:
    st.header("데이터 선택")
    source_mode = st.radio("분석할 데이터", ["가상 데이터 데모", "CSV 직접 업로드"])

if source_mode == "가상 데이터 데모":
    df = make_synthetic_data()
    source_label = "가상 테스트 데이터"
    st.info("현재 화면은 난수와 임의 규칙으로 만든 240행의 가상 데이터입니다. 실제 실험 자료가 아닙니다.")
    st.download_button(
        "가상 데이터 CSV 내려받기",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="CRISPR_AI_synthetic_demo.csv",
        mime="text/csv",
    )
else:
    upload = st.file_uploader("CSV 파일 업로드", type=["csv"])
    if upload is None:
        st.info("분석할 CSV 파일을 업로드해 주세요.")
        st.stop()
    try:
        df = read_csv_safely(upload.getvalue())
    except Exception as exc:
        st.error(f"파일을 읽지 못했습니다: {exc}")
        st.stop()
    source_label = "사용자 업로드 CSV(출처 확인 필요)"

st.subheader("1. 데이터 미리보기")
st.caption(f"데이터 유형: {source_label} · 전체 행: {len(df):,} · 열: {len(df.columns)}개")
if "data_status" in df.columns:
    statuses = df["data_status"].astype(str).str.lower().unique().tolist()
    if any(("synthetic" in item) or ("virtual" in item) or ("가상" in item) for item in statuses):
        st.warning("data_status 열에서 가상 데이터 표시를 확인했습니다. 실제 생물학적 성능으로 해석하지 마세요.")
st.dataframe(df.head(12), use_container_width=True)

columns = list(df.columns)
none_option = "(사용 안 함)"


def pick_column(label: str, candidates: list[str], default: str | None = None) -> str:
    options = [none_option] + columns
    index = options.index(default) if default in options else 0
    return st.selectbox(label, options, index=index, key=f"map_{label}")


def to_optional(value: str) -> str | None:
    return None if value == none_option else value

with st.expander("2. 열 역할 지정", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        sequence_default = "guide_sequence" if "guide_sequence" in columns else None
        seq_selected = pick_column("gRNA / spacer 서열 열", columns, sequence_default)
        # 실제 데이터에서는 'activity'라는 이름만으로 목표값 의미를 단정하지 않는다.
        target_default = "synthetic_activity_score" if "synthetic_activity_score" in columns else None
        target_col = pick_column("예측할 목표값 열", columns, target_default)
    with col2:
        cas_default = "cas9_variant" if "cas9_variant" in columns else None
        cas_selected = pick_column("Cas9 변이체 열", columns, cas_default)
        group_selected = pick_column("그룹 분할용 열(예: guide_id, study_id)", columns)

    sequence_col = to_optional(seq_selected)
    cas9_col = to_optional(cas_selected)
    group_col = to_optional(group_selected)
    if sequence_col and cas9_col:
        st.info(
            "통합 모델을 해석하려면 각 행의 gRNA와 Cas9 정보가 같은 실험 관측치에서 "
            "실제로 함께 기록된 조합이어야 합니다. 서로 다른 자료를 임의로 합친 행은 사용하지 마세요."
        )
    if target_col == none_option:
        st.error("예측할 목표값 열을 지정해야 합니다.")
        st.stop()

    target_label = st.text_input(
        "목표값의 정확한 의미(데이터 사전·논문 정의에 맞게 입력)",
        value=("가상 활성 점수" if source_mode == "가상 데이터 데모" else "편집 활성 측정값"),
    )
    target_unit = st.selectbox(
        "목표값 단위",
        ["단위 미확인 점수", "퍼센트(0~100)", "비율(0~1)"],
        index=0 if source_mode == "가상 데이터 데모" else 0,
    )
    split_mode = st.radio(
        "평가 데이터 분할 방식",
        ["무작위 분할", "그룹 기준 분할"],
        horizontal=True,
        help="같은 guide/표적/연구 출처가 학습과 평가에 섞이지 않게 하려면 그룹 기준 분할을 검토하세요.",
    )

current_signature = (
    hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest(),
    target_col, sequence_col, cas9_col, group_col, split_mode, target_label, target_unit
)
if "crispr_signature" in st.session_state and st.session_state["crispr_signature"] != current_signature:
    for state_key in ["crispr_models", "crispr_mapping", "crispr_metrics", "crispr_test_result", "crispr_valid", "crispr_validation_stats", "crispr_split_description", "crispr_signature"]:
        st.session_state.pop(state_key, None)

if source_mode == "가상 데이터 데모":
    st.caption("가상 데이터의 목표값은 0~100 범위의 임의 점수입니다. 실제 indel 빈도(%)가 아닙니다.")

if st.button("데이터 검증 및 모델 학습", type="primary", use_container_width=True):
    try:
        valid, validation_stats = validate_and_prepare(
            df, target_col, sequence_col, cas9_col
        )
        if cas9_col and valid["__cas9_clean__"].nunique() < 2:
            raise ValueError(
                "선택한 Cas9 열의 고유 범주가 1개뿐입니다. 이 상태에서는 Cas9 차이를 학습할 수 없습니다. "
                "gRNA 모델만 분석하려면 Cas9 열을 '(사용 안 함)'으로 바꾸거나, 서로 다른 Cas9 범주가 포함된 자료를 사용하세요."
            )
        if len(valid) < 12:
            raise ValueError(
                f"유효 행이 {len(valid)}개입니다. 현재 프로토타입의 최소 실행 기준(12행)에 미달합니다. "
                "이는 과학적으로 충분한 표본 수를 뜻하지 않습니다."
            )
        if split_mode == "그룹 기준 분할" and not group_col:
            raise ValueError("그룹 분할용 열을 지정해 주세요.")
        if group_col and group_col in {target_col, sequence_col, cas9_col}:
            raise ValueError("그룹 분할 열은 목표값·서열·Cas9 열과 다른 식별자 열이어야 합니다.")

        # 두 입력이 모두 있으면 두 변수 모두가 존재하는 동일 행으로 공정 비교한다.
        models, metrics, test_result, split_description = evaluate_selected_models(
            valid,
            target_col,
            sequence_col,
            cas9_col,
            split_mode,
            group_col,
        )
        signature = hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()
        st.session_state["crispr_signature"] = (signature, target_col, sequence_col, cas9_col, group_col, split_mode, target_label, target_unit)
        st.session_state["crispr_models"] = models
        st.session_state["crispr_mapping"] = {
            "sequence_col": sequence_col,
            "cas9_col": cas9_col,
            "target_label": target_label,
            "target_unit": target_unit,
            "source_label": source_label,
        }
        st.session_state["crispr_metrics"] = metrics
        st.session_state["crispr_test_result"] = test_result
        st.session_state["crispr_valid"] = valid
        st.session_state["crispr_validation_stats"] = validation_stats
        st.session_state["crispr_split_description"] = split_description
        st.success("평가용 데이터에서 모델 학습과 지표 계산을 마쳤습니다.")
    except Exception as exc:
        st.error(f"분석을 진행하지 못했습니다: {exc}")

if "crispr_metrics" in st.session_state:
    st.subheader("3. 데이터 검증 결과")
    stat_cols = st.columns(5)
    stats = st.session_state["crispr_validation_stats"]
    for col, key in zip(stat_cols, ["전체 행", "목표값 숫자 변환 실패/결측", "잘못된 gRNA 서열", "Cas9 값 결측", "최종 유효 행"]):
        col.metric(key, f"{stats[key]:,}")
    st.caption(st.session_state["crispr_split_description"])

    st.subheader("4. 모델 성능 비교")
    metrics = st.session_state["crispr_metrics"].copy()
    metrics_display = metrics.copy()
    for metric_col in ["MAE", "RMSE", "R²", "Spearman"]:
        metrics_display[metric_col] = metrics_display[metric_col].map(
            lambda x: "계산 불가" if pd.isna(x) else f"{x:.3f}"
        )
    st.dataframe(metrics_display, use_container_width=True, hide_index=True)

    mapping = st.session_state["crispr_mapping"]
    if mapping["target_unit"] == "퍼센트(0~100)":
        unit_text = "퍼센트포인트(pp)"
    elif mapping["target_unit"] == "비율(0~1)":
        unit_text = "비율 단위"
    else:
        unit_text = "목표값의 원래 단위"
    st.markdown(
        f"**목표값:** {mapping['target_label']} · **MAE/RMSE 단위:** {unit_text}. "
        "MAE와 RMSE는 작을수록 평가 데이터에서 예측 오차가 작습니다. "
        "R²는 음수가 될 수도 있으며, Spearman은 실제값과 예측값의 순위 관계를 봅니다."
    )
    if source_mode == "가상 데이터 데모" or mapping["source_label"] == "가상 테스트 데이터":
        st.warning("이 지표는 가상 생성 규칙을 재현한 결과입니다. 실제 CRISPR 효율 예측 성능으로 인용하면 안 됩니다.")
    else:
        st.warning("CSV의 출처·목표값 정의·단위가 공식 자료로 확인되기 전에는 생물학적 의미를 확정하지 마세요.")

    result = st.session_state["crispr_test_result"]
    result_download = result.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "평가 데이터의 실제값·예측값 CSV 내려받기",
        data=result_download,
        file_name="CRISPR_AI_test_predictions.csv",
        mime="text/csv",
    )

    metric_download = metrics.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "모델 성능 지표 CSV 내려받기",
        data=metric_download,
        file_name="CRISPR_AI_model_metrics.csv",
        mime="text/csv",
    )

st.divider()
st.subheader("5. 신규 데이터 예측")
if "crispr_models" not in st.session_state:
    st.info("먼저 위에서 데이터를 검증하고 모델을 학습해 주세요.")
else:
    new_file = st.file_uploader("목표값이 없는 신규 CSV 업로드", type=["csv"], key="new_prediction_upload")
    if new_file is not None:
        try:
            new_df = read_csv_safely(new_file.getvalue())
            saved_models = st.session_state["crispr_models"]
            saved_mapping = st.session_state["crispr_mapping"]
            available_models = list(saved_models.keys())
            selected_model = st.selectbox("사용할 학습 모델", available_models)

            new_cols = list(new_df.columns)
            saved_seq_col = saved_mapping["sequence_col"]
            saved_cas_col = saved_mapping["cas9_col"]
            seq_new = None
            cas_new = None
            if "gRNA 단독" in selected_model or selected_model == "gRNA + Cas9 통합":
                seq_new = st.selectbox("신규 파일의 gRNA 서열 열", new_cols)
            if "Cas9 단독" in selected_model or selected_model == "gRNA + Cas9 통합":
                cas_new = st.selectbox("신규 파일의 Cas9 변이체 열", new_cols)

            if st.button("신규 데이터 예측 실행"):
                features = []
                if seq_new:
                    clean_seq = new_df[seq_new].map(clean_dna_sequence)
                    if clean_seq.isna().any():
                        raise ValueError("신규 파일에 비어 있거나 형식이 잘못된 gRNA 서열이 있습니다.")
                    features.append(sequence_features(clean_seq))
                if cas_new:
                    cas_values = new_df[cas_new].astype("string").str.strip()
                    if cas_values.isna().any() or cas_values.eq("").any():
                        raise ValueError("신규 파일에 Cas9 변이체 값이 비어 있는 행이 있습니다.")
                    features.append(pd.DataFrame({"Cas9_변이체": cas_values.astype(str)}, index=new_df.index))
                X_new = pd.concat(features, axis=1)
                if cas_new and ("Cas9 단독" in selected_model or selected_model == "gRNA + Cas9 통합"):
                    trained_categories = set(st.session_state["crispr_valid"]["__cas9_clean__"].astype(str).unique())
                    unknown_categories = sorted(set(cas_values.astype(str).unique()) - trained_categories)
                    if unknown_categories:
                        st.warning("학습에 없던 Cas9 범주가 포함되어 있습니다. 해당 범주는 원-핫 인코딩에서 모두 0으로 처리될 수 있어 예측 해석에 주의해야 합니다: " + ", ".join(unknown_categories))
                pred = saved_models[selected_model].predict(X_new)
                output = new_df.copy()
                output["모델_예측값"] = pred
                output["예측값_목표_정의"] = saved_mapping["target_label"]
                output["예측값_단위"] = saved_mapping["target_unit"]
                output["예측값_상태"] = "모델 예측값(실측값 아님)"
                st.success(f"{len(output):,}개 행의 예측을 완료했습니다.")
                st.dataframe(output.head(20), use_container_width=True)
                st.download_button(
                    "신규 데이터 예측 결과 CSV 내려받기",
                    data=output.to_csv(index=False).encode("utf-8-sig"),
                    file_name="CRISPR_AI_new_data_predictions.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.error(f"신규 데이터 예측을 진행하지 못했습니다: {exc}")

st.divider()
st.caption(
    "현재 버전의 제한: CSV만 지원하며, 모델은 실행 세션 안에서만 유지됩니다. "
    "앱을 다시 시작하면 학습 결과가 사라질 수 있습니다. XLSX, 영구 모델 저장, "
    "자동 출처 검증, 오프타겟 예측은 구현되어 있지 않습니다."
)
