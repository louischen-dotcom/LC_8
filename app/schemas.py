import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


_FEATURE_STATS_PATH = Path(__file__).with_name("feature_stats.json")
_FEATURE_STATS: dict[str, Any] = json.loads(
    _FEATURE_STATS_PATH.read_text(encoding="utf-8")
)

MODEL_FEATURES: tuple[str, ...] = tuple(_FEATURE_STATS["model_features"])
FEATURE_MEDIANS: dict[str, float] = {
    feature: float(value)
    for feature, value in _FEATURE_STATS["feature_medians"].items()
}
TOP_SHAP_FEATURES: tuple[str, ...] = tuple(_FEATURE_STATS["top_shap_features"])

_MISSING_MEDIANS = set(MODEL_FEATURES) - set(FEATURE_MEDIANS)
if _MISSING_MEDIANS:
    raise RuntimeError(f"Missing feature medians: {sorted(_MISSING_MEDIANS)}")


class CreditApplication(BaseModel):
    model_config = ConfigDict(extra="forbid")

    EXT_SOURCE_MEAN: FiniteFloat = Field(
        0.5245019540617536,
        ge=5.9396509293128426e-06,
        le=0.878903368632926,
        description="Mean of the external credit source scores",
    )
    INST_PAYMENT_DELAY_MAX: int = Field(
        1,
        ge=-156,
        le=2884,
        description="Maximum observed installment payment delay in days",
    )
    AMT_CREDIT: FiniteFloat = Field(
        513531.0,
        ge=45000.0,
        le=4050000.0,
        description="Requested credit amount",
    )
    CODE_GENDER_M: int = Field(
        0,
        ge=0,
        le=1,
        description="1 if applicant is male, else 0",
    )
    PREV_CNT_PAYMENT_MEAN: FiniteFloat = Field(
        12.0,
        ge=0.0,
        le=72.0,
        description="Mean number of installments on previous applications",
    )
    AMT_GOODS_PRICE: FiniteFloat = Field(
        450000.0,
        ge=40500.0,
        le=4050000.0,
        description="Goods price linked to the requested credit",
    )
    EXT_SOURCE_MIN: FiniteFloat = Field(
        0.4031670546503185,
        ge=8.173616518884397e-08,
        le=0.878903368632926,
        description="Minimum of the external credit source scores",
    )
    POS_COUNT: int = Field(
        21,
        ge=0,
        le=295,
        description="Number of POS cash balance records",
    )
    AMT_ANNUITY: FiniteFloat = Field(
        24903.0,
        ge=1615.5,
        le=258025.5,
        description="Loan annuity amount",
    )
    INST_AMT_PAYMENT_SUM: FiniteFloat = Field(
        289330.02,
        ge=0.0,
        le=25537053.78,
        description="Total amount paid on previous installments",
    )
    POS_CNT_INSTALMENT: FiniteFloat = Field(
        11.909090909090908,
        ge=0.0,
        le=72.0,
        description="Mean number of installments in POS cash balances",
    )
    NAME_FAMILY_STATUS_Married: int = Field(
        1,
        ge=0,
        le=1,
        description="1 if applicant is married, else 0",
    )
    NAME_EDUCATION_TYPE_Higher_education: int = Field(
        0,
        ge=0,
        le=1,
        description="1 if applicant has higher education, else 0",
    )
    BUREAU_CREDIT_DEBT_SUM: FiniteFloat = Field(
        87583.5,
        ge=-6981558.210000001,
        le=334498331.20500004,
        description="Total debt amount reported by the credit bureau",
    )
    PREV_AMT_DOWN_PAYMENT_MAX: FiniteFloat = Field(
        4500.0,
        ge=0.0,
        le=3060045.0,
        description="Maximum down payment amount on previous applications",
    )
    DAYS_BIRTH: int = Field(
        -15750,
        ge=-25229,
        le=-7489,
        description="Applicant age as days before application",
    )
    PREV_REFUSED_COUNT: int = Field(
        0,
        ge=0,
        le=68,
        description="Number of refused previous applications",
    )
    DAYS_ID_PUBLISH: int = Field(
        -3254,
        ge=-7197,
        le=0,
        description="Days before application when identity document was published",
    )
    EXT_SOURCE_3: FiniteFloat = Field(
        0.5352762504724826,
        ge=0.0005272652387098,
        le=0.8960095494948396,
        description="External source score 3",
    )
    BUREAU_CREDIT_SUM: FiniteFloat = Field(
        711000.0,
        ge=0.0,
        le=1017957917.385,
        description="Total credit amount reported by the credit bureau",
    )

    @classmethod
    def feature_names(cls) -> tuple[str, ...]:
        return MODEL_FEATURES

    @classmethod
    def exposed_feature_names(cls) -> tuple[str, ...]:
        return TOP_SHAP_FEATURES

    def to_model_input(self) -> dict[str, float]:
        features = FEATURE_MEDIANS.copy()
        features.update(
            {feature: float(value) for feature, value in self.model_dump().items()}
        )
        return {feature: features[feature] for feature in MODEL_FEATURES}


if tuple(CreditApplication.model_fields) != TOP_SHAP_FEATURES:
    raise RuntimeError("CreditApplication fields must match TOP_SHAP_FEATURES")


class PredictionResponse(BaseModel):
    prediction: Literal[0, 1] = Field(description="0 = no default, 1 = default")
    probability_of_default: float = Field(
        ge=0.0,
        le=1.0,
        description="Estimated probability of default from 0.0 to 1.0",
    )
    risk_category: Literal["Low", "Medium", "High"] = Field(
        description="Risk category derived from the default probability"
    )


class HealthResponse(BaseModel):
    status: str = Field(description="Service health status")
    model_loaded: bool = Field(description="Whether the MLflow model is loaded")
