"""Application leg: applicant and loan attributes at the point of application."""

from __future__ import annotations

import numpy as np
import pandas as pd

from acr.core.io import read
from acr.core.legs.common import (
    DAYS_PER_YEAR,
    FeatureDict,
    LegResult,
    clean_days,
    finalise,
    ratio,
)  # fmt: skip
from acr.core.schema import APP_CATEGORICAL, APP_NUMERIC, APP_PROTECTED, DAYS_SENTINEL, ID

SRC = "application_train"

RAW_NUMERIC_DEFS = {
    "EXT_SOURCE_1": "External normalised score 1 (provider undisclosed by Home Credit)",
    "EXT_SOURCE_2": "External normalised score 2 (provider undisclosed by Home Credit)",
    "EXT_SOURCE_3": "External normalised score 3 (provider undisclosed by Home Credit)",
    "AMT_INCOME_TOTAL": "Declared annual income",
    "AMT_CREDIT": "Credit amount of the loan",
    "AMT_ANNUITY": "Loan annuity (periodic instalment)",
    "AMT_GOODS_PRICE": "Price of goods the loan finances",
    "CNT_CHILDREN": "Number of children",
    "CNT_FAM_MEMBERS": "Number of family members",
    "REGION_POPULATION_RELATIVE": "Normalised population of the client's region",
    "REGION_RATING_CLIENT_W_CITY": "Home Credit's rating of the client's region and city (1-3)",
    "OWN_CAR_AGE": "Age of the client's car in years",
    "AMT_REQ_CREDIT_BUREAU_MON": "Bureau enquiries in the month before application",
    "AMT_REQ_CREDIT_BUREAU_QRT": "Bureau enquiries in the 3 months before application",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "Bureau enquiries in the year before application",
    "DEF_30_CNT_SOCIAL_CIRCLE": "Social-circle contacts 30+ DPD",
    "DEF_60_CNT_SOCIAL_CIRCLE": "Social-circle contacts 60+ DPD",
    "OBS_30_CNT_SOCIAL_CIRCLE": "Social-circle contacts with observable 30 DPD history",
    "FLAG_DOCUMENT_3": "Provided document 3",
    "REG_CITY_NOT_WORK_CITY": "Registered city differs from work city",
    "REG_CITY_NOT_LIVE_CITY": "Registered city differs from living city",
    "FLAG_WORK_PHONE": "Provided a work phone",
    "FLAG_PHONE": "Provided a home phone",
    "TOTALAREA_MODE": "Normalised total area of the client's building (mode)",
}
CATEGORICAL_DEFS = {
    "NAME_CONTRACT_TYPE": "Contract type (cash or revolving)",
    "FLAG_OWN_CAR": "Owns a car (Y/N)",
    "FLAG_OWN_REALTY": "Owns real estate (Y/N)",
    "NAME_INCOME_TYPE": "Income type (working, pensioner, ...)",
    "NAME_EDUCATION_TYPE": "Highest education level",
    "NAME_HOUSING_TYPE": "Housing situation",
    "OCCUPATION_TYPE": "Occupation",
    "ORGANIZATION_TYPE": "Employer organisation type",
}
PROTECTED_DEFS = {
    "CODE_GENDER": "Gender. Protected characteristic: segment and fairness analysis only",
    "NAME_FAMILY_STATUS": "Marital status. Protected characteristic: segment and fairness analysis only",
}


def build(paths: dict[str, str], ids: np.ndarray) -> LegResult:
    app = read(paths["application"], [ID, *APP_NUMERIC, *APP_CATEGORICAL, *APP_PROTECTED], ids)
    app = app.set_index(ID)
    fd = FeatureDict("application")
    f = pd.DataFrame(index=app.index)

    skip = {"DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
            "DAYS_LAST_PHONE_CHANGE"}  # fmt: skip
    for col in APP_NUMERIC:
        if col not in skip:
            f[fd.add(f"APP_{col}", RAW_NUMERIC_DEFS[col], SRC)] = app[col].astype("float64")

    emp_raw = app["DAYS_EMPLOYED"].astype("float64")
    emp = clean_days(emp_raw)
    age = -app["DAYS_BIRTH"].astype("float64") / DAYS_PER_YEAR
    f[fd.add("APP_AGE_YEARS", "Age in years at application (-DAYS_BIRTH / 365.25)", SRC)] = age
    f[fd.add("APP_EMPLOYED_YEARS", "Years in current employment; 365243 sentinel set to missing",
             SRC)] = -emp / DAYS_PER_YEAR  # fmt: skip
    f[fd.add("APP_EMPLOYED_SENTINEL_FLAG", "1 if DAYS_EMPLOYED holds the 365243 sentinel "
             "(mostly pensioners / unemployed)", SRC)] = (emp_raw == DAYS_SENTINEL).astype(float)  # fmt: skip
    f[fd.add("APP_EMPLOYED_TO_AGE", "Employment years / age", SRC)] = ratio(
        -emp / DAYS_PER_YEAR, age
    )
    for col, name, label in (
        ("DAYS_REGISTRATION", "APP_REGISTRATION_YEARS", "Years since registration change"),
        ("DAYS_ID_PUBLISH", "APP_ID_PUBLISH_YEARS", "Years since ID document change"),
        ("DAYS_LAST_PHONE_CHANGE", "APP_PHONE_CHANGE_YEARS", "Years since phone change"),
    ):
        f[fd.add(name, f"{label} before application", SRC)] = (
            -app[col].astype("float64") / DAYS_PER_YEAR
        )

    inc, cred = app["AMT_INCOME_TOTAL"], app["AMT_CREDIT"]
    ann, goods = app["AMT_ANNUITY"], app["AMT_GOODS_PRICE"]
    f[fd.add("APP_CREDIT_TO_INCOME", "AMT_CREDIT / AMT_INCOME_TOTAL", SRC)] = ratio(cred, inc)
    f[fd.add("APP_ANNUITY_TO_INCOME", "AMT_ANNUITY / AMT_INCOME_TOTAL (debt-service burden)",
             SRC)] = ratio(ann, inc)  # fmt: skip
    f[fd.add("APP_CREDIT_TO_GOODS", "AMT_CREDIT / AMT_GOODS_PRICE (loan-to-value proxy)",
             SRC)] = ratio(cred, goods)  # fmt: skip
    f[fd.add("APP_CREDIT_TO_ANNUITY", "AMT_CREDIT / AMT_ANNUITY (term proxy, ignores interest)",
             SRC)] = ratio(cred, ann)  # fmt: skip
    f[fd.add("APP_INCOME_PER_PERSON", "AMT_INCOME_TOTAL / CNT_FAM_MEMBERS", SRC)] = ratio(
        inc, app["CNT_FAM_MEMBERS"]
    )
    ext = app[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].astype("float64")
    f[fd.add("APP_EXT_SOURCE_MEAN", "Mean of available EXT_SOURCE_1-3", SRC)] = ext.mean(axis=1)
    f[fd.add("APP_EXT_SOURCE_MIN", "Minimum of available EXT_SOURCE_1-3", SRC)] = ext.min(axis=1)
    f[fd.add("APP_EXT_SOURCE_MISSING_COUNT", "Number of EXT_SOURCE_1-3 missing", SRC)] = (
        ext.isna().sum(axis=1).astype(float)
    )

    for col, d in CATEGORICAL_DEFS.items():
        f[fd.add(f"APP_{col}", d, SRC, kind="categorical")] = app[col].astype("string")
    for col, d in PROTECTED_DEFS.items():
        f[fd.add(f"APP_{col}", d, SRC, kind="categorical", role="segment_only")] = app[col].astype(
            "string"
        )

    has = pd.Series(pd.Index(ids).isin(app.index), index=pd.Index(ids, name=ID))
    return LegResult(finalise(f, fd, ids), fd.frame(), has)
