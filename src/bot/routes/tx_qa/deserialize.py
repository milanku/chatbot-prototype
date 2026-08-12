import json
from enum import Enum
from typing import TypeVar, cast

from bot.logging import log_event
from bot.models.tx_qa.domain import (
    Direction,
    Label,
)
from bot.models.tx_qa.query import (
    RawRangeEndpoint,
    RollingRangeMode,
    RollingRangeUnit,
    TimeframeType,
    TXQAParseRawQueryData,
)

EnumT = TypeVar("EnumT", bound=Enum)

def _optional_int(
    data: dict[str, object],
    key: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None
) -> int | None:
    value = data.get(key)
    
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):  # bool is a subclass of int, so exclude it
        raise ValueError(f"Expected an integer for key '{key}', got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ValueError(f"Value for key '{key}' is below the minimum of {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"Value for key '{key}' is above the maximum of {maximum}, got {value}")
    return value

def _optional_str(
    data: dict[str, object],
    key: str,
    *,
    allowed_values: set[str] | None = None
) -> str | None:
    value = data.get(key)
    
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Expected a string for key '{key}', got {type(value).__name__}")
    if allowed_values is not None and value not in allowed_values:
        raise ValueError(f"Value for key '{key}' is not in the allowed values {allowed_values}, got '{value}'")
    return value

def _optional_float(
    data: dict[str, object],
    key: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None
 ) -> float | None:
    value = data.get(key)
    
    if value is None:
        return None
    if not isinstance(value, (float, int)) or isinstance(value, bool):  # bool is a subclass of int, so exclude it
        raise ValueError(f"Expected a float for key '{key}', got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ValueError(f"Value for key '{key}' is below the minimum of {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"Value for key '{key}' is above the maximum of {maximum}, got {value}")
    return float(value)

def _optional_str_enum(
    data: dict[str, object],
    key: str,
    enum_type: type[EnumT]
) -> EnumT | None:
    value = data.get(key)
    
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"{key!r} must be a string or null, "
            f"got {type(value).__name__}"
        )
    try:
        return enum_type(value)
    except (ValueError, TypeError) as exc:
        allowed_values = ", ".join(
            repr(member.value) for member in enum_type
        )
        raise ValueError(
            f"{key!r} must be one of {allowed_values}, got {value!r}"
        ) from exc

def _required_str_enum(
    data: dict[str, object],
    key: str,
    enum_type: type[EnumT]
) -> EnumT:
    value = data.get(key)
    
    if value is None:
        raise ValueError(f"Missing required key '{key}'")
    if not isinstance(value, str):
        raise ValueError(
            f"{key!r} must be a string or null, "
            f"got {type(value).__name__}"
        )
    try:
        return enum_type(value)
    except (ValueError, TypeError) as exc:
        allowed_values = ", ".join(
            repr(member.value) for member in enum_type
        )
        raise ValueError(
            f"{key!r} must be one of {allowed_values}, got {value!r}"
        ) from exc


def deserialize_raw_query_data(
    serialized_raw_query_data: str,
) -> TXQAParseRawQueryData:
    try:
        raw_data = json.loads(serialized_raw_query_data)
    except json.JSONDecodeError as exc:
        log_event(
            event="tx_qa.deserialize_raw_query_data.invalid_response",
            payload={"serialized_raw_query_data": serialized_raw_query_data}
        )
        raise ValueError("The LLM response must be a valid JSON string") from exc

    data = cast(dict[str, object], raw_data)

    return TXQAParseRawQueryData(
        label=_optional_str_enum(data, "label", Label),
        direction=_optional_str_enum(data, "direction", Direction),
        timeframe_type=_required_str_enum(
            data,
            "timeframe_type",
            TimeframeType,
        ),
        relative_offset=_optional_int(data, "relative_offset"),
        year=_optional_int(data, "year"),
        quarter=_optional_int(
            data,
            "quarter",
            minimum=1,
            maximum=4,
        ),
        month=_optional_int(
            data,
            "month",
            minimum=1,
            maximum=12,
        ),
        day=_optional_int(
            data,
            "day",
            minimum=1,
            maximum=31,
        ),
        start_endpoint=RawRangeEndpoint(
            year=_optional_int(data, "start_year"),
            quarter=_optional_int(
                data,
                "start_quarter",
                minimum=1,
                maximum=4,
            ),
            month=_optional_int(
                data,
                "start_month",
                minimum=1,
                maximum=12,
            ),
            day=_optional_int(
                data,
                "start_day",
                minimum=1,
                maximum=31,
            ),
        ),
        end_endpoint=RawRangeEndpoint(
            year=_optional_int(data, "end_year"),
            quarter=_optional_int(
                data,
                "end_quarter",
                minimum=1,
                maximum=4,
            ),
            month=_optional_int(
                data,
                "end_month",
                minimum=1,
                maximum=12,
            ),
            day=_optional_int(
                data,
                "end_day",
                minimum=1,
                maximum=31,
            ),
        ),
        unit=_optional_str_enum(
            data,
            "unit",
            RollingRangeUnit,
        ),
        mode=_optional_str_enum(
            data,
            "mode",
            RollingRangeMode,
        ),
        amount=_optional_int(
            data,
            "amount",
            minimum=1,
        ),
        confidence=_optional_float(
            data,
            "confidence",
            minimum=0.0,
            maximum=1.0,
        ),
        reason=_optional_str(data, "reason"),
    )