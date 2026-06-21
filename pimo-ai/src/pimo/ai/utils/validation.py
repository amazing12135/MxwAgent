"""JSON Schema 工具参数验证与类型强制转换。

对 LLM 返回的工具调用参数进行 JSON Schema 验证。在验证前自动进行类型
强制转换（如 LLM 返回 str "42" 但 schema 要求 int，则转为 42），
减少因类型不匹配导致的误拒绝。
"""

from __future__ import annotations

import copy
import json
from typing import Any

from jsonschema import Draft202012Validator, validators
from jsonschema.exceptions import ValidationError

from pimo.ai.types import ToolCall

# =============================================================================
# 缓存已编译的 Schema Validator
# =============================================================================

# 使用 id 作为 key（schema dict 不可哈希，取 tuple of items 作为弱键）
_validator_cache: dict[int, Any] = {}


def _get_validator(schema: dict[str, Any]) -> Any:
    """获取（或编译）JSON Schema validator，带缓存。

    Args:
        schema: JSON Schema 对象。

    Returns:
        编译完成的 jsonschema Validator。
    """
    # 用 schema 的 frozen 表示作为缓存 key
    key = _schema_cache_key(schema)
    if key in _validator_cache:
        return _validator_cache[key]
    # 创建支持 draft2020-12 的 validator（兼容大多数 schema）
    try:
        cls = validators.validator_for(schema)
    except Exception:
        cls = Draft202012Validator
    v = cls(schema)
    _validator_cache[key] = v
    return v


def _schema_cache_key(schema: dict[str, Any]) -> int:
    """为 schema dict 生成稳定的 hash key。"""
    return hash(json.dumps(schema, sort_keys=True))


# =============================================================================
# 公共函数
# =============================================================================


def validate_tool_call(
    tools: list[dict[str, Any]],
    tool_call: ToolCall,
) -> dict[str, Any]:
    """按名称查找工具定义并验证工具调用参数。

    Args:
        tools: 工具定义列表（含 JSON Schema parameters）。
        tool_call: LLM 返回的工具调用。

    Returns:
        验证通过（且可能已强制转换）的参数字典。

    Raises:
        ValueError: 工具未找到或验证失败。
    """
    for tool in tools:
        if isinstance(tool, dict) and tool.get("name") == tool_call.name:
            return validate_tool_arguments(tool, tool_call)
    raise ValueError(f'Tool "{tool_call.name}" not found')


def validate_tool_arguments(
    tool: dict[str, Any],
    tool_call: ToolCall,
) -> dict[str, Any]:
    """对工具调用参数进行 JSON Schema 验证与类型强制转换。

    验证流程:
    1. 深拷贝参数（不修改原始 toolCall）
    2. 类型强制转换（递归处理 object/array/anyOf/oneOf/allOf）
    3. JSON Schema 编译 + 缓存
    4. 验证 → 不通过则格式化错误消息抛出 ValueError

    Args:
        tool: 工具定义 dict（含 "parameters" JSON Schema）。
        tool_call: LLM 返回的工具调用。

    Returns:
        验证通过且已强制转换的参数字典。

    Raises:
        ValueError: JSON Schema 验证失败。
    """
    args = copy.deepcopy(tool_call.arguments)
    schema = tool.get("parameters")
    if not isinstance(schema, dict):
        return args

    # Step 1: 类型强制转换
    coerced = _coerce_with_json_schema(args, schema)

    # Step 2: 合并强制转换结果到 args
    if coerced is not args:
        if isinstance(args, dict) and isinstance(coerced, dict):
            args.clear()
            args.update(coerced)
        else:
            return coerced if _check_valid(schema, coerced) else args

    # Step 3: 验证
    if _check_valid(schema, args):
        return args

    # Step 4: 收集错误并格式化
    validator = _get_validator(schema)
    errors = validator.iter_errors(args)
    error_lines = [
        f"  - {_format_error_path(e)}: {e.message}"
        for e in errors
    ]
    error_text = "\n".join(error_lines) or "Unknown validation error"

    raise ValueError(
        f'Validation failed for tool "{tool_call.name}":\n'
        f"{error_text}\n\n"
        f"Received arguments:\n"
        f"{json.dumps(tool_call.arguments, indent=2)}"
    )


def _check_valid(schema: dict[str, Any], value: Any) -> bool:
    """检查值是否通过 schema 验证。"""
    v = _get_validator(schema)
    return v.is_valid(value)


def _format_error_path(error: ValidationError) -> str:
    """格式化 jsonschema 错误路径为可读字符串。"""
    path = list(error.absolute_path)
    if error.validator == "required":
        # 提取缺失的 required 属性名
        missing = error.validator_value
        if missing:
            prefix = ".".join(str(p) for p in path)
            prop = missing[0] if isinstance(missing, list) else missing
            return f"{prefix}.{prop}" if prefix else str(prop)
    if not path:
        return "root"
    return ".".join(str(p) for p in path)


# =============================================================================
# 内部: 类型强制转换
# =============================================================================


def _coerce_primitive_by_type(value: Any, json_type: str) -> Any:
    """根据 JSON Schema 类型尝试强制转换原始值。

    Args:
        value: 原始值。
        json_type: JSON Schema 类型名。

    Returns:
        强制转换后的值（无法转换时返回原值）。
    """
    if json_type == "number":
        if value is None:
            return 0
        if isinstance(value, str) and value.strip():
            try:
                parsed = float(value)
                import math
                if math.isfinite(parsed):
                    return parsed
            except (ValueError, OverflowError):
                pass
        if isinstance(value, bool):
            return 1 if value else 0
        return value

    if json_type == "integer":
        if value is None:
            return 0
        if isinstance(value, str) and value.strip():
            try:
                parsed = int(value)
                return parsed
            except (ValueError, OverflowError):
                pass
        if isinstance(value, bool):
            return 1 if value else 0
        return value

    if json_type == "boolean":
        if value is None:
            return False
        if isinstance(value, str):
            if value == "true":
                return True
            if value == "false":
                return False
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value == 1:
                return True
            if value == 0:
                return False
        return value

    if json_type == "string":
        if value is None:
            return ""
        if isinstance(value, (int, float, bool)):
            return str(value)
        return value

    if json_type == "null":
        if value in ("", 0, False):
            return None
        return value

    return value


def _get_schema_types(schema: dict[str, Any]) -> list[str]:
    """从 schema 中提取 type 字段为字符串列表。

    Args:
        schema: JSON Schema 对象。

    Returns:
        类型名列表。
    """
    t = schema.get("type")
    if isinstance(t, str):
        return [t]
    if isinstance(t, list):
        return [x for x in t if isinstance(x, str)]
    return []


def _matches_json_type(value: Any, json_type: str) -> bool:
    """检查值的运行时类型是否匹配 JSON Schema 类型。

    Args:
        value: 待检查值。
        json_type: JSON Schema 类型名。

    Returns:
        True 表示类型匹配。
    """
    if json_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "boolean":
        return isinstance(value, bool)
    if json_type == "string":
        return isinstance(value, str)
    if json_type == "null":
        return value is None
    if json_type == "array":
        return isinstance(value, list)
    if json_type == "object":
        return isinstance(value, dict) and not isinstance(value, list)
    return False


def _coerce_with_json_schema(
    value: Any, schema: dict[str, Any]
) -> Any:
    """递归对值进行 JSON Schema 强制转换。

    处理 allOf/anyOf/oneOf/object（递归属性）/array（递归元素）。

    Args:
        value: 待转换的值。
        schema: JSON Schema 定义。

    Returns:
        强制转换后的值。
    """
    next_value = value

    # allOf: 连续应用子 schema
    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        for nested in all_of:
            if isinstance(nested, dict):
                next_value = _coerce_with_json_schema(
                    next_value, nested
                )

    # anyOf / oneOf: 尝试每个分支，取第一个验证通过的
    for key in ("anyOf", "oneOf"):
        subs = schema.get(key)
        if isinstance(subs, list):
            next_value = _coerce_union(next_value, subs)

    # 原始类型强制
    schema_types = _get_schema_types(schema)
    matches_union = (
        len(schema_types) > 1
        and any(
            _matches_json_type(next_value, st)
            for st in schema_types
        )
    )
    if schema_types and not matches_union:
        for st in schema_types:
            candidate = _coerce_primitive_by_type(next_value, st)
            if candidate is not next_value:
                next_value = candidate
                break

    # object: 递归处理属性
    if "object" in schema_types and isinstance(next_value, dict):
        _apply_schema_object_coercion(next_value, schema)

    # array: 递归处理元素
    if "array" in schema_types and isinstance(next_value, list):
        _apply_schema_array_coercion(next_value, schema)

    return next_value


def _coerce_union(
    value: Any, schemas: list[dict[str, Any]]
) -> Any:
    """对 anyOf/oneOf 联合类型尝试强制转换。

    依次尝试每个分支的 schema，取第一个验证通过的强制转换结果。

    Args:
        value: 原始值。
        schemas: 子 schema 列表。

    Returns:
        第一个验证通过分支的强制转换结果。全失败时返回原值。
    """
    for s in schemas:
        if not isinstance(s, dict):
            continue
        candidate = _coerce_with_json_schema(
            copy.deepcopy(value), s
        )
        if _check_valid(s, candidate):
            return candidate
    return value


def _apply_schema_object_coercion(
    value: dict[str, Any], schema: dict[str, Any]
) -> None:
    """原地对 object 的属性进行 JSON Schema 强制转换。

    Args:
        value: 待转换的 dict（原地修改）。
        schema: JSON Schema 定义。
    """
    properties = schema.get("properties")
    if isinstance(properties, dict):
        defined_keys = set(properties.keys())
        for key, prop_schema in properties.items():
            if key in value and isinstance(prop_schema, dict):
                value[key] = _coerce_with_json_schema(
                    value[key], prop_schema
                )

        # additionalProperties: 处理未定义的属性
        additional = schema.get("additionalProperties")
        if isinstance(additional, dict):
            for key, prop_value in list(value.items()):
                if key not in defined_keys:
                    value[key] = _coerce_with_json_schema(
                        prop_value, additional
                    )


def _apply_schema_array_coercion(
    value: list, schema: dict[str, Any]
) -> None:
    """原地对 array 的元素进行 JSON Schema 强制转换。

    Args:
        value: 待转换的 list（原地修改）。
        schema: JSON Schema 定义。
    """
    items = schema.get("items")
    if isinstance(items, list):
        # 元组形式: 每个位置对应不同 schema
        for i, item_schema in enumerate(items):
            if i < len(value) and isinstance(item_schema, dict):
                value[i] = _coerce_with_json_schema(
                    value[i], item_schema
                )
    elif isinstance(items, dict):
        # 单一 schema 适用于所有元素
        for i in range(len(value)):
            value[i] = _coerce_with_json_schema(value[i], items)
