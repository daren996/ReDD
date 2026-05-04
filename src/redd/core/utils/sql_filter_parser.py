"""
SQL Filter Parser

Parses SQL WHERE clauses into individual attribute filters (predicates)
for use in the proxy runtime.

Supported SQL operators:
- Comparison: =, !=, <>, <, >, <=, >=
- String: LIKE, NOT LIKE, IN, NOT IN
- Null: IS NULL, IS NOT NULL
- Logical: AND, OR (splits into separate filters)

Example:
    ```python
    from core.utils.sql_filter_parser import SQLFilterParser
    
    parser = SQLFilterParser()
    sql = "SELECT * FROM instructor WHERE dept_name = 'Comp. Sci.' AND salary > 50000"
    filters = parser.parse(sql)
    # Returns: [
    #   AttributePredicate(attribute='dept_name', operator='=', value='Comp. Sci.'),
    #   AttributePredicate(attribute='salary', operator='>', value=50000)
    # ]
    ```
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = [
    "SQLFilterParser",
    "AttributePredicate",
    "PredicateOperator",
    "create_predicate_function",
    "predicates_to_filter_dict",
    "parse_alias_mapping",
    "group_predicates_by_table",
    "JoinCondition",
    "JoinGraph",
    "parse_join_conditions",
    "has_join",
    "get_join_graph",
    "compute_table_processing_order",
]


class PredicateOperator(Enum):
    """SQL comparison operators."""
    EQ = "="
    NEQ = "!="
    NEQ_ALT = "<>"
    LT = "<"
    GT = ">"
    LTE = "<="
    GTE = ">="
    LIKE = "LIKE"
    NOT_LIKE = "NOT LIKE"
    IN = "IN"
    NOT_IN = "NOT IN"
    IS_NULL = "IS NULL"
    IS_NOT_NULL = "IS NOT NULL"
    BETWEEN = "BETWEEN"
    CONTAINS = "CONTAINS"  # Custom for string containment


@dataclass
class AttributePredicate:
    """
    Represents a single attribute predicate from a SQL WHERE clause.
    
    Attributes:
        attribute: Column/attribute name (e.g., 'salary', 'dept_name')
        operator: Comparison operator (e.g., '=', '>', 'LIKE')
        value: Value to compare against (can be str, int, float, list, None)
        table_alias: Optional table alias (e.g., 'T1' in 'T1.salary')
        original_sql: Original SQL fragment for this predicate
    """
    attribute: str
    operator: str
    value: Any
    table_alias: Optional[str] = None
    original_sql: str = ""
    
    # Additional metadata
    is_numeric: bool = False
    is_string: bool = False
    
    def __post_init__(self):
        """Infer types from value."""
        if isinstance(self.value, (int, float)) and not isinstance(self.value, bool):
            self.is_numeric = True
        elif isinstance(self.value, str):
            self.is_string = True
    
    @property
    def full_attribute_name(self) -> str:
        """Get full attribute name including table alias."""
        if self.table_alias:
            return f"{self.table_alias}.{self.attribute}"
        return self.attribute
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "attribute": self.attribute,
            "operator": self.operator,
            "value": self.value,
            "table_alias": self.table_alias,
            "original_sql": self.original_sql,
            "is_numeric": self.is_numeric,
            "is_string": self.is_string,
        }
    
    def __str__(self) -> str:
        return f"{self.full_attribute_name} {self.operator} {repr(self.value)}"


class SQLFilterParser:
    """
    Parses SQL WHERE clauses into individual attribute predicates.
    
    Handles:
    - Simple comparisons: col = value, col > value
    - String patterns: col LIKE '%pattern%'
    - List membership: col IN ('a', 'b', 'c')
    - NULL checks: col IS NULL, col IS NOT NULL
    - Compound conditions: cond1 AND cond2 AND cond3
    
    Note: OR conditions are currently not split (treated as single complex predicate).
    """
    
    # Regex patterns for parsing
    PATTERNS = {
        # Match table.column or just column
        "column": r"([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)",
        
        # Match string literals (single or double quoted)
        "string_literal": r"'([^']*)'|\"([^\"]*)\"",
        
        # Match numeric literals
        "numeric_literal": r"(-?\d+(?:\.\d+)?)",
        
        # Match IN clause values
        "in_values": r"\(([^)]+)\)",
    }
    
    # Operators in order of precedence (longer patterns first)
    OPERATORS = [
        ("IS NOT NULL", PredicateOperator.IS_NOT_NULL),
        ("IS NULL", PredicateOperator.IS_NULL),
        ("NOT LIKE", PredicateOperator.NOT_LIKE),
        ("NOT IN", PredicateOperator.NOT_IN),
        ("LIKE", PredicateOperator.LIKE),
        ("BETWEEN", PredicateOperator.BETWEEN),
        ("IN", PredicateOperator.IN),
        ("<=", PredicateOperator.LTE),
        (">=", PredicateOperator.GTE),
        ("<>", PredicateOperator.NEQ_ALT),
        ("!=", PredicateOperator.NEQ),
        ("=", PredicateOperator.EQ),
        ("<", PredicateOperator.LT),
        (">", PredicateOperator.GT),
    ]
    
    def __init__(self, strip_table_aliases: bool = True):
        """
        Initialize SQL filter parser.
        
        Args:
            strip_table_aliases: If True, remove table aliases from attribute names
        """
        self.strip_table_aliases = strip_table_aliases
    
    def parse(self, sql: str) -> List[AttributePredicate]:
        """
        Parse SQL statement and extract WHERE clause predicates.
        
        Args:
            sql: Full SQL statement or just WHERE clause
            
        Returns:
            List of AttributePredicate objects
        """
        # Extract WHERE clause
        where_clause = self._extract_where_clause(sql)
        
        if not where_clause:
            logging.debug(f"[SQLFilterParser] No WHERE clause found in: {sql[:100]}...")
            return []
        
        # Split by AND (preserve OR as single predicates for now)
        conditions = self._split_conditions(where_clause)
        
        # Parse each condition
        predicates = []
        for condition in conditions:
            pred = self._parse_condition(condition)
            if pred:
                predicates.append(pred)
        
        logging.info(f"[SQLFilterParser] Parsed {len(predicates)} predicates from SQL")
        return predicates
    
    def _extract_where_clause(self, sql: str) -> Optional[str]:
        """
        Extract the WHERE clause from a SQL statement.
        
        Args:
            sql: Full SQL statement
            
        Returns:
            WHERE clause content (without 'WHERE' keyword), or None
        """
        sql = sql.strip()
        
        # Case-insensitive search for WHERE
        where_pattern = re.compile(
            r'\bWHERE\s+(.+?)(?:\s+(?:GROUP|ORDER|HAVING|LIMIT|UNION|EXCEPT|INTERSECT)\s+|;|$)',
            re.IGNORECASE | re.DOTALL
        )
        
        match = where_pattern.search(sql)
        if match:
            return match.group(1).strip()
        
        # Try simpler pattern if the above fails
        simple_pattern = re.compile(r'\bWHERE\s+(.+)', re.IGNORECASE | re.DOTALL)
        match = simple_pattern.search(sql)
        if match:
            clause = match.group(1).strip()
            # Remove trailing clauses
            for keyword in ['GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT', ';']:
                idx = clause.upper().find(keyword)
                if idx > 0:
                    clause = clause[:idx].strip()
            return clause
        
        return None
    
    def _split_conditions(self, where_clause: str) -> List[str]:
        """
        Split WHERE clause by AND operator.
        
        Handles parentheses properly (doesn't split inside parentheses).
        
        Args:
            where_clause: WHERE clause content
            
        Returns:
            List of condition strings
        """
        # Replace AND with a marker (case insensitive), but not inside parentheses
        result = []
        current = ""
        paren_depth = 0
        i = 0
        
        while i < len(where_clause):
            char = where_clause[i]
            
            if char == '(':
                paren_depth += 1
                current += char
            elif char == ')':
                paren_depth -= 1
                current += char
            elif paren_depth == 0:
                # Check for AND keyword
                remaining = where_clause[i:].upper()
                if remaining.startswith(' AND ') or remaining.startswith('\nAND '):
                    if current.strip():
                        result.append(current.strip())
                    current = ""
                    i += 5  # Skip ' AND '
                    continue
                elif remaining.startswith('AND '):
                    if current.strip():
                        result.append(current.strip())
                    current = ""
                    i += 4  # Skip 'AND '
                    continue
                else:
                    current += char
            else:
                current += char
            
            i += 1
        
        if current.strip():
            result.append(current.strip())
        
        return result
    
    def _parse_condition(self, condition: str) -> Optional[AttributePredicate]:
        """
        Parse a single condition into an AttributePredicate.
        
        Args:
            condition: Single condition string (e.g., "salary > 50000")
            
        Returns:
            AttributePredicate or None if parsing fails
        """
        condition = self._strip_outer_parentheses(condition.strip())
        
        # Try each operator pattern
        for op_str, op_enum in self.OPERATORS:
            # Case-insensitive pattern for the operator
            if op_str in ['IS NULL', 'IS NOT NULL']:
                # These are suffix operators
                pattern = rf"(.+?)\s+{re.escape(op_str)}\s*$"
            elif op_str == 'BETWEEN':
                # BETWEEN needs special handling
                pattern = r"(.+?)\s+BETWEEN\s+(.+?)\s+AND\s+(.+)"
            elif op_str in ['IN', 'NOT IN']:
                pattern = rf"(.+?)\s+{re.escape(op_str)}\s*\(([^)]+)\)"
            elif op_str in ['LIKE', 'NOT LIKE']:
                pattern = rf"(.+?)\s+{re.escape(op_str)}\s+(.+)"
            else:
                # Standard comparison operators
                pattern = rf"(.+?)\s*{re.escape(op_str)}\s*(.+)"
            
            match = re.match(pattern, condition, re.IGNORECASE)
            if match:
                return self._create_predicate(match, op_str, op_enum, condition)
        
        logging.warning(f"[SQLFilterParser] Could not parse condition: {condition}")
        return None
    
    def _create_predicate(
        self, 
        match: re.Match, 
        op_str: str, 
        op_enum: PredicateOperator,
        original: str
    ) -> Optional[AttributePredicate]:
        """
        Create an AttributePredicate from a regex match.
        
        Args:
            match: Regex match object
            op_str: Operator string
            op_enum: Operator enum
            original: Original condition string
            
        Returns:
            AttributePredicate or None
        """
        groups = match.groups()
        
        # Extract column name
        column_part = groups[0].strip()
        table_alias, attribute = self._parse_column_name(column_part)
        
        # Extract value based on operator type
        if op_enum in [PredicateOperator.IS_NULL, PredicateOperator.IS_NOT_NULL]:
            value = None
        elif op_enum == PredicateOperator.BETWEEN:
            # BETWEEN has two values
            val1 = self._parse_value(groups[1].strip())
            val2 = self._parse_value(groups[2].strip())
            value = (val1, val2)
        elif op_enum in [PredicateOperator.IN, PredicateOperator.NOT_IN]:
            # Parse list of values
            values_str = groups[1].strip()
            value = self._parse_in_values(values_str)
        else:
            # Standard comparison
            value = self._parse_value(groups[1].strip())
        
        return AttributePredicate(
            attribute=attribute,
            operator=op_str,
            value=value,
            table_alias=table_alias if not self.strip_table_aliases else None,
            original_sql=original
        )
    
    def _parse_column_name(self, column_str: str) -> Tuple[Optional[str], str]:
        """
        Parse column name, separating table alias if present.
        
        Args:
            column_str: Column string (e.g., 'T1.salary' or 'salary')
            
        Returns:
            (table_alias, column_name) tuple
        """
        column_str = self._normalize_column_expr(column_str.strip())
        
        if '.' in column_str:
            parts = column_str.split('.', 1)
            return self._strip_identifier_quotes(parts[0].strip()), self._strip_identifier_quotes(parts[1].strip())
        
        return None, self._strip_identifier_quotes(column_str)

    def _strip_outer_parentheses(self, value: str) -> str:
        """Remove balanced parentheses wrapping a whole condition/expression."""
        text = value.strip()
        while text.startswith("(") and text.endswith(")") and self._has_wrapping_parentheses(text):
            text = text[1:-1].strip()
        return text

    @staticmethod
    def _has_wrapping_parentheses(value: str) -> bool:
        depth = 0
        for index, char in enumerate(value):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0 and index != len(value) - 1:
                    return False
            if depth < 0:
                return False
        return depth == 0

    def _normalize_column_expr(self, column_str: str) -> str:
        """Reduce simple SQL expression wrappers to the underlying column name."""
        text = self._strip_outer_parentheses(column_str.strip())
        cast_match = re.match(r"(?is)^CAST\s*\((.+?)\s+AS\s+[^)]+\)$", text)
        if cast_match:
            return self._normalize_column_expr(cast_match.group(1))
        func_match = re.match(r"(?is)^(?:LOWER|UPPER|TRIM)\s*\((.+)\)$", text)
        if func_match:
            return self._normalize_column_expr(func_match.group(1))
        return self._strip_identifier_quotes(text)

    @staticmethod
    def _strip_identifier_quotes(value: str) -> str:
        text = value.strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'", "`"}:
            return text[1:-1]
        return text
    
    def _parse_value(self, value_str: str) -> Any:
        """
        Parse a value string into appropriate Python type.
        
        Args:
            value_str: Value string (e.g., "'Fall'", "50000", "NULL")
            
        Returns:
            Parsed value (str, int, float, or None)
        """
        value_str = value_str.strip()
        
        # Check for NULL
        if value_str.upper() == 'NULL':
            return None
        
        # Check for string literal
        if (value_str.startswith("'") and value_str.endswith("'")) or \
           (value_str.startswith('"') and value_str.endswith('"')):
            return value_str[1:-1]
        
        # Try numeric parsing
        try:
            if '.' in value_str:
                return float(value_str)
            return int(value_str)
        except ValueError:
            pass
        
        # Return as-is (might be a column reference or variable)
        return value_str
    
    def _parse_in_values(self, values_str: str) -> List[Any]:
        """
        Parse IN clause values.
        
        Args:
            values_str: Values string (e.g., "'a', 'b', 'c'" or "1, 2, 3")
            
        Returns:
            List of parsed values
        """
        # Split by comma, but not inside quotes
        values = []
        current = ""
        in_quotes = False
        quote_char = None
        
        for char in values_str:
            if char in ("'", '"') and not in_quotes:
                in_quotes = True
                quote_char = char
                current += char
            elif char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
                current += char
            elif char == ',' and not in_quotes:
                if current.strip():
                    values.append(self._parse_value(current.strip()))
                current = ""
            else:
                current += char
        
        if current.strip():
            values.append(self._parse_value(current.strip()))
        
        return values


def create_predicate_function(
    predicate: AttributePredicate,
    case_insensitive: bool = True
) -> Callable[[Any], bool]:
    """
    Create a Python predicate function from an AttributePredicate.
    
    Args:
        predicate: AttributePredicate object
        case_insensitive: If True, string comparisons are case-insensitive
        
    Returns:
        Function that takes a value and returns True/False
    """
    op = predicate.operator.upper()
    target_value = predicate.value
    
    def normalize_string(s: Any) -> str:
        """Convert to lowercase string for comparison."""
        if s is None:
            return ""
        s = str(s)
        return s.lower() if case_insensitive else s
    
    def safe_compare(v: Any) -> bool:
        """Safe comparison handling type coercion."""
        if v is None:
            return False
        
        try:
            if op == '=':
                if predicate.is_numeric:
                    return float(v) == float(target_value)
                return normalize_string(v) == normalize_string(target_value)
            
            elif op in ('!=', '<>'):
                if predicate.is_numeric:
                    return float(v) != float(target_value)
                return normalize_string(v) != normalize_string(target_value)
            
            elif op == '<':
                return float(v) < float(target_value)
            
            elif op == '>':
                return float(v) > float(target_value)
            
            elif op == '<=':
                return float(v) <= float(target_value)
            
            elif op == '>=':
                return float(v) >= float(target_value)
            
            elif op == 'LIKE':
                # Convert SQL LIKE pattern to regex
                pattern = normalize_string(target_value)
                pattern = pattern.replace('%', '.*').replace('_', '.')
                pattern = f'^{pattern}$'
                return bool(re.match(pattern, normalize_string(v)))
            
            elif op == 'NOT LIKE':
                pattern = normalize_string(target_value)
                pattern = pattern.replace('%', '.*').replace('_', '.')
                pattern = f'^{pattern}$'
                return not bool(re.match(pattern, normalize_string(v)))
            
            elif op == 'IN':
                v_norm = normalize_string(v) if not predicate.is_numeric else float(v)
                targets = [normalize_string(t) if not predicate.is_numeric else float(t) 
                          for t in target_value]
                return v_norm in targets
            
            elif op == 'NOT IN':
                v_norm = normalize_string(v) if not predicate.is_numeric else float(v)
                targets = [normalize_string(t) if not predicate.is_numeric else float(t) 
                          for t in target_value]
                return v_norm not in targets
            
            elif op == 'IS NULL':
                return v is None or str(v).strip() == ''
            
            elif op == 'IS NOT NULL':
                return v is not None and str(v).strip() != ''
            
            elif op == 'BETWEEN':
                low, high = float(target_value[0]), float(target_value[1])
                return low <= float(v) <= high
            
            elif op == 'CONTAINS':
                return normalize_string(target_value) in normalize_string(v)
            
            else:
                logging.warning(f"[create_predicate_function] Unknown operator: {op}")
                return False
                
        except (ValueError, TypeError) as e:
            logging.debug(f"[create_predicate_function] Comparison failed: {e}")
            return False
    
    # Set function name for debugging
    safe_compare.__name__ = f"pred_{predicate.attribute}_{op}"
    safe_compare.__doc__ = str(predicate)
    
    return safe_compare


def predicates_to_filter_dict(
    predicates: List[AttributePredicate],
    case_insensitive: bool = True
) -> Dict[str, Callable[[Any], bool]]:
    """
    Convert list of predicates to a dictionary of filter functions.
    
    Args:
        predicates: List of AttributePredicate objects
        case_insensitive: If True, string comparisons are case-insensitive
        
    Returns:
        Dictionary mapping attribute names to predicate functions
    """
    result = {}
    for pred in predicates:
        result[pred.attribute] = create_predicate_function(pred, case_insensitive)
    return result


def parse_alias_mapping(sql: str) -> Dict[str, str]:
    """
    Parse SQL FROM and JOIN clauses to build alias -> table name mapping.
    
    Handles patterns:
    - table AS alias
    - table alias
    - alias is the table name itself (when no alias given)
    
    Args:
        sql: Full SQL statement
        
    Returns:
        Dict mapping alias/identifier to canonical table name.
        E.g., {"T1": "instructor", "T2": "teaches", "instructor": "instructor"}
    """
    sql = sql.strip()
    result: Dict[str, str] = {}
    
    # Extract FROM clause and JOIN clauses (before WHERE)
    # Match: FROM ... (JOIN ...)* until WHERE, GROUP, ORDER, etc. or end of string
    from_match = re.search(
        r'\bFROM\s+(.+?)(?=\bWHERE\b|\bGROUP\b|\bORDER\b|\bHAVING\b|\bLIMIT\b|\bUNION\b|\bINTERSECT\b|\bEXCEPT\b|;|$)',
        sql,
        re.IGNORECASE | re.DOTALL
    )
    if not from_match:
        return result
    
    from_join_clause = from_match.group(1).strip()
    
    # Split by JOIN (case insensitive) - first part is FROM, rest are JOINs
    parts = re.split(r'\bJOIN\b', from_join_clause, flags=re.IGNORECASE)
    
    for part in parts:
        part = part.strip()
        # Remove ON ... and USING (...) if present
        on_idx = re.search(r'\bON\b', part, re.IGNORECASE)
        if on_idx:
            part = part[:on_idx.start()].strip()
        using_match = re.search(r'\bUSING\s*\(', part, re.IGNORECASE)
        if using_match:
            part = part[:using_match.start()].strip()
        
        # Parse "table AS alias" or "table alias" or just "table"
        # Pattern: optional schema.table, optional AS alias
        # table_name (AS)? alias?
        match = re.match(
            r'([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*$',
            part.strip(),
            re.IGNORECASE
        )
        if match:
            table_ref, alias = match.groups()
            table_name = table_ref.split('.')[-1]  # Use last part if schema.table
            result[alias] = table_name
            result[table_name] = table_name  # Table name is also its own ref
        else:
            # Single identifier: table name
            single = re.match(r'([A-Za-z_][A-Za-z0-9_]*)', part)
            if single:
                table_name = single.group(1)
                result[table_name] = table_name
    
    return result


def group_predicates_by_table(
    sql: str,
    schema: List[Dict[str, Any]],
    query_tables: Optional[List[str]] = None,
) -> Dict[str, List[AttributePredicate]]:
    """
    Parse SQL predicates and group them by table name.
    
    Uses alias mapping from FROM/JOIN for predicates with table_alias.
    Uses schema to resolve predicates without alias (attribute in single table).
    
    Args:
        sql: Full SQL statement with WHERE clause
        schema: List of schema dicts with "Schema Name" and "Attributes"
        query_tables: Optional list of table names used in the query (from query_info)
        
    Returns:
        Dict mapping table name to list of AttributePredicate for that table
    """
    # Build attribute -> tables mapping from schema
    attr_to_tables: Dict[str, List[str]] = {}
    for s in schema:
        table_name = s.get("Schema Name") or s.get("table_name", "")
        if not table_name:
            continue
        for attr_info in s.get("Attributes", []):
            attr_name = attr_info.get("Attribute Name") or attr_info.get("attribute_name", "")
            if isinstance(attr_info, str):
                attr_name = attr_info
            if attr_name:
                attr_to_tables.setdefault(attr_name, []).append(table_name)
    
    # Parse alias mapping
    alias_to_table = parse_alias_mapping(sql)
    
    # Parse predicates with aliases preserved
    parser = SQLFilterParser(strip_table_aliases=False)
    predicates = parser.parse(sql)
    
    result: Dict[str, List[AttributePredicate]] = {}
    
    for pred in predicates:
        table_name = None
        
        if pred.table_alias:
            table_name = alias_to_table.get(pred.table_alias)
        
        if table_name is None:
            # No alias: use schema to find table
            tables_with_attr = attr_to_tables.get(pred.attribute, [])
            if len(tables_with_attr) == 1:
                table_name = tables_with_attr[0]
            elif query_tables and len(tables_with_attr) > 1:
                # Attribute in multiple tables: pick first that's in query
                for t in query_tables:
                    if t in tables_with_attr:
                        table_name = t
                        break
            elif tables_with_attr:
                table_name = tables_with_attr[0]
        
        if table_name is None and query_tables and len(query_tables) == 1:
            table_name = query_tables[0]
        
        if table_name is None:
            logging.warning(
                f"[group_predicates_by_table] Could not resolve table for predicate "
                f"{pred.attribute} {pred.operator} {pred.value}; skipping"
            )
            continue
        
        result.setdefault(table_name, []).append(pred)
    
    return result


# =============================================================================
# Join Parsing
# =============================================================================


@dataclass
class JoinCondition:
    """
    Represents a join condition from SQL ON clause: table_a.attr_a = table_b.attr_b.
    
    For "a JOIN b ON a.c = b.c", parent (table_a) is processed first;
    child (table_b) depends on parent's join-key values.
    
    Attributes:
        table_parent: Table that provides join key (processed first)
        attr_parent: Join attribute in parent table
        table_child: Table that references parent (processed second)
        attr_child: Join attribute in child table
    """
    table_parent: str
    attr_parent: str
    table_child: str
    attr_child: str
    
    def __str__(self) -> str:
        return f"{self.table_parent}.{self.attr_parent} = {self.table_child}.{self.attr_child}"


@dataclass
class JoinGraph:
    """
    Join graph: table dependencies and join keys.
    
    Attributes:
        conditions: List of JoinCondition (parent -> child)
        parent_to_children: Dict[table, List[table]] - tables that depend on this one
        child_to_parent: Dict[table, List[tuple]] - (parent_table, attr_parent, attr_child)
    """
    conditions: List[JoinCondition]
    parent_to_children: Dict[str, List[str]]
    child_to_parent: Dict[str, List[Tuple[str, str, str]]]


def has_join(sql: str) -> bool:
    """
    Quick check if SQL contains a JOIN.
    
    Args:
        sql: Full SQL statement
        
    Returns:
        True if JOIN keyword appears (excluding string literals)
    """
    sql = sql.strip()
    # Simple heuristic: JOIN as a word boundary (avoid matching inside strings)
    pattern = re.compile(r'\bJOIN\b', re.IGNORECASE)
    return bool(pattern.search(sql))


def _extract_on_clauses(sql: str) -> List[str]:
    """
    Extract ON clause conditions from FROM/JOIN section.
    Returns list of condition strings (e.g., ["T1.name = T2.instructor_name"]).
    """
    sql = sql.strip()
    on_conditions: List[str] = []
    
    # Match FROM ... (JOIN ... ON ...)* until WHERE, etc. or end of string
    from_match = re.search(
        r'\bFROM\s+(.+?)(?=\bWHERE\b|\bGROUP\b|\bORDER\b|\bHAVING\b|\bLIMIT\b|\bUNION\b|\bINTERSECT\b|\bEXCEPT\b|;|$)',
        sql,
        re.IGNORECASE | re.DOTALL
    )
    if not from_match:
        return on_conditions
    
    from_join_clause = from_match.group(1).strip()
    
    # Find all ON clauses: ON <condition> (until next JOIN or end)
    on_pattern = re.compile(r'\bON\s+(.+?)(?=\b(?:JOIN|LEFT|RIGHT|INNER|OUTER|WHERE|GROUP|ORDER|HAVING|LIMIT)\b|$)', 
                            re.IGNORECASE | re.DOTALL)
    
    for match in on_pattern.finditer(from_join_clause):
        cond = match.group(1).strip()
        # Split by AND for multiple conditions
        and_parts = re.split(r'\s+AND\s+', cond, flags=re.IGNORECASE)
        for part in and_parts:
            part = part.strip().strip('()')
            if part:
                on_conditions.append(part)
    
    return on_conditions


def _parse_column_eq_column(condition: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Parse "alias_a.attr_a = alias_b.attr_b" into (alias_a, attr_a, alias_b, attr_b).
    Returns None if not a column=column equality.
    """
    condition = condition.strip()
    # Match: x.y = z.w (column = column)
    match = re.match(
        r'([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)',
        condition,
        re.IGNORECASE
    )
    if match:
        return match.groups()  # (alias_left, attr_left, alias_right, attr_right)
    return None


def parse_join_conditions(sql: str) -> List[JoinCondition]:
    """
    Parse SQL and extract join conditions from ON clauses.
    
    Handles: a JOIN b ON a.col = b.col, a JOIN b ON T1.x = T2.y (with aliases).
    Resolves aliases to table names via parse_alias_mapping.
    
    Args:
        sql: Full SQL statement
        
    Returns:
        List of JoinCondition (table_parent, attr_parent, table_child, attr_child).
        Parent is the table that appears first in the JOIN (FROM side).
    """
    alias_to_table = parse_alias_mapping(sql)
    on_conditions = _extract_on_clauses(sql)
    join_conditions: List[JoinCondition] = []
    
    # Get JOIN order to determine parent vs child
    from_match = re.search(
        r'\bFROM\s+(.+?)(?=\bWHERE\b|\bGROUP\b|\bORDER\b|\bHAVING\b|\bLIMIT\b|\bUNION\b|\bINTERSECT\b|\bEXCEPT\b|;|$)',
        sql,
        re.IGNORECASE | re.DOTALL
    )
    if not from_match:
        return join_conditions
    
    from_join_clause = from_match.group(1).strip()
    parts = re.split(r'\bJOIN\b', from_join_clause, flags=re.IGNORECASE)
    
    # First part is FROM table; each subsequent part is a JOIN
    # For "FROM a JOIN b ON a.c = b.c": left side of = is from earlier table (parent)
    # We need to know which alias belongs to which "side" of the JOIN.
    # Simpler: for each ON condition "x.y = z.w", the first table in FROM order
    # that contains x or z is parent. Actually: in "FROM a JOIN b ON a.c = b.c",
    # a is parent. So we need to know table order. First part = first table.
    
    # Build ordered list of (alias, table) from JOIN order
    join_order: List[Tuple[str, str]] = []
    for part in parts:
        part = part.strip()
        on_idx = re.search(r'\bON\b', part, re.IGNORECASE)
        if on_idx:
            part = part[:on_idx.start()].strip()
        using_match = re.search(r'\bUSING\s*\(', part, re.IGNORECASE)
        if using_match:
            part = part[:using_match.start()].strip()
        
        match = re.match(
            r'([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*$',
            part,
            re.IGNORECASE
        )
        if match:
            table_ref, alias = match.groups()
            table_name = table_ref.split('.')[-1]
            join_order.append((alias, table_name))
        else:
            single = re.match(r'([A-Za-z_][A-Za-z0-9_]*)', part)
            if single:
                ident = single.group(1)
                table_name = alias_to_table.get(ident, ident)
                join_order.append((ident, table_name))
    
    # Resolve alias -> table for lookup
    alias_to_table_resolved = dict(alias_to_table)
    for alias, tbl in join_order:
        alias_to_table_resolved[alias] = tbl
    
    for cond_str in on_conditions:
        parsed = _parse_column_eq_column(cond_str)
        if not parsed:
            continue
        alias_left, attr_left, alias_right, attr_right = parsed
        
        table_left = alias_to_table_resolved.get(alias_left, alias_left)
        table_right = alias_to_table_resolved.get(alias_right, alias_right)
        
        # Determine parent (earlier in JOIN order) vs child
        idx_left = next((i for i, (a, _) in enumerate(join_order) if a == alias_left), 999)
        idx_right = next((i for i, (a, _) in enumerate(join_order) if a == alias_right), 999)
        
        if idx_left <= idx_right:
            parent_table, parent_attr = table_left, attr_left
            child_table, child_attr = table_right, attr_right
        else:
            parent_table, parent_attr = table_right, attr_right
            child_table, child_attr = table_left, attr_left
        
        join_conditions.append(JoinCondition(
            table_parent=parent_table,
            attr_parent=parent_attr,
            table_child=child_table,
            attr_child=child_attr,
        ))
    
    return join_conditions


def get_join_graph(
    sql: str,
    schema: List[Dict[str, Any]],
    query_tables: Optional[List[str]] = None,
) -> Optional[JoinGraph]:
    """
    Build join graph from SQL and schema.
    
    Args:
        sql: Full SQL statement
        schema: List of schema dicts
        query_tables: Optional list of table names in query
        
    Returns:
        JoinGraph or None if no joins
    """
    if not has_join(sql):
        return None
    
    conditions = parse_join_conditions(sql)
    if not conditions:
        return None
    
    parent_to_children: Dict[str, List[str]] = {}
    child_to_parent: Dict[str, List[Tuple[str, str, str]]] = {}
    
    for jc in conditions:
        parent_to_children.setdefault(jc.table_parent, []).append(jc.table_child)
        child_to_parent.setdefault(jc.table_child, []).append(
            (jc.table_parent, jc.attr_parent, jc.attr_child)
        )
    
    return JoinGraph(
        conditions=conditions,
        parent_to_children=parent_to_children,
        child_to_parent=child_to_parent,
    )


def compute_table_processing_order(
    join_graph: Optional[JoinGraph],
    tables: List[str],
) -> List[str]:
    """
    Compute table processing order for join-aware pipeline.
    Parent tables (no dependencies) come first; children follow.
    
    Args:
        join_graph: JoinGraph from get_join_graph, or None for no joins
        tables: List of table names to process
        
    Returns:
        Ordered list of table names (topological order)
    """
    if not join_graph or not join_graph.conditions:
        return list(tables)
    
    # Topological sort: tables with no incoming join edges first
    # child_to_parent: table -> [(parent, attr_p, attr_c)]
    # Tables that are never children (or only parents) go first
    in_degree: Dict[str, int] = {t: 0 for t in tables}
    
    for child in join_graph.child_to_parent:
        if child in tables:
            in_degree[child] = len(join_graph.child_to_parent[child])
    
    # Tables not in child_to_parent have in_degree 0
    result: List[str] = []
    remaining = set(tables)
    
    while remaining:
        # Find tables with no dependencies (in_degree 0)
        ready = [t for t in remaining if in_degree.get(t, 0) == 0]
        if not ready:
            # Cycle or missing table - add remaining in arbitrary order
            ready = list(remaining)
        
        for t in ready:
            result.append(t)
            remaining.discard(t)
            # Reduce in_degree for children of t
            for child in join_graph.parent_to_children.get(t, []):
                if child in remaining:
                    in_degree[child] = max(0, in_degree.get(child, 0) - 1)
    
    return result
