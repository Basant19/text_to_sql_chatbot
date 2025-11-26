# app/graph/nodes/validate_node.py

import sys
from typing import Dict, Any, List, Optional

from app.logger import get_logger
from app.exception import CustomException
from app import utils

logger = get_logger("validate_node")


class ValidateNode:
    """
    Node to validate SQL queries against schema and safety rules.

    Features:
    - Ensures only SELECT queries are allowed.
    - Checks that referenced tables and columns exist in the provided schema.
    - Enforces optional safety rules: max row limit, forbidden tables.
    - Returns structured validation info including errors.
    - Designed for testability and modularity.
    """

    def __init__(self, safety_rules: Optional[Dict[str, Any]] = None):
        """
        Optional safety_rules can include:
          - "max_row_limit": int
          - "forbidden_tables": List[str]
        """
        try:
            self.safety_rules = safety_rules or {}
            self.max_row_limit = self.safety_rules.get("max_row_limit")
            self.forbidden_tables = self.safety_rules.get("forbidden_tables", [])
            logger.info("ValidateNode initialized with safety_rules: %s", self.safety_rules)
        except Exception as e:
            logger.exception("Failed to initialize ValidateNode")
            raise CustomException(e, sys)

    def run(self, sql: str, schemas: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate SQL against schema and safety rules.

        Parameters
        ----------
        sql : str
            SQL query string to validate.
        schemas : Dict[str, Dict[str, Any]]
            Table schemas:
            {
                "table_name": {
                    "columns": ["col1", "col2", ...]
                },
                ...
            }

        Returns
        -------
        Dict[str, Any]
            {
                "sql": <possibly modified SQL>,
                "valid": <bool>,
                "errors": <list of error messages>
            }

        Raises
        ------
        CustomException
            For unexpected errors during validation.
        """
        try:
            errors: List[str] = []

            sql = sql.strip()

            # 1) Check for empty query
            if not sql:
                errors.append("SQL query is empty.")

            # 2) Allow only SELECT queries
            if sql and not utils.is_select_query(sql):
                errors.append("Only SELECT statements are allowed.")

            # 3) Validate referenced tables exist
            if sql:
                missing_tables = utils.validate_tables_in_sql(sql, schemas)
                if missing_tables:
                    errors.append(f"Tables not found in schema: {', '.join(missing_tables)}")

            # 4) Validate referenced columns exist
            if sql:
                missing_columns = utils.validate_columns_in_sql(sql, schemas)
                if missing_columns:
                    errors.append(f"Columns not found in schema: {', '.join(missing_columns)}")

            # 5) Enforce max row limit
            if sql and self.max_row_limit:
                if utils.exceeds_row_limit(sql, self.max_row_limit):
                    errors.append(f"Query exceeds maximum row limit of {self.max_row_limit}")
                    sql = utils.limit_sql_rows(sql, self.max_row_limit)  # automatically limit

            # 6) Check forbidden tables
            if sql and self.forbidden_tables:
                forbidden_used = utils.check_forbidden_tables(sql, self.forbidden_tables)
                if forbidden_used:
                    errors.append(f"Query uses forbidden tables: {', '.join(forbidden_used)}")

            valid = len(errors) == 0

            if valid:
                logger.info("ValidateNode: SQL validation passed.")
            else:
                logger.warning("ValidateNode: SQL validation failed: %s", errors)

            return {
                "sql": sql,
                "valid": valid,
                "errors": errors
            }

        except CustomException:
            raise
        except Exception as e:
            logger.exception("ValidateNode.run failed")
            raise CustomException(e, sys)
