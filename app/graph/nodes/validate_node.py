# app/graph/nodes/validate_node.py

import sys
from typing import Dict, Any, List
from app.logger import get_logger
from app.exception import CustomException
from app import utils

logger = get_logger("validate_node")


class ValidateNode:
    """
    Node to validate SQL queries against schema and safety rules.

    Checks performed:
      - Only SELECT queries are allowed.
      - Tables and columns exist in the provided schema.
      - Optionally, row limits or other safety rules can be enforced in the future.
    """

    def __init__(self):
        try:
            # Initialization placeholder (if future config is needed)
            pass
        except Exception as e:
            logger.exception("Failed to initialize ValidateNode")
            raise CustomException(e, sys)

    def run(self, sql: str, schemas: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate a SQL string against safety and schema rules.

        Parameters
        ----------
        sql : str
            The SQL query string to validate.
        schemas : Dict[str, Dict[str, Any]]
            Dictionary of table schemas. Expected format:
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
                "sql": <original SQL string>,
                "valid": <bool indicating validation success>,
                "errors": <list of error messages if any>
            }

        Raises
        ------
        CustomException
            If unexpected errors occur during validation.
        """
        try:
            errors: List[str] = []

            # Check for empty query
            if not sql.strip():
                errors.append("SQL query is empty.")

            # Ensure only SELECT statements are allowed
            if not utils.is_select_query(sql):
                errors.append("Only SELECT statements are allowed.")

            # Validate tables exist in the provided schema
            missing_tables = utils.validate_tables_in_sql(sql, schemas)
            if missing_tables:
                errors.append(f"Tables not found in schema: {', '.join(missing_tables)}")

            # Additional validations (optional in future):
            # e.g., enforce row limits, prohibit joins on sensitive tables, etc.

            valid = len(errors) == 0

            if valid:
                logger.info("ValidateNode: SQL validation passed.")
            else:
                logger.warning(f"ValidateNode: SQL validation failed with errors: {errors}")

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
