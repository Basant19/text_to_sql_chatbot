# app/graph/nodes/validate_node.py

import sys
from typing import Dict, Any
from app.logger import get_logger
from app.exception import CustomException
from app import utils

logger = get_logger("validate_node")


class ValidateNode:
    """
    Node to validate SQL queries against schema and safety rules.

    Checks:
      - Only SELECT queries allowed
      - Tables/columns exist in provided schema
      - Optionally enforce row limits or other safe execution rules
    """

    def __init__(self):
        try:
            # Any initialization if needed
            pass
        except Exception as e:
            logger.exception("Failed to initialize ValidateNode")
            raise CustomException(e, sys)

    def run(self, sql: str, schemas: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate SQL string.

        Returns:
          {
              "sql": validated SQL string (unchanged if valid),
              "valid": True/False,
              "errors": []  # list of error messages if any
          }
        Raises CustomException if unexpected failure occurs.
        """
        try:
            errors = []

            if not sql.strip():
                errors.append("SQL query is empty.")

            # Only allow SELECT queries
            if not utils.is_select_query(sql):
                errors.append("Only SELECT statements are allowed.")

            # Validate tables/columns
            missing_tables = utils.validate_tables_in_sql(sql, schemas)
            if missing_tables:
                errors.append(f"Tables not found in schema: {', '.join(missing_tables)}")

            valid = len(errors) == 0
            if valid:
                logger.info("ValidateNode: SQL validation passed")
            else:
                logger.warning(f"ValidateNode: SQL validation failed with errors: {errors}")

            return {"sql": sql, "valid": valid, "errors": errors}

        except CustomException:
            raise
        except Exception as e:
            logger.exception("ValidateNode.run failed")
            raise CustomException(e, sys)
