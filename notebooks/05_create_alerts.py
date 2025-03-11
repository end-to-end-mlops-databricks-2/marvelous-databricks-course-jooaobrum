# Databricks notebook source
# !pip install house_price-1.1.3-py3-none-any.whl

# COMMAND ----------
# MAGIC ### Create a query that checks the accuracy of the model
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql

w = WorkspaceClient()

srcs = w.data_sources.list()

alert_query = """
SELECT
  (MEAN(f1_score_weighted) * 100.0 AS avg_f1_score_weighted)
FROM mlops_prod.hotel_reservation.model_monitoring_profile_metrics"""


query = w.queries.create(
    query=sql.CreateQueryRequestQuery(
        display_name=f"hotel-reservation-alert-query-{time.time_ns()}",
        warehouse_id=srcs[0].warehouse_id,
        description="Alert on f1 score weighted from hotel reservation model",
        query_text=alert_query,
    )
)

alert = w.alerts.create(
    alert=sql.CreateAlertRequestAlert(
        condition=sql.AlertCondition(
            operand=sql.AlertConditionOperand(column=sql.AlertOperandColumn(name="avg_f1_score_weighted")),
            op=sql.AlertOperator.LESS_THAN,
            threshold=sql.AlertConditionThreshold(value=sql.AlertOperandValue(double_value=75)),
        ),
        display_name=f"house-price-mae-alert-{time.time_ns()}",
        query_id=query.id,
    )
)


# COMMAND ----------

# cleanup
w.queries.delete(id=query.id)
w.alerts.delete(id=alert.id)
