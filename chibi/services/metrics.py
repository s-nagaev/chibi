from influxdb_client import Point
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
from loguru import logger

from chibi.config import application_settings
from chibi.models import User
from chibi.schemas.app import MetricTagsSchema, UsageSchema
from chibi.services.task_manager import task_manager


class MetricsService:
    @classmethod
    def _prepare_point(cls, metric: UsageSchema, tags: MetricTagsSchema) -> Point:
        point = Point("usage")

        for tag_name, value in tags.model_dump(exclude_none=True).items():
            point.tag(tag_name, str(value))

        exclude_fields = {"completion_tokens_details", "prompt_tokens_details"}
        for field_name, value in metric.model_dump(exclude=exclude_fields).items():
            point.field(field_name, value)
        return point

    @classmethod
    async def _send_to_influx(cls, metric: UsageSchema, tags: MetricTagsSchema) -> None:
        point = cls._prepare_point(metric=metric, tags=tags)

        try:
            assert application_settings.influxdb_url
            assert application_settings.influxdb_token
            assert application_settings.influxdb_org
            assert application_settings.influxdb_bucket

            async with InfluxDBClientAsync(
                url=application_settings.influxdb_url,
                token=application_settings.influxdb_token,
                org=application_settings.influxdb_org,
            ) as client:
                write_api = client.write_api()
                await write_api.write(bucket=application_settings.influxdb_bucket, record=point)
        except Exception as e:
            logger.error(f"Failed to send metrics to InfluxDB due to exception: {e}")

    @classmethod
    def send_usage_metrics(cls, metric: UsageSchema, model: str, provider: str, user: User | None = None) -> None:
        if not application_settings.is_influx_configured:
            return None

        tags = MetricTagsSchema(
            user_id=user.id if user else 0,
            provider=provider,
            model=model,
        )
        task_manager.run_task(cls._send_to_influx(metric=metric, tags=tags))
