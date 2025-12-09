import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chibi.config.app import ApplicationSettings
from chibi.schemas.app import MetricTagsSchema, UsageSchema
from chibi.services.metrics import MetricsService
from chibi.services.task_manager import task_manager


@pytest.fixture
def mock_settings():
    settings = MagicMock(spec=ApplicationSettings)
    settings.influxdb_url = "http://localhost:8086"
    settings.influxdb_token = "token"
    settings.influxdb_org = "org"
    settings.influxdb_bucket = "bucket"
    settings.is_influx_configured = True
    return settings


@pytest.fixture
def metric():
    return UsageSchema(
        completion_tokens=10,
        prompt_tokens=20,
        total_tokens=30,
        cost=Decimal("0.05"),
    )


@pytest.fixture
def tags():
    return MetricTagsSchema(
        user_id=123,
        user_name="test_user",
        provider="openai",
        model="gpt-4",
        completion_tokens=10,
        prompt_tokens=20,
        total_tokens=30,
        cost=Decimal("0.05"),
    )


@pytest.fixture(autouse=True)
async def clear_tasks():
    # Wait for existing tasks (cleanup)
    await task_manager.shutdown()
    yield
    # Wait for tasks created during test
    await task_manager.shutdown()


@pytest.mark.asyncio
async def test_send_to_influx(mock_settings, metric, tags):
    with patch("chibi.services.metrics.application_settings", mock_settings):
        # Mock InfluxDBClientAsync
        with patch("chibi.services.metrics.InfluxDBClientAsync") as mock_client_cls:
            mock_client = AsyncMock()
            mock_write_api = AsyncMock()

            mock_client_cls.return_value.__aenter__.return_value = mock_client
            # write_api() is a synchronous method returning an object
            mock_client.write_api = MagicMock(return_value=mock_write_api)

            await MetricsService._send_to_influx(metric, tags)

            mock_client_cls.assert_called_with(url="http://localhost:8086", token="token", org="org")
            mock_write_api.write.assert_called_once()


@pytest.mark.asyncio
async def test_send_usage_metrics_delegation(mock_settings, metric):
    """Test that send_usage_metrics delegates to task_manager"""
    with patch("chibi.services.metrics.application_settings", mock_settings):
        # Patch run_task on the global task_manager instance
        with patch.object(task_manager, "run_task") as mock_run_task:
            MetricsService.send_usage_metrics(metric, model="gpt-4", provider="openai")

            mock_run_task.assert_called_once()
            # Verify that the argument passed to run_task is a coroutine
            args, _ = mock_run_task.call_args
            assert asyncio.iscoroutine(args[0])
            # Clean up the coroutine to avoid warnings
            args[0].close()


@pytest.mark.asyncio
async def test_integration_with_manager(mock_settings, metric):
    """Integration test: Service -> Manager -> Execution"""
    with patch("chibi.services.metrics.application_settings", mock_settings):
        # Mock the actual sending logic
        with patch.object(MetricsService, "_send_to_influx", new_callable=AsyncMock) as mock_send:
            MetricsService.send_usage_metrics(metric, model="gpt-4", provider="openai")

            # Allow loop to switch context so task starts
            await asyncio.sleep(0.01)

            # Check that _send_to_influx was called with metric and tags
            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert call_args[1]["metric"] == metric
            assert call_args[1]["tags"].model == "gpt-4"
            assert call_args[1]["tags"].provider == "openai"
