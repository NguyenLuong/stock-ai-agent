"""Configure structlog for JSON output in production, console in development."""

import structlog

_SENSITIVE_KEYS = frozenset({"api_key", "token", "password", "secret", "authorization"})

# Fields that remain at the top level of the JSON log output.
_TOP_LEVEL_FIELDS = frozenset({"timestamp", "component", "level", "message"})


def _filter_sensitive_keys(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: dict,
) -> dict:
    """Recursively remove sensitive fields from log output."""

    def _scrub(d: dict) -> None:
        for key in list(d.keys()):
            if key in _SENSITIVE_KEYS:
                del d[key]
            elif isinstance(d[key], dict):
                _scrub(d[key])

    _scrub(event_dict)
    return event_dict


def _format_log_structure(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: dict,
) -> dict:
    """Rename event→message and nest extra kwargs under context.

    Produces: {timestamp, component, level, message, context: {...}}
    """
    event_dict["message"] = event_dict.pop("event", "")
    context = {k: v for k, v in event_dict.items() if k not in _TOP_LEVEL_FIELDS}
    for k in list(context.keys()):
        del event_dict[k]
    if context:
        event_dict["context"] = context
    return event_dict


def configure_logging(env: str = "production") -> None:
    """Configure structlog with JSON renderer in production, console in dev."""
    shared_processors: list[structlog.types.Processor] = [
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.stdlib.add_log_level,
        _filter_sensitive_keys,
        _format_log_structure,
    ]

    if env == "production":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(0),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(component: str) -> structlog.types.BindableLogger:
    """Return a bound logger with the given component name."""
    return structlog.get_logger(component=component)
