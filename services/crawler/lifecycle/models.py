"""Pydantic models for the data lifecycle pipeline."""

from __future__ import annotations

from pydantic import BaseModel


class LifecyclePipelineResult(BaseModel):
    """Kết quả tổng hợp sau mỗi lần chạy lifecycle pipeline.

    Dùng để log và theo dõi hiệu quả pipeline:
    - total: tổng số bài viết được xử lý
    - summarized_count: số bài đã tóm tắt thành công
    - skipped_count: số bài bỏ qua (đã có summary)
    - failed_count: số bài lỗi khi gọi LLM
    - tokens_used: ước tính token sử dụng (prompt + summary)
    - duration_seconds: thời gian chạy pipeline (giây)
    """

    total: int
    summarized_count: int
    skipped_count: int
    failed_count: int
    tokens_used: int
    duration_seconds: float
