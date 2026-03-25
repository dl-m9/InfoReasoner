# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Client for IG info gain service."""

import os
import time
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _convert_to_serializable(obj: Any) -> Any:
    """Convert numpy types and other non-serializable objects to Python native types."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {key: _convert_to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(item) for item in obj]
    return obj


def _normalize_answers(answers: Any) -> List[str]:
    """Normalize answers to a list of strings, handling various input types."""
    answers = _convert_to_serializable(answers)
    if not isinstance(answers, list):
        answers = [answers]

    flattened: List[Any] = []
    for item in answers:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    return [str(item) for item in flattened]


class IGClient:
    """Client for IG info gain computation service."""

    def __init__(
        self,
        service_url: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.service_url = service_url or os.getenv("IG_SERVICE_URL", "http://localhost:8000")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.service_url = self.service_url.rstrip("/")

        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=retry_delay,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def health_check(self) -> bool:
        try:
            response = self.session.get(f"{self.service_url}/health", timeout=self.timeout)
            return response.status_code == 200
        except Exception:
            return False

    def compute_info_gain(self, question: str, context: str, answers: List[str]) -> float:
        items = [{"question": question, "context": context, "answers": answers}]
        scores = self.compute_info_gain_batch(items)
        return scores[0] if scores else 0.0

    def compute_info_gain_batch(self, items: List[Dict[str, Any]]) -> List[float]:
        if not items:
            return []

        request_data = {
            "items": [
                {
                    "question": item["question"],
                    "context": item.get("context", ""),
                    "answers": _normalize_answers(item["answers"]),
                }
                for item in items
            ]
        }

        last_exception: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    f"{self.service_url}/compute_info_gain",
                    json=request_data,
                    timeout=self.timeout,
                )
                if response.status_code == 200:
                    result = response.json()
                    scores = result.get("scores", [])
                    errors = result.get("errors", [])
                    if errors and any(errors):
                        error_msgs = [e for e in errors if e]
                        print(f"[WARNING] IG service returned errors: {error_msgs}")
                    return scores

                error_msg = f"HTTP {response.status_code}: {response.text}"
                if attempt < self.max_retries:
                    print(f"[WARNING] Request failed, retrying... ({attempt + 1}/{self.max_retries})")
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise RuntimeError(error_msg)

            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < self.max_retries:
                    print(f"[WARNING] Request exception, retrying... ({attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"Failed to connect to IG service at {self.service_url}: {e}")

        if last_exception:
            raise RuntimeError(f"Failed after {self.max_retries} retries: {last_exception}")
        return []


_ig_client: Optional[IGClient] = None


def get_ig_client(service_url: Optional[str] = None) -> Optional[IGClient]:
    """Get or create global IG client instance."""
    global _ig_client

    url = service_url or os.getenv("IG_SERVICE_URL")
    if not url:
        return None

    if _ig_client is None or _ig_client.service_url != url:
        _ig_client = IGClient(service_url=url)
    return _ig_client


def is_ig_service_available() -> bool:
    client = get_ig_client()
    if client is None:
        return False
    return client.health_check()

