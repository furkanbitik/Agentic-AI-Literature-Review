"""Semantic Scholar API client for academic paper search."""

import time
from datetime import datetime
from typing import Optional

import requests


BASE_URL = "https://api.semanticscholar.org/graph/v1"
FIELDS = "title,authors,year,citationCount,abstract,url,venue,publicationDate,externalIds,openAccessPdf"


def search_papers(
    query: str,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    limit: int = 20,
    offset: int = 0,
) -> dict:
    """Search for papers on Semantic Scholar."""
    if year_start is None:
        year_start = datetime.now().year - 5
    if year_end is None:
        year_end = datetime.now().year

    params = {
        "query": query,
        "fields": FIELDS,
        "limit": min(limit, 100),
        "offset": offset,
        "year": f"{year_start}-{year_end}",
    }
    resp = _request_with_retry(f"{BASE_URL}/paper/search", params=params)
    return resp.json() if resp else {"total": 0, "data": []}


def get_paper_details(paper_id: str) -> dict:
    """Get detailed information about a specific paper."""
    params = {"fields": FIELDS + ",references,citations,tldr"}
    resp = _request_with_retry(f"{BASE_URL}/paper/{paper_id}", params=params)
    return resp.json() if resp else {}


def get_most_cited(query: str, year_start: Optional[int] = None, top_n: int = 10) -> list[dict]:
    """Get most cited papers for a query, sorted by citation count."""
    if year_start is None:
        year_start = datetime.now().year - 5

    all_papers = []
    for offset in range(0, 100, 100):
        result = search_papers(query, year_start=year_start, limit=100, offset=offset)
        all_papers.extend(result.get("data", []))
        if len(result.get("data", [])) < 100:
            break
        time.sleep(1)

    all_papers.sort(key=lambda p: p.get("citationCount", 0) or 0, reverse=True)
    return all_papers[:top_n]


def search_datasets(query: str) -> list[dict]:
    """Search for datasets related to a query using Semantic Scholar."""
    params = {
        "query": f"{query} dataset benchmark",
        "fields": FIELDS,
        "limit": 20,
    }
    resp = _request_with_retry(f"{BASE_URL}/paper/search", params=params)
    if not resp:
        return []

    data = resp.json().get("data", [])
    dataset_papers = []
    for paper in data:
        abstract = (paper.get("abstract") or "").lower()
        title = (paper.get("title") or "").lower()
        if any(kw in abstract or kw in title for kw in ["dataset", "benchmark", "corpus", "data set"]):
            dataset_papers.append(paper)
    return dataset_papers


def _request_with_retry(url: str, params: dict, retries: int = 3) -> Optional[requests.Response]:
    """Make a request with retry logic for rate limiting."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                time.sleep(wait)
                continue
            resp.raise_for_status()
        except requests.RequestException:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None
