"""Build the Tamil Nadu Government Schemes knowledge base.

This crawler follows the official three-level structure:
  schemes.php -> scheme_list.php?dep_id=... -> scheme_details.php?id=...

It writes each fetched scheme page to Neo4j and builds the FAISS index used by
streamlit_app.py. Run it directly, or click Refresh Neo4j and FAISS in Streamlit.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from neo4j import GraphDatabase

START_URL = "https://www.tn.gov.in/schemes.php"
ALLOWED_HOST = "www.tn.gov.in"
OUTPUT_DIR = Path("data/processed")
HEADERS = {"User-Agent": "TamilNaduSchemesRAG/1.0 (educational project)"}


@dataclass(frozen=True)
class SchemeRecord:
    id: str
    name: str
    department: str
    detail_url: str
    content: str
    facts: tuple[tuple[str, str], ...]


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def is_allowed(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.netloc == ALLOWED_HOST


def fetch(url: str, delay_seconds: float) -> str:
    """Fetch one official page, with a small delay between requests."""
    response = requests.get(url, headers=HEADERS, timeout=45)
    response.raise_for_status()
    time.sleep(delay_seconds)
    return response.text


def page_text(html: str) -> str:
    """Keep content while removing repeated navigation and non-content markup."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "iframe"]):
        tag.decompose()
    return clean_text(soup.get_text(" ", strip=True))


def official_links(html: str, base_url: str, required_path: str) -> list[tuple[str, str]]:
    """Return unique official links whose URL contains the given PHP endpoint."""
    soup = BeautifulSoup(html, "html.parser")
    found: dict[str, str] = {}
    for anchor in soup.find_all("a", href=True):
        url = urljoin(base_url, anchor["href"])
        label = clean_text(anchor.get_text(" ", strip=True))
        if is_allowed(url) and required_path in url:
            found.setdefault(url, label)
    return list(found.items())


def labelled_facts(soup: BeautifulSoup) -> tuple[tuple[str, str], ...]:
    """Read label/value rows on a scheme-details page without inventing fields."""
    facts: dict[str, str] = {}
    for row in soup.find_all("tr"):
        cells = row.find_all(["th", "td"])
        if len(cells) < 2:
            continue
        label = clean_text(cells[0].get_text(" ", strip=True)).rstrip(":")
        value = clean_text(cells[1].get_text(" ", strip=True))
        if label and value and len(label) <= 100:
            facts[label] = value
    return tuple(facts.items())


def detail_record(url: str, fallback_name: str, department: str, delay_seconds: float) -> SchemeRecord:
    html = fetch(url, delay_seconds)
    soup = BeautifulSoup(html, "html.parser")
    facts = labelled_facts(soup)
    fact_map = {key.lower(): value for key, value in facts}
    name = next(
        (value for key, value in fact_map.items() if "scheme title" in key or "scheme name" in key),
        fallback_name or "Unnamed scheme",
    )
    page_department = next(
        (value for key, value in fact_map.items() if "concerned department" in key),
        department,
    )
    return SchemeRecord(
        id=stable_id(url),
        name=name,
        department=page_department,
        detail_url=url,
        content=page_text(html),
        facts=facts,
    )


def crawl_schemes(start_url: str, max_schemes: int, delay_seconds: float) -> list[SchemeRecord]:
    """Follow all department and scheme-detail links below the chosen start page."""
    start_html = fetch(start_url, delay_seconds)
    department_links = official_links(start_html, start_url, "scheme_list.php")
    if not department_links:
        raise ValueError("No department scheme-list links found on SCHEME_START_URL.")

    records: list[SchemeRecord] = []
    seen_details: set[str] = set()
    for department_url, department_name in department_links:
        print(f"Reading department: {department_name or department_url}")
        department_html = fetch(department_url, delay_seconds)
        for detail_url, scheme_name in official_links(department_html, department_url, "scheme_details.php"):
            if detail_url in seen_details:
                continue
            seen_details.add(detail_url)
            records.append(detail_record(detail_url, scheme_name, department_name, delay_seconds))
            print(f"  Loaded {len(records)}: {records[-1].name}")
            if max_schemes and len(records) >= max_schemes:
                return records
    return records


def create_constraints(driver, database: str) -> None:
    with driver.session(database=database) as session:
        session.run("CREATE CONSTRAINT scheme_id IF NOT EXISTS FOR (s:Scheme) REQUIRE s.id IS UNIQUE").consume()
        session.run("CREATE CONSTRAINT source_url IF NOT EXISTS FOR (p:SourcePage) REQUIRE p.url IS UNIQUE").consume()
        session.run("CREATE CONSTRAINT fact_id IF NOT EXISTS FOR (f:SchemeFact) REQUIRE f.id IS UNIQUE").consume()


def write_neo4j(driver, database: str, records: list[SchemeRecord]) -> None:
    """Write source-attributed Scheme nodes, facts, and relationships in one batch."""
    rows = []
    for record in records:
        facts = [
            {"id": stable_id(f"{record.id}|{label}|{value}"), "label": label, "value": value}
            for label, value in record.facts
        ]
        rows.append({
            "id": record.id, "name": record.name, "department": record.department,
            "url": record.detail_url, "content": record.content, "facts": facts,
        })
    query = """
    UNWIND $rows AS row
    MERGE (d:Department {name: row.department})
    MERGE (s:Scheme {id: row.id})
    SET s.name = row.name, s.source_url = row.url, s.content = row.content, s.updated_at = datetime()
    MERGE (p:SourcePage {url: row.url})
    SET p.title = row.name, p.fetched_at = datetime()
    MERGE (d)-[:ADMINISTERS]->(s)
    MERGE (s)-[:SOURCE]->(p)
    FOREACH (fact IN row.facts |
        MERGE (f:SchemeFact {id: fact.id})
        SET f.field = fact.label, f.value = fact.value
        MERGE (s)-[:HAS_FACT]->(f)
    )
    """
    with driver.session(database=database) as session:
        session.run(query, rows=rows).consume()


def build_faiss(records: list[SchemeRecord]) -> Path:
    """Index the full text from every individual scheme detail page."""
    documents = [
        Document(
            page_content=record.content,
            metadata={"source": record.detail_url, "scheme_id": record.id,
                      "scheme_name": record.name, "department": record.department},
        )
        for record in records
    ]
    chunks = RecursiveCharacterTextSplitter(chunk_size=1_000, chunk_overlap=150).split_documents(documents)
    if not chunks:
        raise ValueError("The scheme pages contained no text to index.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index_path = OUTPUT_DIR / "tn_schemes_faiss"
    FAISS.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-3-small")).save_local(str(index_path))
    return index_path


def main() -> None:
    load_dotenv()
    uri = os.getenv("NEO4J_URI")
    password = os.getenv("NEO4J_PASSWORD")
    if not uri or not password or not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set NEO4J_URI, NEO4J_PASSWORD, and OPENAI_API_KEY in .env.")

    start_url = os.getenv("SCHEME_START_URL", START_URL)
    max_schemes = int(os.getenv("MAX_SCHEMES", "0"))  # 0 means every listed scheme.
    delay_seconds = float(os.getenv("REQUEST_DELAY_SECONDS", "0.3"))
    records = crawl_schemes(start_url, max_schemes, delay_seconds)
    if not records:
        raise ValueError("No scheme-detail pages were collected.")

    database = os.getenv("NEO4J_DATABASE", "3641677c")
    driver = GraphDatabase.driver(uri, auth=(os.getenv("NEO4J_USERNAME", "3641677c"), password))
    try:
        driver.verify_connectivity()
        create_constraints(driver, database)
        write_neo4j(driver, database, records)
    finally:
        driver.close()
    index_path = build_faiss(records)
    print(f"Completed: {len(records)} schemes written to Neo4j and indexed at {index_path}")


if __name__ == "__main__":
    main()
