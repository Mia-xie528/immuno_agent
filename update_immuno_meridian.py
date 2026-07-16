#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
update_immuno_meridian.py

用途：
    自动检索 PubMed 最近 7 天的中医免疫相关文献，
    调用兼容 OpenAI 接口格式的大模型抽取知识三元组，
    再把新知识融合进 nodes.json 和 edges.json。

安装依赖：
    pip install requests openai

运行方法：
    1. 把本文件与 nodes.json、edges.json 放在同一目录。
    2. 在下方“配置区”填写 LLM_API_BASE、LLM_API_KEY、LLM_MODEL。
    3. 在终端执行：
       python update_immuno_meridian.py

说明：
    - 不依赖外部数据库。
    - PubMed E-utilities 不需要 API Key。
    - 更新前会自动备份原 JSON 文件。
    - 同一个 PMID 重复运行时，不会重复增加证据计数。
"""

# =============================================================================
# 配置区：通常只需要修改这里
# =============================================================================

import os
import sys
import io
# 强制设定 Python 终端和请求使用 UTF-8 编码，彻底解决 `ascii` 报错
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pathlib import Path

# ---------------------------- 大模型配置 -------------------------------------
# 示例（DeepSeek 官方 OpenAI 兼容地址）：
# LLM_API_BASE = "https://api.deepseek.com"
#
# 其他兼容 OpenAI 接口格式的平台，也可以把这里改为相应的 base_url。
LLM_API_BASE = "https://api.deepseek.com/v1"

# 建议优先通过环境变量设置密钥：
# Windows PowerShell:
#   $env:LLM_API_KEY="你的密钥"
# Linux / macOS:
#   export LLM_API_KEY="你的密钥"
#
# 也可以直接把下面第二个参数替换成真实密钥，但不建议把密钥上传到公开仓库。
LLM_API_KEY = "sk-9e03334ab0434383aad7567f8ba65fa8"
print("【调试】当前读取到的密钥是:", LLM_API_KEY)
# 模型名称必须与所使用的平台一致。
# DeepSeek 示例：deepseek-v4-flash
# 智谱或其他平台请改成该平台实际提供的兼容模型名称。
LLM_MODEL = "deepseek-chat"

# 大模型调用参数
LLM_TEMPERATURE = 0.1
LLM_TIMEOUT_SECONDS = 90
LLM_MAX_RETRIES = 3

# ---------------------------- PubMed 配置 ------------------------------------
PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# ====================== 插入这部分 ======================
SEARCH_QUERY = (
    'acupuncture OR meridian OR "traditional Chinese medicine" OR "spleen qi" OR "wei qi" '
    'OR immunology OR cytokine OR inflammation OR immune OR neuroimmune'
)
# ========================================================

LOOKBACK_DAYS = 7
MAX_ARTICLES = 10
PUBMED_REQUEST_TIMEOUT_SECONDS = 30
PUBMED_MAX_RETRIES = 3

# NCBI 建议工具提供名称和联系邮箱。邮箱可以留空，但正式长期运行时建议填写。
NCBI_TOOL_NAME = "immuno_meridian_updater"
NCBI_CONTACT_EMAIL = ""

# ---------------------------- 文件路径配置 -----------------------------------
# 默认以本脚本所在目录为工作目录，而不是以终端当前目录为准。
SCRIPT_DIR = Path(__file__).resolve().parent

NODES_FILE = SCRIPT_DIR / "nodes.json"
EDGES_FILE = SCRIPT_DIR / "edges.json"
BACKUP_DIR = SCRIPT_DIR / "backups"
REPORT_FILE = SCRIPT_DIR / "update_report.md"
LOG_FILE = SCRIPT_DIR / "update_immuno_meridian.log"

# JSON 写入格式
JSON_INDENT = 2
JSON_ENSURE_ASCII = False

# ---------------------------- 知识约束配置 -----------------------------------
# 大模型应优先使用这些实体类型。
ALLOWED_ENTITY_TYPES = {
    "经络",
    "穴位",
    "脏腑",
    "免疫细胞",
    "细胞因子",
    "信号通路",
    "能量代谢物",
    # 以下两个是容错类型，避免把神经结构或无法确定的实体硬塞入错误类别。
    "神经结构",
    "其他",
}

# 按需求限定关系类型。
ALLOWED_RELATIONS = {"促进", "抑制", "转化", "归经", "表里"}

# =============================================================================
# 正式代码区：一般不需要修改
# =============================================================================

import copy
import hashlib
import json
import logging
import random
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import requests
except ImportError as exc:
    raise SystemExit(
        "缺少 requests 库。请先执行：pip install requests openai"
    ) from exc

try:
    from openai import OpenAI
except (ImportError, AttributeError) as exc:
    raise SystemExit(
        "缺少新版 openai 库，或当前版本过旧。"
        "请先执行：pip install -U requests openai"
    ) from exc


# -----------------------------------------------------------------------------
# 数据结构
# -----------------------------------------------------------------------------

@dataclass
class Article:
    """保存一篇 PubMed 文献的基本信息。"""

    pmid: str
    title: str
    abstract: str
    journal: str = ""
    publication_date: str = ""
    doi: str = ""
    authors: List[str] = field(default_factory=list)

    @property
    def pubmed_url(self) -> str:
        """返回该文献的 PubMed 页面地址。"""
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"


@dataclass
class Triple:
    """保存一条规范化后的知识三元组。"""

    entity1: str
    entity1_type: str
    relation: str
    entity2: str
    entity2_type: str
    evidence: str = ""


@dataclass
class ExtractionResult:
    """保存某篇文献的大模型抽取结果。"""

    relevant: bool
    triples: List[Triple] = field(default_factory=list)
    reason: str = ""
    raw_response: str = ""


@dataclass
class UpdateStats:
    """记录本次运行的数据统计，最后用于生成报告。"""

    retrieved_articles: int = 0
    parsed_articles: int = 0
    valid_articles: int = 0
    irrelevant_articles: int = 0
    failed_articles: int = 0

    new_nodes: int = 0
    new_edges: int = 0
    updated_nodes: int = 0
    updated_edges: int = 0

    skipped_triples: int = 0
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    failed_article_details: List[Dict[str, str]] = field(default_factory=list)


# -----------------------------------------------------------------------------
# 日志
# -----------------------------------------------------------------------------

def setup_logging() -> None:
    """同时把日志输出到终端和日志文件。"""

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
        ],
        force=True,
    )


# -----------------------------------------------------------------------------
# 通用工具函数
# -----------------------------------------------------------------------------

def utc_timestamp() -> str:
    """生成适合文件名使用的 UTC 时间戳。"""
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def now_local_text() -> str:
    """生成便于阅读的本地时间。"""
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def normalize_text(value: Any) -> str:
    """
    统一文本格式：
    - 转成字符串
    - 合并多余空格
    - 去掉首尾空白
    """
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_name(value: Any) -> str:
    """
    用于比较实体名称是否相同。

    这里会：
    - 去掉全部空白
    - 转为小写
    - 统一常见全角标点
    但不会擅自翻译或合并不同术语。
    """
    text = normalize_text(value).lower()
    replacements = {
        "（": "(",
        "）": ")",
        "，": ",",
        "：": ":",
        "－": "-",
        "—": "-",
        "–": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", "", text)


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    """去重但保留原有顺序。"""
    result: List[str] = []
    seen: Set[str] = set()
    for item in items:
        normalized = normalize_text(item)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def safe_int(value: Any, default: int = 0) -> int:
    """尽可能把值转成整数；失败时返回默认值。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_first(mapping: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    """按顺序读取字典中的第一个有效字段。"""
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return default


def text_from_xml(element: Optional[ET.Element]) -> str:
    """
    提取 XML 元素中的全部文本。

    PubMed 标题或摘要中可能包含斜体、上下标等子标签，
    不能只使用 element.text。
    """
    if element is None:
        return ""
    return normalize_text("".join(element.itertext()))


def atomic_write_json(path: Path, data: Any) -> None:
    """
    原子化写入 JSON。

    先写入临时文件，再替换正式文件，避免程序中途退出导致 JSON 只写了一半。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=JSON_ENSURE_ASCII,
            indent=JSON_INDENT,
        )
        file.write("\n")

    os.replace(temp_path, path)


def atomic_write_text(path: Path, text: str) -> None:
    """原子化写入普通文本文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(text, encoding="utf-8")
    os.replace(temp_path, path)


# -----------------------------------------------------------------------------
# PubMed 检索与解析
# -----------------------------------------------------------------------------

def request_with_retry(
    session: requests.Session,
    url: str,
    params: Dict[str, Any],
    response_type: str,
) -> Any:
    """
    带重试的 HTTP GET 请求。

    response_type:
        - "json"：返回 JSON
        - "text"：返回文本
    """

    last_error: Optional[Exception] = None

    for attempt in range(1, PUBMED_MAX_RETRIES + 1):
        try:
            response = session.get(
                url,
                params=params,
                timeout=PUBMED_REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()

            if response_type == "json":
                return response.json()
            return response.text

        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            logging.warning(
                "PubMed 请求失败，第 %d/%d 次尝试：%s",
                attempt,
                PUBMED_MAX_RETRIES,
                exc,
            )
            if attempt < PUBMED_MAX_RETRIES:
                time.sleep(2 ** (attempt - 1))

    raise RuntimeError(f"PubMed 请求在多次重试后仍失败：{last_error}")


def get_date_range() -> Tuple[str, str]:
    """
    计算检索日期范围。

    LOOKBACK_DAYS=7 时，包含今天在内共 7 个自然日。
    E-utilities 使用 YYYY/MM/DD 格式。
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=max(LOOKBACK_DAYS - 1, 0))
    return start_date.strftime("%Y/%m/%d"), end_date.strftime("%Y/%m/%d")


def search_pubmed_ids(session: requests.Session) -> Tuple[List[str], str, str]:
    """
    使用 ESearch 检索最近若干天的 PubMed ID。

    datetype=pdat 表示按出版日期筛选。
    sort=pub+date 表示优先返回较新的文献。
    """
    start_date, end_date = get_date_range()

    params: Dict[str, Any] = {
        "db": "pubmed",
        "term": SEARCH_QUERY,
        "retmode": "json",
        "retmax": MAX_ARTICLES,
        "sort": "pub date",
        "datetype": "pdat",
        "mindate": start_date,
        "maxdate": end_date,
        "tool": NCBI_TOOL_NAME,
    }

    if NCBI_CONTACT_EMAIL:
        params["email"] = NCBI_CONTACT_EMAIL

    logging.info(
        "正在检索 PubMed，日期范围：%s 至 %s，最多 %d 篇。",
        start_date,
        end_date,
        MAX_ARTICLES,
    )

    payload = request_with_retry(
        session=session,
        url=PUBMED_ESEARCH_URL,
        params=params,
        response_type="json",
    )

    id_list = payload.get("esearchresult", {}).get("idlist", [])
    pmids = [normalize_text(pmid) for pmid in id_list if normalize_text(pmid)]

    logging.info("PubMed 返回 %d 个 PMID。", len(pmids))
    return pmids, start_date, end_date


def parse_pubmed_date(article_element: ET.Element) -> str:
    """尽可能从 PubMed XML 中解析出版日期。"""

    pub_date = article_element.find(
        "./MedlineCitation/Article/Journal/JournalIssue/PubDate"
    )

    if pub_date is not None:
        medline_date = text_from_xml(pub_date.find("MedlineDate"))
        if medline_date:
            return medline_date

        year = text_from_xml(pub_date.find("Year"))
        month = text_from_xml(pub_date.find("Month"))
        day = text_from_xml(pub_date.find("Day"))

        parts = [part for part in (year, month, day) if part]
        if parts:
            return "-".join(parts)

    # 如果期刊出版日期缺失，再尝试读取 ArticleDate。
    article_date = article_element.find(
        "./MedlineCitation/Article/ArticleDate"
    )
    if article_date is not None:
        year = text_from_xml(article_date.find("Year"))
        month = text_from_xml(article_date.find("Month"))
        day = text_from_xml(article_date.find("Day"))
        parts = [part for part in (year, month, day) if part]
        if parts:
            return "-".join(parts)

    return ""


def parse_authors(article_element: ET.Element) -> List[str]:
    """解析作者姓名。"""

    authors: List[str] = []

    for author in article_element.findall(
        "./MedlineCitation/Article/AuthorList/Author"
    ):
        collective_name = text_from_xml(author.find("CollectiveName"))
        if collective_name:
            authors.append(collective_name)
            continue

        last_name = text_from_xml(author.find("LastName"))
        fore_name = text_from_xml(author.find("ForeName"))
        initials = text_from_xml(author.find("Initials"))

        name_parts = [part for part in (fore_name or initials, last_name) if part]
        if name_parts:
            authors.append(" ".join(name_parts))

    return unique_preserve_order(authors)


def parse_doi(article_element: ET.Element) -> str:
    """从 ArticleIdList 中寻找 DOI。"""

    for article_id in article_element.findall(
        "./PubmedData/ArticleIdList/ArticleId"
    ):
        if article_id.attrib.get("IdType", "").lower() == "doi":
            return text_from_xml(article_id)
    return ""


def parse_abstract(article_element: ET.Element) -> str:
    """
    解析摘要。

    有些摘要分为 BACKGROUND、METHODS、RESULTS 等多个部分，
    这里会保留段落标签并拼接。
    """

    abstract_parts: List[str] = []

    for abstract_text in article_element.findall(
        "./MedlineCitation/Article/Abstract/AbstractText"
    ):
        content = text_from_xml(abstract_text)
        if not content:
            continue

        label = normalize_text(abstract_text.attrib.get("Label", ""))
        if label:
            abstract_parts.append(f"{label}: {content}")
        else:
            abstract_parts.append(content)

    return "\n".join(abstract_parts)


def fetch_pubmed_articles(
    session: requests.Session,
    pmids: Sequence[str],
) -> List[Article]:
    """使用 EFetch 一次性下载并解析文献基本信息和摘要。"""

    if not pmids:
        return []

    params: Dict[str, Any] = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "tool": NCBI_TOOL_NAME,
    }

    if NCBI_CONTACT_EMAIL:
        params["email"] = NCBI_CONTACT_EMAIL

    logging.info("正在通过 EFetch 获取文献标题和摘要。")

    xml_text = request_with_retry(
        session=session,
        url=PUBMED_EFETCH_URL,
        params=params,
        response_type="text",
    )

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise RuntimeError(f"PubMed XML 解析失败：{exc}") from exc

    articles: List[Article] = []

    for article_element in root.findall("./PubmedArticle"):
        pmid = text_from_xml(article_element.find("./MedlineCitation/PMID"))
        title = text_from_xml(
            article_element.find("./MedlineCitation/Article/ArticleTitle")
        )
        abstract = parse_abstract(article_element)
        journal = text_from_xml(
            article_element.find("./MedlineCitation/Article/Journal/Title")
        )

        if not pmid:
            logging.warning("跳过一条缺少 PMID 的 PubMed 记录。")
            continue

        articles.append(
            Article(
                pmid=pmid,
                title=title or "(无标题)",
                abstract=abstract,
                journal=journal,
                publication_date=parse_pubmed_date(article_element),
                doi=parse_doi(article_element),
                authors=parse_authors(article_element),
            )
        )

    # EFetch 返回顺序通常与输入一致，但这里显式按 PMID 输入顺序排序。
    order = {pmid: index for index, pmid in enumerate(pmids)}
    articles.sort(key=lambda article: order.get(article.pmid, 10**9))

    logging.info("成功解析 %d 篇文献。", len(articles))
    return articles


# -----------------------------------------------------------------------------
# 大模型知识抽取
# -----------------------------------------------------------------------------

ENTITY_TYPE_ALIASES = {
    "经脉": "经络",
    "meridian": "经络",
    "channel": "经络",
    "acupoint": "穴位",
    "acupuncture point": "穴位",
    "穴": "穴位",
    "organ": "脏腑",
    "zang-fu": "脏腑",
    "zangfu": "脏腑",
    "immune cell": "免疫细胞",
    "免疫细胞": "免疫细胞",
    "cytokine": "细胞因子",
    "细胞因子": "细胞因子",
    "signaling pathway": "信号通路",
    "signal pathway": "信号通路",
    "pathway": "信号通路",
    "代谢物": "能量代谢物",
    "metabolite": "能量代谢物",
    "energy metabolite": "能量代谢物",
    "neural structure": "神经结构",
    "nerve": "神经结构",
    "神经": "神经结构",
    "other": "其他",
}

RELATION_ALIASES = {
    "促进": "促进",
    "激活": "促进",
    "增强": "促进",
    "上调": "促进",
    "升高": "促进",
    "增加": "促进",
    "promote": "促进",
    "promotes": "促进",
    "activate": "促进",
    "activates": "促进",
    "enhance": "促进",
    "enhances": "促进",
    "upregulate": "促进",
    "upregulates": "促进",
    "increase": "促进",
    "increases": "促进",

    "抑制": "抑制",
    "下调": "抑制",
    "降低": "抑制",
    "减少": "抑制",
    "inhibit": "抑制",
    "inhibits": "抑制",
    "suppress": "抑制",
    "suppresses": "抑制",
    "downregulate": "抑制",
    "downregulates": "抑制",
    "decrease": "抑制",
    "decreases": "抑制",

    "转化": "转化",
    "转换": "转化",
    "分化": "转化",
    "convert": "转化",
    "converts": "转化",
    "transform": "转化",
    "transforms": "转化",
    "differentiate": "转化",
    "differentiates": "转化",

    "归经": "归经",
    "channel tropism": "归经",
    "meridian tropism": "归经",

    "表里": "表里",
    "interior-exterior": "表里",
    "exterior-interior": "表里",
}


def normalize_entity_type(raw_type: Any) -> str:
    """把模型输出的实体类型规范到允许集合中。"""

    value = normalize_text(raw_type)
    if value in ALLOWED_ENTITY_TYPES:
        return value

    normalized = value.lower()
    if normalized in ENTITY_TYPE_ALIASES:
        return ENTITY_TYPE_ALIASES[normalized]

    # 用包含关系做一次宽松匹配。
    for alias, canonical in ENTITY_TYPE_ALIASES.items():
        if alias in normalized:
            return canonical

    return "其他"


def normalize_relation(raw_relation: Any) -> Optional[str]:
    """
    把模型输出的关系规范到五种允许关系中。

    无法可靠映射时返回 None，避免把不合规关系写入图谱。
    """

    value = normalize_text(raw_relation)
    if value in ALLOWED_RELATIONS:
        return value

    normalized = value.lower()
    if normalized in RELATION_ALIASES:
        return RELATION_ALIASES[normalized]

    for alias, canonical in RELATION_ALIASES.items():
        if alias in normalized:
            return canonical

    return None


def build_llm_messages(article: Article) -> List[Dict[str, str]]:
    """构造发送给大模型的提示词。"""

    system_prompt = """
你是一名严谨的中医免疫学知识图谱信息抽取专家。

任务：
根据用户给出的 PubMed 文献标题和摘要，判断它是否包含“中医/针灸/经络/脏腑理论”与
“免疫、炎症、细胞因子、神经免疫或能量代谢”之间的明确关联。

输出规则：
1. 如果文献与上述主题无关，只输出两个汉字：无关
2. 如果相关，只输出一个合法 JSON 对象，不要输出 Markdown、代码围栏或额外解释。
3. JSON 格式必须是：
{
  "relevant": true,
  "reason": "一句简短判断依据",
  "triples": [
    {
      "entity1": "实体1",
      "entity1_type": "实体1类型",
      "relation": "关系",
      "entity2": "实体2",
      "entity2_type": "实体2类型",
      "evidence": "摘要中支持该三元组的简短证据，必须忠于原文"
    }
  ]
}

实体类型优先从以下集合选择：
经络、穴位、脏腑、免疫细胞、细胞因子、信号通路、能量代谢物、神经结构、其他。

关系只能从以下集合选择：
促进、抑制、转化、归经、表里。

特别要求：
- 不得仅凭常识补充摘要中没有陈述的知识。
- 不得把“相关性”强行写成“促进”或“抑制”。
- 如果摘要只说明相关但没有方向，宁可不输出该条三元组。
- 实体名称应简洁、规范；中医术语优先用中文，现代生物医学实体可保留通用英文缩写。
- 同一事实不要重复输出。
- 最多输出 12 条三元组。
""".strip()

    abstract_text = article.abstract or "（PubMed 未提供摘要，请仅根据标题谨慎判断。）"

    user_prompt = f"""
PMID：{article.pmid}
标题：{article.title}
摘要：
{abstract_text}
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    从模型响应中提取 JSON 对象。

    即使模型偶尔错误地加上 ```json 代码围栏，也尽量容错解析。
    """

    cleaned = normalize_text(text)

    # 去掉常见 Markdown 代码围栏。
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        payload = json.loads(cleaned)
        if not isinstance(payload, dict):
            raise ValueError("模型返回的 JSON 顶层不是对象。")
        return payload
    except json.JSONDecodeError:
        pass

    # 尝试截取第一个 { 到最后一个 }。
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        candidate = cleaned[start : end + 1]
        payload = json.loads(candidate)
        if not isinstance(payload, dict):
            raise ValueError("模型返回的 JSON 顶层不是对象。")
        return payload

    raise ValueError("未能从模型响应中解析出 JSON 对象。")


def parse_extraction_response(raw_response: str) -> ExtractionResult:
    """校验并规范化大模型返回的数据。"""

    cleaned = normalize_text(raw_response)

    if cleaned in {"无关", "不相关", "irrelevant"}:
        return ExtractionResult(
            relevant=False,
            triples=[],
            reason="模型判定文献与主题无关。",
            raw_response=raw_response,
        )

    payload = extract_json_object(raw_response)
    relevant = bool(payload.get("relevant", True))
    reason = normalize_text(payload.get("reason", ""))

    if not relevant:
        return ExtractionResult(
            relevant=False,
            triples=[],
            reason=reason or "模型判定文献与主题无关。",
            raw_response=raw_response,
        )

    raw_triples = payload.get("triples", [])
    if not isinstance(raw_triples, list):
        raise ValueError("模型返回的 triples 不是列表。")

    triples: List[Triple] = []
    seen: Set[Tuple[str, str, str]] = set()

    for item in raw_triples[:12]:
        if not isinstance(item, dict):
            continue

        entity1 = normalize_text(
            get_first(item, ["entity1", "subject", "head"], "")
        )
        entity2 = normalize_text(
            get_first(item, ["entity2", "object", "tail"], "")
        )
        relation = normalize_relation(
            get_first(item, ["relation", "predicate"], "")
        )

        if not entity1 or not entity2 or not relation:
            continue

        # 避免把同一实体连向自己。
        if normalize_name(entity1) == normalize_name(entity2):
            continue

        entity1_type = normalize_entity_type(
            get_first(item, ["entity1_type", "subject_type", "head_type"], "其他")
        )
        entity2_type = normalize_entity_type(
            get_first(item, ["entity2_type", "object_type", "tail_type"], "其他")
        )
        evidence = normalize_text(item.get("evidence", ""))

        key = (normalize_name(entity1), relation, normalize_name(entity2))
        if key in seen:
            continue

        seen.add(key)
        triples.append(
            Triple(
                entity1=entity1,
                entity1_type=entity1_type,
                relation=relation,
                entity2=entity2,
                entity2_type=entity2_type,
                evidence=evidence,
            )
        )

    # 如果模型称相关但没有任何合规三元组，不把它计为有效文献。
    if not triples:
        return ExtractionResult(
            relevant=False,
            triples=[],
            reason=reason or "没有抽取到方向明确且格式合规的三元组。",
            raw_response=raw_response,
        )

    return ExtractionResult(
        relevant=True,
        triples=triples,
        reason=reason,
        raw_response=raw_response,
    )


def create_llm_client() -> OpenAI:
    """创建 OpenAI 兼容客户端，并强制注入底层编码修复。"""
    
    # 1. 强制修复 Python 编码环境变量 (防止中文 ASCII 报错)
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'
    sys.stdout.reconfigure(encoding='utf-8')

    # 2. 强制指定你的密钥（在函数内部强行赋值，不再依赖外部占位符判断）
    global LLM_API_KEY, LLM_API_BASE, LLM_MODEL
    LLM_API_KEY = "sk-9e03334ab0434383aad7567f8ba65fa8"  # 👈 记得把这里换成你真实的 Key！
    LLM_API_BASE = "https://api.deepseek.com/v1"
    LLM_MODEL = "deepseek-chat"

    # 极简防空检查
    if not normalize_text(LLM_API_BASE):
        raise RuntimeError("LLM_API_BASE 不能为空。")
    if not normalize_text(LLM_MODEL):
        raise RuntimeError("LLM_MODEL 不能为空。")

    # 3. 使用自定义 HTTP 客户端修复底层传输编码 Bug
    import httpx
    http_client = httpx.Client(
        timeout=LLM_TIMEOUT_SECONDS,
        limits=httpx.Limits(max_keepalive_connections=5)
    )

    return OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_API_BASE.rstrip("/"),
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=0,
        http_client=http_client
    )


def extract_triples_with_llm(
    client: OpenAI,
    article: Article,
) -> ExtractionResult:
    """调用大模型抽取一篇文献中的知识三元组。"""

    last_error: Optional[Exception] = None

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            logging.info(
                "正在抽取 PMID %s：%s",
                article.pmid,
                article.title[:80],
            )

            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=build_llm_messages(article),
                temperature=LLM_TEMPERATURE,
                stream=False,
            )

            if not response.choices:
                raise RuntimeError("模型响应中没有 choices。")

            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("模型返回了空内容。")

            return parse_extraction_response(content)

        except Exception as exc:
            # 这里捕获兼容端点可能抛出的不同异常类型。
            last_error = exc
            logging.warning(
                "PMID %s 的大模型调用/解析失败，第 %d/%d 次：%s",
                article.pmid,
                attempt,
                LLM_MAX_RETRIES,
                exc,
            )
            if attempt < LLM_MAX_RETRIES:
                time.sleep(2 ** (attempt - 1))

    raise RuntimeError(
        f"PMID {article.pmid} 在多次重试后仍无法完成知识抽取：{last_error}"
    )


# -----------------------------------------------------------------------------
# JSON 图谱读取与结构兼容
# -----------------------------------------------------------------------------

def load_json_or_default(path: Path, default: Any) -> Any:
    """
    读取 JSON。

    文件不存在时返回默认值，保证脚本仍可独立启动；
    但会在日志和报告中提示。
    """

    if not path.exists():
        logging.warning("%s 不存在，将使用空数据结构创建。", path.name)
        return copy.deepcopy(default)

    try:
        with path.open("r", encoding="utf-8-sig") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{path.name} 不是合法 JSON，位置：第 {exc.lineno} 行，"
            f"第 {exc.colno} 列。"
        ) from exc


def extract_collection(
    data: Any,
    preferred_key: str,
    file_name: str,
) -> Tuple[List[Dict[str, Any]], Any, Optional[str]]:
    """
    从 JSON 中找到节点或边列表，并保留原顶层结构。

    支持两种常见格式：
    1. 直接列表：
       [ {...}, {...} ]

    2. 包装对象：
       {"nodes": [ {...} ]}
       {"edges": [ {...} ]}

    返回：
        collection：实际节点/边列表
        root_data：原顶层数据
        wrapper_key：如果是包装对象，则记录键名；否则为 None
    """

    if isinstance(data, list):
        collection = data
        wrapper_key = None
    elif isinstance(data, dict):
        candidate_keys = [
            preferred_key,
            "data",
            "items",
            "elements",
        ]

        found_key: Optional[str] = None
        for key in candidate_keys:
            if isinstance(data.get(key), list):
                found_key = key
                break

        if found_key is None:
            raise RuntimeError(
                f"{file_name} 的顶层是对象，但找不到可用的列表字段。"
                f"建议使用 {{\"{preferred_key}\": [...]}}。"
            )

        collection = data[found_key]
        wrapper_key = found_key
    else:
        raise RuntimeError(
            f"{file_name} 顶层必须是 JSON 数组或包含数组的 JSON 对象。"
        )

    for index, item in enumerate(collection):
        if not isinstance(item, dict):
            raise RuntimeError(
                f"{file_name} 第 {index + 1} 项不是 JSON 对象。"
            )

    return collection, data, wrapper_key


def node_id(node: Dict[str, Any]) -> str:
    """兼容读取节点 ID。"""
    return normalize_text(get_first(node, ["id", "node_id", "nodeId"], ""))


def node_name(node: Dict[str, Any]) -> str:
    """兼容读取节点名称。"""
    return normalize_text(get_first(node, ["name", "label", "title"], ""))


def node_type(node: Dict[str, Any]) -> str:
    """兼容读取节点类型。"""
    return normalize_text(get_first(node, ["type", "category", "entity_type"], ""))


def edge_id(edge: Dict[str, Any]) -> str:
    """兼容读取边 ID。"""
    return normalize_text(get_first(edge, ["id", "edge_id", "edgeId"], ""))


def edge_source(edge: Dict[str, Any]) -> str:
    """兼容读取边的起点。"""
    return normalize_text(get_first(edge, ["source", "from", "source_id"], ""))


def edge_target(edge: Dict[str, Any]) -> str:
    """兼容读取边的终点。"""
    return normalize_text(get_first(edge, ["target", "to", "target_id"], ""))


def edge_relation(edge: Dict[str, Any]) -> str:
    """兼容读取边关系。"""
    return normalize_text(get_first(edge, ["relation", "predicate", "label", "type"], ""))


def generate_next_id(
    existing_ids: Iterable[str],
    prefix: str,
    width: int = 6,
) -> str:
    """
    自动生成不重复 ID。

    例如：
        N000001
        E000001
    """

    existing = {normalize_text(value) for value in existing_ids if normalize_text(value)}
    max_number = 0

    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$", re.IGNORECASE)

    for value in existing:
        match = pattern.match(value)
        if match:
            max_number = max(max_number, int(match.group(1)))

    candidate_number = max_number + 1

    while True:
        candidate = f"{prefix}{candidate_number:0{width}d}"
        if candidate not in existing:
            return candidate
        candidate_number += 1


# -----------------------------------------------------------------------------
# 节点分层与三维坐标
# -----------------------------------------------------------------------------

LAYER_BY_ENTITY_TYPE = {
    "经络": "经络层",
    "穴位": "经络层",
    "脏腑": "脏腑层",
    "免疫细胞": "免疫细胞层",
    "细胞因子": "分子信号层",
    "信号通路": "分子信号层",
    "能量代谢物": "分子信号层",
    "神经结构": "神经免疫层",
    "其他": "其他层",
}

# 没有可识别脏腑时，各层的默认中心坐标。
LAYER_ANCHORS: Dict[str, Tuple[float, float, float]] = {
    "经络层": (0.0, 0.0, 0.0),
    "脏腑层": (0.0, 0.0, 35.0),
    "免疫细胞层": (0.0, 0.0, 75.0),
    "分子信号层": (0.0, 0.0, 115.0),
    "神经免疫层": (0.0, 0.0, 55.0),
    "其他层": (0.0, 0.0, 150.0),
}

# 这些坐标是前端布局用的“视觉锚点”，不是人体解剖学精确坐标。
ORGAN_ANCHORS: Dict[str, Tuple[float, float, float]] = {
    "心": (0.0, 65.0, 35.0),
    "heart": (0.0, 65.0, 35.0),
    "心包": (-12.0, 53.0, 38.0),
    "pericard": (-12.0, 53.0, 38.0),

    "肺": (35.0, 65.0, 35.0),
    "lung": (35.0, 65.0, 35.0),

    "肝": (-60.0, 15.0, 35.0),
    "liver": (-60.0, 15.0, 35.0),

    "胆": (-70.0, 0.0, 35.0),
    "gallbladder": (-70.0, 0.0, 35.0),

    "脾": (-30.0, -10.0, 35.0),
    "spleen": (-30.0, -10.0, 35.0),

    "胃": (5.0, -15.0, 35.0),
    "stomach": (5.0, -15.0, 35.0),

    "肾": (55.0, -35.0, 35.0),
    "kidney": (55.0, -35.0, 35.0),

    "膀胱": (45.0, -75.0, 35.0),
    "bladder": (45.0, -75.0, 35.0),

    "大肠": (40.0, -45.0, 35.0),
    "large intestine": (40.0, -45.0, 35.0),
    "colon": (40.0, -45.0, 35.0),

    "小肠": (15.0, -48.0, 35.0),
    "small intestine": (15.0, -48.0, 35.0),

    "三焦": (0.0, 20.0, 45.0),
    "triple burner": (0.0, 20.0, 45.0),
    "sanjiao": (0.0, 20.0, 45.0),
}


def infer_layer(entity_type_value: str) -> str:
    """根据实体类型自动确定图谱层级。"""
    return LAYER_BY_ENTITY_TYPE.get(entity_type_value, "其他层")


def deterministic_jitter(seed_text: str, magnitude: float = 10.0) -> Tuple[float, float, float]:
    """
    根据实体名称生成稳定的小范围偏移。

    同一实体每次运行会得到相同偏移，避免坐标无规律变化。
    """
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()

    values = []
    for index in range(3):
        number = int.from_bytes(
            digest[index * 2 : index * 2 + 2],
            byteorder="big",
        )
        normalized = number / 65535.0
        values.append((normalized * 2.0 - 1.0) * magnitude)

    return values[0], values[1], values[2]


def find_organ_anchor(
    entity_name: str,
    context_names: Sequence[str],
) -> Optional[Tuple[float, float, float]]:
    """
    根据实体自身名称及同一三元组中的上下文，寻找最接近的脏腑锚点。
    """

    search_space = " ".join([entity_name, *context_names]).lower()

    # 先匹配更长关键词，避免“心”先于“心包”命中。
    for keyword in sorted(ORGAN_ANCHORS.keys(), key=len, reverse=True):
        if keyword.lower() in search_space:
            return ORGAN_ANCHORS[keyword]

    return None


def assign_initial_coordinates(
    entity_name: str,
    entity_type_value: str,
    context_names: Sequence[str],
) -> Tuple[float, float, float]:
    """
    为新节点分配初始三维坐标。

    规则：
    1. 如果实体或上下文中能识别脏腑，则以该脏腑锚点为中心。
    2. 不同知识层在 z 轴上叠加不同高度。
    3. 再加入基于名称的稳定微小偏移，避免节点完全重叠。
    """

    layer = infer_layer(entity_type_value)
    layer_anchor = LAYER_ANCHORS[layer]
    organ_anchor = find_organ_anchor(entity_name, context_names)

    if organ_anchor is None:
        base_x, base_y, base_z = layer_anchor
    else:
        base_x, base_y, organ_z = organ_anchor
        # 保留脏腑的 x/y 邻近关系，但 z 高度由知识层决定。
        base_z = layer_anchor[2]
        # 脏腑层自身使用器官锚点高度。
        if layer == "脏腑层":
            base_z = organ_z

    jitter_x, jitter_y, jitter_z = deterministic_jitter(
        f"{entity_name}|{entity_type_value}",
        magnitude=9.0,
    )

    return (
        round(base_x + jitter_x, 3),
        round(base_y + jitter_y, 3),
        round(base_z + jitter_z * 0.45, 3),
    )


# -----------------------------------------------------------------------------
# 证据管理与知识融合
# -----------------------------------------------------------------------------

def get_evidence_pmids(item: Dict[str, Any]) -> List[str]:
    """读取并规范化节点/边中已有的 PMID 证据列表。"""

    raw = get_first(
        item,
        ["evidence_pmids", "pmids", "evidencePmids"],
        [],
    )

    if isinstance(raw, str):
        # 兼容逗号分隔字符串。
        values = re.split(r"[,;，；\s]+", raw)
    elif isinstance(raw, list):
        values = raw
    else:
        values = []

    return unique_preserve_order(normalize_text(value) for value in values)


def build_source_record(article: Article, evidence: str = "") -> Dict[str, Any]:
    """构造一条文献来源记录。"""

    record: Dict[str, Any] = {
        "pmid": article.pmid,
        "title": article.title,
        "pubmed_url": article.pubmed_url,
    }

    if article.doi:
        record["doi"] = article.doi
    if article.publication_date:
        record["publication_date"] = article.publication_date
    if evidence:
        record["evidence"] = evidence

    return record


def upsert_source_record(
    item: Dict[str, Any],
    article: Article,
    evidence: str = "",
) -> None:
    """向节点或边的 sources 中添加/更新文献来源。"""

    sources = item.get("sources")
    if not isinstance(sources, list):
        sources = []
        item["sources"] = sources

    existing_record: Optional[Dict[str, Any]] = None

    for source in sources:
        if isinstance(source, dict) and normalize_text(source.get("pmid")) == article.pmid:
            existing_record = source
            break

    new_record = build_source_record(article, evidence)

    if existing_record is None:
        sources.append(new_record)
    else:
        # 只更新非空字段，保留原有额外信息。
        for key, value in new_record.items():
            if value not in (None, ""):
                existing_record[key] = value


def add_evidence(
    item: Dict[str, Any],
    article: Article,
    evidence: str = "",
) -> bool:
    """
    给节点或边添加 PMID 证据。

    返回值：
        True  = 本 PMID 是第一次出现，证据计数已增加
        False = 本 PMID 已存在，没有重复计数
    """

    pmids = get_evidence_pmids(item)

    if article.pmid in pmids:
        # 即使 PMID 已有，也可补充来源详细信息。
        upsert_source_record(item, article, evidence)
        item["evidence_pmids"] = pmids
        return False

    pmids.append(article.pmid)
    item["evidence_pmids"] = pmids

    current_count = safe_int(
        get_first(item, ["evidence_count", "evidenceCount"], 0),
        default=0,
    )

    item["evidence_count"] = current_count + 1
    upsert_source_record(item, article, evidence)
    return True


def create_new_node(
    new_id: str,
    name: str,
    entity_type_value: str,
    article: Article,
    context_names: Sequence[str],
) -> Dict[str, Any]:
    """创建一个新的节点对象。"""

    layer = infer_layer(entity_type_value)
    x, y, z = assign_initial_coordinates(
        entity_name=name,
        entity_type_value=entity_type_value,
        context_names=context_names,
    )

    node = {
        "id": new_id,
        "name": name,
        "type": entity_type_value,
        "layer": layer,

        # 同时提供顶层 x/y/z 和 position，兼容不同前端读取方式。
        "x": x,
        "y": y,
        "z": z,
        "position": {
            "x": x,
            "y": y,
            "z": z,
        },

        "evidence_count": 0,
        "evidence_pmids": [],
        "sources": [],
        "created_at": now_local_text(),
        "updated_at": now_local_text(),
    }

    add_evidence(node, article)
    return node


def create_new_edge(
    new_id: str,
    source_id: str,
    target_id: str,
    relation: str,
    article: Article,
    evidence: str,
) -> Dict[str, Any]:
    """创建一个新的边对象。"""

    edge = {
        "id": new_id,
        "source": source_id,
        "target": target_id,
        "relation": relation,
        "evidence_count": 0,
        "evidence_pmids": [],
        "sources": [],
        "created_at": now_local_text(),
        "updated_at": now_local_text(),
    }

    add_evidence(edge, article, evidence)
    return edge


class KnowledgeGraph:
    """封装 nodes.json 与 edges.json 的查找、去重和更新逻辑。"""

    def __init__(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        stats: UpdateStats,
    ) -> None:
        self.nodes = nodes
        self.edges = edges
        self.stats = stats

        self.node_by_normalized_name: Dict[str, Dict[str, Any]] = {}
        self.node_by_id: Dict[str, Dict[str, Any]] = {}
        self.edge_by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

        self._rebuild_indexes()

    def _rebuild_indexes(self) -> None:
        """根据当前数据重建索引，并检查明显的数据问题。"""

        self.node_by_normalized_name.clear()
        self.node_by_id.clear()
        self.edge_by_key.clear()

        # 节点索引
        for node in self.nodes:
            current_id = node_id(node)
            current_name = node_name(node)

            if current_id:
                if current_id in self.node_by_id:
                    self.stats.warnings.append(
                        f"发现重复节点 ID：{current_id}"
                    )
                self.node_by_id[current_id] = node

            if current_name:
                normalized = normalize_name(current_name)
                if normalized in self.node_by_normalized_name:
                    existing = self.node_by_normalized_name[normalized]
                    self.stats.warnings.append(
                        "发现名称重复的节点："
                        f"{current_name}（ID: {node_id(existing)} 与 {current_id}）"
                    )
                else:
                    self.node_by_normalized_name[normalized] = node

        # 边索引
        for edge in self.edges:
            source = self._resolve_endpoint_id(edge_source(edge))
            target = self._resolve_endpoint_id(edge_target(edge))
            relation = normalize_relation(edge_relation(edge)) or edge_relation(edge)

            if source and target and relation:
                key = (source, target, relation)
                if key in self.edge_by_key:
                    self.stats.warnings.append(
                        "发现重复边："
                        f"{source} --{relation}--> {target}"
                    )
                else:
                    self.edge_by_key[key] = edge

    def _resolve_endpoint_id(self, endpoint: str) -> str:
        """
        把旧边端点尽可能解析为节点 ID。

        有些前端数据用 source/target 保存节点 ID，另一些旧数据直接保存
        节点名称。本函数同时兼容两种形式，避免重复创建同一条边。
        """

        value = normalize_text(endpoint)
        if not value:
            return ""

        if value in self.node_by_id:
            return value

        node = self.node_by_normalized_name.get(normalize_name(value))
        if node is not None and node_id(node):
            return node_id(node)

        # 无法解析时保留原值，以免破坏未知格式的数据。
        return value

    def _next_node_id(self) -> str:
        """生成下一个节点 ID。"""
        return generate_next_id(
            (node_id(node) for node in self.nodes),
            prefix="N",
        )

    def _next_edge_id(self) -> str:
        """生成下一个边 ID。"""
        return generate_next_id(
            (edge_id(edge) for edge in self.edges),
            prefix="E",
        )

    def get_or_create_node(
        self,
        name: str,
        entity_type_value: str,
        article: Article,
        context_names: Sequence[str],
    ) -> Tuple[Dict[str, Any], bool]:
        """
        获取已有节点或创建新节点。

        返回：
            (节点对象, 是否为新节点)
        """

        normalized = normalize_name(name)
        existing = self.node_by_normalized_name.get(normalized)

        if existing is not None:
            # 如果旧节点缺少类型或类型是“其他”，允许用更具体的新类型补充。
            old_type = node_type(existing)
            if (
                entity_type_value != "其他"
                and old_type in {"", "其他"}
            ):
                existing["type"] = entity_type_value
                existing["layer"] = infer_layer(entity_type_value)

            evidence_added = add_evidence(existing, article)
            if evidence_added:
                self.stats.updated_nodes += 1
                existing["updated_at"] = now_local_text()

            return existing, False

        new_node = create_new_node(
            new_id=self._next_node_id(),
            name=name,
            entity_type_value=entity_type_value,
            article=article,
            context_names=context_names,
        )

        self.nodes.append(new_node)
        self.node_by_normalized_name[normalized] = new_node
        self.node_by_id[node_id(new_node)] = new_node
        self.stats.new_nodes += 1

        return new_node, True

    def detect_opposite_conflict(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        article: Article,
    ) -> None:
        """
        检测“促进”和“抑制”的方向冲突。

        这里只记录潜在冲突，不自动删除或覆盖任何边。
        """

        opposite = {
            "促进": "抑制",
            "抑制": "促进",
        }.get(relation)

        if not opposite:
            return

        reverse_or_same_keys = [
            (source_id, target_id, opposite),
            # 某些模型或旧数据可能把方向写反，也一并提示人工检查。
            (target_id, source_id, opposite),
        ]

        for key in reverse_or_same_keys:
            existing_edge = self.edge_by_key.get(key)
            if existing_edge is None:
                continue

            conflict_key = (
                edge_id(existing_edge),
                article.pmid,
                source_id,
                target_id,
                relation,
            )

            # 防止同一冲突重复记录。
            if any(
                item.get("_dedupe_key") == conflict_key
                for item in self.stats.conflicts
            ):
                continue

            source_name = node_name(self.node_by_id.get(source_id, {})) or source_id
            target_name = node_name(self.node_by_id.get(target_id, {})) or target_id

            self.stats.conflicts.append(
                {
                    "_dedupe_key": conflict_key,
                    "source": source_name,
                    "target": target_name,
                    "new_relation": relation,
                    "existing_relation": opposite,
                    "new_pmid": article.pmid,
                    "existing_edge_id": edge_id(existing_edge),
                }
            )

    def add_or_update_edge(
        self,
        source_node: Dict[str, Any],
        target_node: Dict[str, Any],
        relation: str,
        article: Article,
        evidence: str,
    ) -> Tuple[Dict[str, Any], bool]:
        """
        获取已有边或创建新边。

        返回：
            (边对象, 是否为新边)
        """

        source_id = node_id(source_node)
        target_id = node_id(target_node)

        if not source_id or not target_id:
            raise RuntimeError("节点缺少 ID，无法创建边。")

        self.detect_opposite_conflict(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            article=article,
        )

        key = (source_id, target_id, relation)
        existing = self.edge_by_key.get(key)

        if existing is not None:
            evidence_added = add_evidence(existing, article, evidence)
            if evidence_added:
                self.stats.updated_edges += 1
                existing["updated_at"] = now_local_text()
            return existing, False

        new_edge = create_new_edge(
            new_id=self._next_edge_id(),
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            article=article,
            evidence=evidence,
        )

        self.edges.append(new_edge)
        self.edge_by_key[key] = new_edge
        self.stats.new_edges += 1

        return new_edge, True

    def merge_article(
        self,
        article: Article,
        extraction: ExtractionResult,
    ) -> None:
        """把一篇文献抽取出的全部三元组融合到图谱。"""

        for triple in extraction.triples:
            if triple.relation not in ALLOWED_RELATIONS:
                self.stats.skipped_triples += 1
                continue

            source_node, _ = self.get_or_create_node(
                name=triple.entity1,
                entity_type_value=triple.entity1_type,
                article=article,
                context_names=[triple.entity2],
            )

            target_node, _ = self.get_or_create_node(
                name=triple.entity2,
                entity_type_value=triple.entity2_type,
                article=article,
                context_names=[triple.entity1],
            )

            self.add_or_update_edge(
                source_node=source_node,
                target_node=target_node,
                relation=triple.relation,
                article=article,
                evidence=triple.evidence,
            )


# -----------------------------------------------------------------------------
# 备份、保存与报告
# -----------------------------------------------------------------------------

def backup_existing_files() -> List[Path]:
    """把旧的 nodes.json 和 edges.json 复制到 backups 目录。"""

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = utc_timestamp()
    backups: List[Path] = []

    for source_path in (NODES_FILE, EDGES_FILE):
        if not source_path.exists():
            continue

        backup_path = BACKUP_DIR / (
            f"{source_path.stem}_{timestamp}{source_path.suffix}"
        )
        shutil.copy2(source_path, backup_path)
        backups.append(backup_path)
        logging.info(
            "已备份 %s -> %s",
            source_path.name,
            backup_path.relative_to(SCRIPT_DIR),
        )

    return backups


def markdown_escape(value: Any) -> str:
    """简单转义 Markdown 表格中的竖线与换行。"""
    text = normalize_text(value)
    return text.replace("|", r"\|").replace("\n", " ")


def generate_report(
    stats: UpdateStats,
    start_date: str,
    end_date: str,
    articles: Sequence[Article],
    relevant_articles: Sequence[Tuple[Article, ExtractionResult]],
    backups: Sequence[Path],
    run_error: str = "",
) -> str:
    """生成 update_report.md 的完整内容。"""

    lines: List[str] = []

    lines.append("# 中医免疫知识图谱自动更新报告")
    lines.append("")
    lines.append(f"- **运行时间：** {now_local_text()}")
    lines.append(f"- **检索日期范围：** {start_date} 至 {end_date}")
    lines.append(f"- **检索上限：** {MAX_ARTICLES} 篇")
    lines.append(f"- **实际检索到文献：** {stats.retrieved_articles} 篇")
    lines.append(f"- **成功解析文献：** {stats.parsed_articles} 篇")
    lines.append(f"- **有效相关文献：** {stats.valid_articles} 篇")
    lines.append(f"- **判定无关文献：** {stats.irrelevant_articles} 篇")
    lines.append(f"- **处理失败文献：** {stats.failed_articles} 篇")
    lines.append("")

    lines.append("## 数据更新结果")
    lines.append("")
    lines.append(f"- **新增节点：** {stats.new_nodes} 个")
    lines.append(f"- **新增边：** {stats.new_edges} 条")
    lines.append(f"- **已有节点新增证据：** {stats.updated_nodes} 个")
    lines.append(f"- **已有边新增证据：** {stats.updated_edges} 条")
    lines.append(f"- **跳过的不合规三元组：** {stats.skipped_triples} 条")
    lines.append(f"- **潜在冲突：** {len(stats.conflicts)} 项")
    lines.append("")

    lines.append("## 本次有效文献")
    lines.append("")

    if relevant_articles:
        lines.append("| PMID | 标题 | 三元组数 | DOI |")
        lines.append("|---|---|---:|---|")
        for article, extraction in relevant_articles:
            title = markdown_escape(article.title)
            doi = markdown_escape(article.doi or "—")
            lines.append(
                f"| [{article.pmid}]({article.pubmed_url}) "
                f"| {title} | {len(extraction.triples)} | {doi} |"
            )
    else:
        lines.append("本次没有抽取到符合规则的有效文献。")

    lines.append("")
    lines.append("## 潜在知识冲突")
    lines.append("")

    if stats.conflicts:
        lines.append(
            "以下冲突仅用于提示人工复核，脚本不会自动删除任何一方证据。"
        )
        lines.append("")
        lines.append(
            "| 实体1 | 实体2 | 既有关系 | 新关系 | 新证据 PMID | 既有边 ID |"
        )
        lines.append("|---|---|---|---|---|---|")

        for conflict in stats.conflicts:
            lines.append(
                "| {source} | {target} | {existing_relation} | "
                "{new_relation} | {new_pmid} | {existing_edge_id} |".format(
                    source=markdown_escape(conflict.get("source", "")),
                    target=markdown_escape(conflict.get("target", "")),
                    existing_relation=markdown_escape(
                        conflict.get("existing_relation", "")
                    ),
                    new_relation=markdown_escape(
                        conflict.get("new_relation", "")
                    ),
                    new_pmid=markdown_escape(conflict.get("new_pmid", "")),
                    existing_edge_id=markdown_escape(
                        conflict.get("existing_edge_id", "")
                    ),
                )
            )
    else:
        lines.append("未发现“促进—抑制”方向上的潜在冲突。")

    lines.append("")
    lines.append("## 备份文件")
    lines.append("")

    if backups:
        for path in backups:
            try:
                display_path = path.relative_to(SCRIPT_DIR)
            except ValueError:
                display_path = path
            lines.append(f"- `{display_path}`")
    else:
        lines.append("本次没有可备份的旧文件。")

    if stats.failed_article_details:
        lines.append("")
        lines.append("## 文献处理失败明细")
        lines.append("")
        lines.append("| PMID | 标题 | 错误 |")
        lines.append("|---|---|---|")
        for item in stats.failed_article_details:
            lines.append(
                "| {pmid} | {title} | {error} |".format(
                    pmid=markdown_escape(item.get("pmid", "")),
                    title=markdown_escape(item.get("title", "")),
                    error=markdown_escape(item.get("error", "")),
                )
            )

    if stats.warnings:
        lines.append("")
        lines.append("## 数据质量提示")
        lines.append("")
        for warning in unique_preserve_order(stats.warnings):
            lines.append(f"- {warning}")

    if run_error:
        lines.append("")
        lines.append("## 运行错误")
        lines.append("")
        lines.append(f"`{markdown_escape(run_error)}`")

    lines.append("")
    lines.append("## 检索式")
    lines.append("")
    lines.append("```text")
    lines.append(SEARCH_QUERY)
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# 主程序
# -----------------------------------------------------------------------------

def main() -> int:
    """程序入口。返回 0 表示成功，返回非 0 表示失败。"""

    setup_logging()
    logging.info("=" * 72)
    logging.info("开始运行中医免疫知识图谱更新 Agent。")

    stats = UpdateStats()
    start_date, end_date = get_date_range()
    parsed_articles: List[Article] = []
    relevant_articles: List[Tuple[Article, ExtractionResult]] = []
    backups: List[Path] = []

    try:
        # 1. 先验证大模型配置，避免检索完成后才发现没有密钥。
        client = create_llm_client()

        # 2. 读取原图谱数据。
        nodes_root = load_json_or_default(NODES_FILE, [])
        edges_root = load_json_or_default(EDGES_FILE, [])

        nodes, _, nodes_wrapper_key = extract_collection(
            data=nodes_root,
            preferred_key="nodes",
            file_name=NODES_FILE.name,
        )
        edges, _, edges_wrapper_key = extract_collection(
            data=edges_root,
            preferred_key="edges",
            file_name=EDGES_FILE.name,
        )

        graph = KnowledgeGraph(nodes=nodes, edges=edges, stats=stats)

        # 3. 检索 PubMed。
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": (
                    f"{NCBI_TOOL_NAME}/1.0 "
                    f"({NCBI_CONTACT_EMAIL or 'contact-email-not-set'})"
                )
            }
        )

        pmids, start_date, end_date = search_pubmed_ids(session)
        stats.retrieved_articles = len(pmids)

        if pmids:
            # 两次 NCBI 请求之间稍作间隔，避免给公共服务造成过高频率。
            time.sleep(0.4)
            parsed_articles = fetch_pubmed_articles(session, pmids)

        stats.parsed_articles = len(parsed_articles)

        missing_count = len(pmids) - len(parsed_articles)
        if missing_count > 0:
            stats.warnings.append(
                f"有 {missing_count} 个 PMID 未能从 EFetch 结果中解析。"
            )

        # 4. 对每篇文献调用大模型并融合知识。
        for article in parsed_articles:
            try:
                extraction = extract_triples_with_llm(client, article)

                if not extraction.relevant:
                    stats.irrelevant_articles += 1
                    logging.info(
                        "PMID %s 被跳过：%s",
                        article.pmid,
                        extraction.reason,
                    )
                    continue

                stats.valid_articles += 1
                relevant_articles.append((article, extraction))
                graph.merge_article(article, extraction)

                logging.info(
                    "PMID %s 抽取到 %d 条有效三元组。",
                    article.pmid,
                    len(extraction.triples),
                )

            except Exception as exc:
                stats.failed_articles += 1
                stats.failed_article_details.append(
                    {
                        "pmid": article.pmid,
                        "title": article.title,
                        "error": str(exc),
                    }
                )
                logging.exception(
                    "处理 PMID %s 时失败，继续处理下一篇。",
                    article.pmid,
                )

        # 5. 只有在全部融合操作结束后才备份并保存，减少半更新状态。
        backups = backup_existing_files()

        # 保持原来的顶层 JSON 结构。
        if nodes_wrapper_key is not None:
            nodes_root[nodes_wrapper_key] = nodes
            nodes_output = nodes_root
        else:
            nodes_output = nodes

        if edges_wrapper_key is not None:
            edges_root[edges_wrapper_key] = edges
            edges_output = edges_root
        else:
            edges_output = edges

        atomic_write_json(NODES_FILE, nodes_output)
        atomic_write_json(EDGES_FILE, edges_output)

        # 生成报告前删除仅用于内部去重的字段。
        for conflict in stats.conflicts:
            conflict.pop("_dedupe_key", None)

        report = generate_report(
            stats=stats,
            start_date=start_date,
            end_date=end_date,
            articles=parsed_articles,
            relevant_articles=relevant_articles,
            backups=backups,
        )
        atomic_write_text(REPORT_FILE, report)

        logging.info(
            "更新完成：新增节点 %d 个，新增边 %d 条，潜在冲突 %d 项。",
            stats.new_nodes,
            stats.new_edges,
            len(stats.conflicts),
        )
        logging.info("报告已保存到：%s", REPORT_FILE)
        logging.info("日志已保存到：%s", LOG_FILE)
        logging.info("=" * 72)

        return 0

    except Exception as exc:
        logging.exception("程序运行失败。")

        # 即使主流程失败，也尽量输出一份错误报告，方便非技术用户排查。
        try:
            for conflict in stats.conflicts:
                conflict.pop("_dedupe_key", None)

            error_report = generate_report(
                stats=stats,
                start_date=start_date,
                end_date=end_date,
                articles=parsed_articles,
                relevant_articles=relevant_articles,
                backups=backups,
                run_error=str(exc),
            )
            atomic_write_text(REPORT_FILE, error_report)
            logging.info("错误报告已保存到：%s", REPORT_FILE)
        except Exception:
            logging.exception("生成错误报告时也发生异常。")

        return 1


if __name__ == "__main__":
    raise SystemExit(main())
