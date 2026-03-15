"""LangChain-based autonomous academic literature review agent."""

from datetime import datetime

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from src import scholar_client


CURRENT_YEAR = datetime.now().year


@tool
def search_literature(query: str) -> str:
    """Search academic papers on a given topic from the last 5 years.
    Returns titles, authors, citation counts, and abstracts."""
    results = scholar_client.search_papers(query, year_start=CURRENT_YEAR - 5, limit=20)
    papers = results.get("data", [])
    if not papers:
        return f"'{query}' konusu için makale bulunamadı."

    total = results.get("total", 0)
    lines = [f"Toplam {total} makale bulundu. İlk {len(papers)} tanesi:\n"]
    for i, p in enumerate(papers, 1):
        authors = ", ".join(a.get("name", "") for a in (p.get("authors") or [])[:3])
        if len(p.get("authors") or []) > 3:
            authors += " et al."
        abstract_snippet = (p.get("abstract") or "Özet yok.")[:200]
        lines.append(
            f"{i}. **{p.get('title', 'Başlık yok')}** ({p.get('year', '?')})\n"
            f"   Yazarlar: {authors}\n"
            f"   Atıf: {p.get('citationCount', 0)} | Venue: {p.get('venue', 'N/A')}\n"
            f"   Özet: {abstract_snippet}...\n"
            f"   URL: {p.get('url', 'N/A')}\n"
        )
    return "\n".join(lines)


@tool
def get_most_cited_papers(query: str) -> str:
    """Get the most cited papers on a topic from the last 5 years.
    Returns top 10 papers sorted by citation count."""
    papers = scholar_client.get_most_cited(query, year_start=CURRENT_YEAR - 5, top_n=10)
    if not papers:
        return f"'{query}' konusu için yüksek atıflı makale bulunamadı."

    lines = [f"Son 5 yılda en çok atıf alan ilk {len(papers)} makale:\n"]
    for i, p in enumerate(papers, 1):
        authors = ", ".join(a.get("name", "") for a in (p.get("authors") or [])[:3])
        if len(p.get("authors") or []) > 3:
            authors += " et al."
        lines.append(
            f"{i}. **{p.get('title', 'Başlık yok')}** ({p.get('year', '?')})\n"
            f"   Yazarlar: {authors}\n"
            f"   Atıf Sayısı: {p.get('citationCount', 0)}\n"
            f"   Venue: {p.get('venue', 'N/A')}\n"
            f"   URL: {p.get('url', 'N/A')}\n"
        )
    return "\n".join(lines)


@tool
def find_datasets(query: str) -> str:
    """Find public datasets and benchmarks related to a research topic.
    Searches for papers that introduce or describe datasets."""
    papers = scholar_client.search_datasets(query)
    if not papers:
        return (
            f"'{query}' konusuyla doğrudan ilişkili veri seti makalesi bulunamadı. "
            "Farklı anahtar kelimelerle tekrar deneyebilirsiniz."
        )

    lines = [f"{len(papers)} adet veri seti / benchmark makalesi bulundu:\n"]
    for i, p in enumerate(papers, 1):
        abstract_snippet = (p.get("abstract") or "")[:300]
        pdf_url = ""
        if p.get("openAccessPdf"):
            pdf_url = f"\n   PDF: {p['openAccessPdf'].get('url', '')}"
        lines.append(
            f"{i}. **{p.get('title', 'Başlık yok')}** ({p.get('year', '?')})\n"
            f"   Atıf: {p.get('citationCount', 0)}\n"
            f"   Özet: {abstract_snippet}...{pdf_url}\n"
            f"   URL: {p.get('url', 'N/A')}\n"
        )
    return "\n".join(lines)


@tool
def get_paper_detail(paper_title: str) -> str:
    """Get detailed information about a specific paper by searching its title.
    Returns abstract, references, citations, and TLDR if available."""
    results = scholar_client.search_papers(paper_title, limit=1)
    papers = results.get("data", [])
    if not papers:
        return f"'{paper_title}' başlıklı makale bulunamadı."

    paper_id = papers[0].get("paperId")
    if not paper_id:
        return "Makale ID'si bulunamadı."

    detail = scholar_client.get_paper_details(paper_id)
    if not detail:
        return "Makale detayları alınamadı."

    authors = ", ".join(a.get("name", "") for a in (detail.get("authors") or []))
    tldr = detail.get("tldr", {})
    tldr_text = tldr.get("text", "TLDR mevcut değil.") if tldr else "TLDR mevcut değil."

    ref_count = len(detail.get("references") or [])
    cit_count = len(detail.get("citations") or [])

    top_refs = (detail.get("references") or [])[:5]
    ref_lines = []
    for r in top_refs:
        ref_lines.append(f"  - {r.get('title', 'N/A')} ({r.get('year', '?')})")

    return (
        f"**{detail.get('title', 'N/A')}** ({detail.get('year', '?')})\n"
        f"Yazarlar: {authors}\n"
        f"Venue: {detail.get('venue', 'N/A')}\n"
        f"Atıf: {detail.get('citationCount', 0)}\n"
        f"TLDR: {tldr_text}\n\n"
        f"Özet: {detail.get('abstract', 'Özet yok.')}\n\n"
        f"Referans Sayısı: {ref_count} | Atıf Yapan: {cit_count}\n"
        f"Önemli Referanslar:\n" + "\n".join(ref_lines)
    )


SYSTEM_PROMPT = """Sen akademik literatür taraması yapan uzman bir araştırma asistanısın.
Kullanıcının verdiği araştırma konusu hakkında kapsamlı bir literatür taraması yapmalısın.

Görevlerin:
1. **Literatür Taraması**: Konuyla ilgili son 5 yılın ({start_year}-{current_year}) akademik makalelerini ara ve özetle.
2. **En Çok Atıf Alan Makaleler**: Bu alandaki en etkili ve çok atıf alan çalışmaları bul.
3. **Veri Setleri**: Konuyla ilgili public veri setleri ve benchmark'ları araştır.
4. **Özgün Değer Analizi**: Mevcut literatürdeki boşlukları (gap) belirle ve olası özgün katkı alanlarını öner.

Her adımda ilgili tool'u kullan. Sonuçları Türkçe olarak, akademik bir dilde ve yapılandırılmış şekilde sun.

Yanıtını şu bölümlerle yapılandır:
- **1. Genel Literatür Özeti**: Alanın genel durumu, ana temalar ve trendler
- **2. En Çok Atıf Alan Makaleler**: En etkili çalışmalar ve katkıları
- **3. Public Veri Setleri ve Benchmark'lar**: Kullanılabilir açık veri kaynakları
- **4. Araştırma Boşlukları ve Özgün Değer Önerileri**: Literatürdeki eksiklikler ve potansiyel katkı alanları

Sonunda kısa bir sonuç ve öneri bölümü ekle."""


def create_agent(model_name: str = "gpt-4o-mini", temperature: float = 0.1) -> AgentExecutor:
    """Create and return the literature review agent."""
    llm = ChatOpenAI(model=model_name, temperature=temperature)

    tools = [search_literature, get_most_cited_papers, find_datasets, get_paper_detail]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT.format(
            start_year=CURRENT_YEAR - 5,
            current_year=CURRENT_YEAR,
        )),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=15,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )
