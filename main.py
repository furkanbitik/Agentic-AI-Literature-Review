"""Academic Literature Review Agent - Main entry point."""

import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.agent import create_agent

load_dotenv()


def save_report(topic: str, report: str) -> str:
    """Save the literature review report to a markdown file."""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = "".join(c if c.isalnum() or c in " -_" else "" for c in topic)[:50].strip()
    filename = f"{timestamp}_{safe_topic.replace(' ', '_')}.md"
    filepath = output_dir / filename

    content = f"# Literatür Tarama Raporu: {topic}\n\n"
    content += f"**Tarih:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    content += f"---\n\n{report}\n"

    filepath.write_text(content, encoding="utf-8")
    return str(filepath)


def run_review(topic: str, model: str = "gpt-4o-mini") -> str:
    """Run a literature review on the given topic."""
    if not os.getenv("OPENAI_API_KEY"):
        print("HATA: OPENAI_API_KEY ortam değişkeni ayarlanmamış.")
        print(".env dosyasına OPENAI_API_KEY=sk-... şeklinde ekleyin.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Akademik Literatür Tarama Ajanı")
    print(f"  Konu: {topic}")
    print(f"  Model: {model}")
    print(f"{'='*60}\n")

    agent = create_agent(model_name=model)

    prompt = (
        f"'{topic}' konusu hakkında kapsamlı bir akademik literatür taraması yap. "
        f"Son 5 yılın ({datetime.now().year - 5}-{datetime.now().year}) literatürünü tara. "
        "Sırasıyla şunları yap:\n"
        "1. Konuyla ilgili genel literatür araması yap\n"
        "2. En çok atıf alan makaleleri bul\n"
        "3. İlgili public veri setlerini araştır\n"
        "4. Tüm bulgulara dayanarak araştırma boşluklarını ve özgün değer önerilerini belirle\n\n"
        "Her bölümü detaylı şekilde Türkçe olarak raporla."
    )

    result = agent.invoke({"input": prompt})
    report = result["output"]

    filepath = save_report(topic, report)
    print(f"\nRapor kaydedildi: {filepath}")

    return report


def interactive_mode():
    """Run the agent in interactive mode."""
    print("\n" + "=" * 60)
    print("  Akademik Literatür Tarama Ajanı - İnteraktif Mod")
    print("  Çıkmak için 'q' veya 'çıkış' yazın")
    print("=" * 60)

    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    while True:
        print()
        topic = input("Araştırma konusu girin: ").strip()
        if topic.lower() in ("q", "quit", "çıkış", "exit"):
            print("Güle güle!")
            break
        if not topic:
            print("Lütfen bir konu girin.")
            continue

        try:
            run_review(topic, model=model)
        except KeyboardInterrupt:
            print("\nİşlem iptal edildi.")
        except Exception as e:
            print(f"\nHata oluştu: {e}")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
        run_review(topic)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
