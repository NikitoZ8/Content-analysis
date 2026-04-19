import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List
import shutil

import matplotlib.pyplot as plt
import pandas as pd


def read_data() -> pd.DataFrame:
    current_dir = Path(__file__).parent
    file_path = current_dir / "Образ репититора.xlsx"
    return pd.read_excel(file_path)


def compute_numeric_summary(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    return df[columns].describe().round(2)


def compute_frequency_counts(df: pd.DataFrame, column: str) -> pd.DataFrame:
    counts = df[column].value_counts(dropna=False).reset_index()
    counts.columns = [column, "Количество"]
    return counts


def group_by_mean(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    result = (
        df.groupby(group_col)[value_col]
        .mean()
        .round(2)
        .sort_values(ascending=False)
        .reset_index()
    )
    result.columns = [group_col, f"Среднее значение: {value_col}"]
    return result


def group_by_median(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    result = (
        df.groupby(group_col)[value_col]
        .median()
        .round(2)
        .sort_values(ascending=False)
        .reset_index()
    )
    result.columns = [group_col, f"Медианное значение: {value_col}"]
    return result


def tokenize_russian(texts: Iterable[str]) -> List[str]:
    stopwords = {
        'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а',
        'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же',
        'вы', 'за', 'бы', 'по', 'ее', 'мне', 'есть', 'они', 'тут', 'мы',
        'про', 'них', 'их', 'для', 'такой', 'это', 'этот', 'та',
        'те', 'который', 'которые', 'быть', 'или', 'до',
        'ли', 'сам', 'свой', 'тот', 'чтобы', 'тоже'
    }

    pattern = re.compile(r"[\W_\d]+", re.UNICODE)
    tokens_all = []

    for text in texts:
        if not isinstance(text, str):
            continue
        cleaned = pattern.sub(" ", text.lower())
        tokens = cleaned.split()

        for tok in tokens:
            if len(tok) < 3 or tok in stopwords:
                continue
            tokens_all.append(tok)

    return tokens_all


def compute_top_words(df: pd.DataFrame, text_cols: List[str], top_n: int = 20) -> pd.DataFrame:
    combined_texts = []

    for col in text_cols:
        combined_texts.extend(df[col].dropna().astype(str))

    tokens = tokenize_russian(combined_texts)
    counter = Counter(tokens)

    top_words = counter.most_common(top_n)
    return pd.DataFrame(top_words, columns=["Слово", "Частота"])


def prepare_output_paths(base_dir: Path) -> tuple[Path, Path]:
    excel_path = base_dir / "content_analysis_report.xlsx"
    charts_dir = base_dir / "charts"

    if excel_path.exists():
        excel_path.unlink()

    if charts_dir.exists():
        shutil.rmtree(charts_dir)

    charts_dir.mkdir(parents=True, exist_ok=True)

    return excel_path, charts_dir


def save_tables_to_excel(
    numeric_summary: pd.DataFrame,
    platform_counts: pd.DataFrame,
    gender_counts: pd.DataFrame,
    status_counts: pd.DataFrame,
    mode_counts: pd.DataFrame,
    mean_gender: pd.DataFrame,
    mean_platform: pd.DataFrame,
    mean_mode: pd.DataFrame,
    median_gender: pd.DataFrame,
    median_platform: pd.DataFrame,
    median_mode: pd.DataFrame,
    top_words: pd.DataFrame,
    output_path: Path
) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        numeric_summary.to_excel(writer, sheet_name="Статистика")
        platform_counts.to_excel(writer, sheet_name="Платформы", index=False)
        gender_counts.to_excel(writer, sheet_name="Пол", index=False)
        status_counts.to_excel(writer, sheet_name="Соц статус", index=False)
        mode_counts.to_excel(writer, sheet_name="Формат", index=False)
        mean_gender.to_excel(writer, sheet_name="Среднее по полу", index=False)
        mean_platform.to_excel(writer, sheet_name="Среднее по платформе", index=False)
        mean_mode.to_excel(writer, sheet_name="Среднее по формату", index=False)
        median_gender.to_excel(writer, sheet_name="Медиана по полу", index=False)
        median_platform.to_excel(writer, sheet_name="Медиана по платформе", index=False)
        median_mode.to_excel(writer, sheet_name="Медиана по формату", index=False)
        top_words.to_excel(writer, sheet_name="Топ слов", index=False)


def build_charts(
    platform_counts: pd.DataFrame,
    median_platform: pd.DataFrame,
    top_words: pd.DataFrame,
    output_dir: Path
) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(platform_counts["Платформа"], platform_counts["Количество"])
    plt.title("Количество репетиторов по платформам")
    plt.xlabel("Платформа")
    plt.ylabel("Количество")
    plt.tight_layout()
    plt.savefig(output_dir / "platform_counts.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(
        median_platform["Платформа"],
        median_platform["Медианное значение: Стоимость часа"]
    )
    plt.title("Медианная стоимость часа по платформам")
    plt.xlabel("Платформа")
    plt.ylabel("Стоимость часа")
    plt.tight_layout()
    plt.savefig(output_dir / "median_price_by_platform.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(top_words["Слово"], top_words["Частота"])
    plt.title("Топ слов в текстовых полях")
    plt.xlabel("Слово")
    plt.ylabel("Частота")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "top_words.png")
    plt.close()


def main():
    try:
        print("Скрипт запустился")

        df = read_data()
        print("Файл прочитан")

        numeric_cols = ['Возраст', 'Стаж', 'Оценка', 'Количество отзывов', 'Стоимость часа']
        text_cols = ['Достижения', 'Особенности работы']

        numeric_summary = compute_numeric_summary(df, numeric_cols)

        platform_counts = compute_frequency_counts(df, 'Платформа')
        gender_counts = compute_frequency_counts(df, 'Пол')
        status_counts = compute_frequency_counts(df, 'Социальный статус')
        mode_counts = compute_frequency_counts(df, 'Офлайн / онлайн')

        mean_gender = group_by_mean(df, 'Пол', 'Стоимость часа')
        mean_platform = group_by_mean(df, 'Платформа', 'Стоимость часа')
        mean_mode = group_by_mean(df, 'Офлайн / онлайн', 'Стоимость часа')

        median_gender = group_by_median(df, 'Пол', 'Стоимость часа')
        median_platform = group_by_median(df, 'Платформа', 'Стоимость часа')
        median_mode = group_by_median(df, 'Офлайн / онлайн', 'Стоимость часа')

        top_words = compute_top_words(df, text_cols)

        current_dir = Path(__file__).parent
        excel_path, charts_dir = prepare_output_paths(current_dir)

        save_tables_to_excel(
            numeric_summary=numeric_summary,
            platform_counts=platform_counts,
            gender_counts=gender_counts,
            status_counts=status_counts,
            mode_counts=mode_counts,
            mean_gender=mean_gender,
            mean_platform=mean_platform,
            mean_mode=mean_mode,
            median_gender=median_gender,
            median_platform=median_platform,
            median_mode=median_mode,
            top_words=top_words,
            output_path=excel_path
        )

        build_charts(
            platform_counts=platform_counts,
            median_platform=median_platform,
            top_words=top_words,
            output_dir=charts_dir
        )

        print("Готово.")
        print(f"Создан файл: {excel_path.name}")
        print(f"Создана папка: {charts_dir.name}")

    except Exception as e:
        print("Ошибка:", e)

    finally:
        input("\nНажми Enter, чтобы закрыть...")


if __name__ == "__main__":
    main()