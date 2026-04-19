import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd


def read_data() -> pd.DataFrame:
    current_dir = Path(__file__).parent
    file_path = current_dir / "Образ репититора.xlsx"
    return pd.read_excel(file_path)


def prepare_output_paths(base_dir: Path) -> tuple[Path, Path]:
    excel_path = base_dir / "content_analysis_report.xlsx"
    charts_dir = base_dir / "charts"

    if excel_path.exists():
        excel_path.unlink()

    if charts_dir.exists():
        shutil.rmtree(charts_dir)

    charts_dir.mkdir(parents=True, exist_ok=True)
    return excel_path, charts_dir


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


def crosstab_median(df: pd.DataFrame, row_col: str, col_col: str, value_col: str) -> pd.DataFrame:
    result = pd.pivot_table(
        df,
        values=value_col,
        index=row_col,
        columns=col_col,
        aggfunc="median"
    ).round(2)
    return result


def crosstab_mean(df: pd.DataFrame, row_col: str, col_col: str, value_col: str) -> pd.DataFrame:
    result = pd.pivot_table(
        df,
        values=value_col,
        index=row_col,
        columns=col_col,
        aggfunc="mean"
    ).round(2)
    return result


def add_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Возрастная группа"] = pd.cut(
        df["Возраст"],
        bins=[0, 25, 35, 45, 55, 100],
        labels=["до 25", "26–35", "36–45", "46–55", "56+"]
    )
    return df


def add_experience_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Группа стажа"] = pd.cut(
        df["Стаж"],
        bins=[0, 3, 7, 15, 100],
        labels=["1–3", "4–7", "8–15", "16+"]
    )
    return df


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


def compute_content_categories(df: pd.DataFrame) -> pd.DataFrame:
    category_dict = {
        "Квалификация и образование": [
            "окончил", "университет", "мгу", "мфти", "спбгу",
            "кандидат", "магистрант", "диплом", "образование"
        ],
        "Опыт": [
            "опыт", "стаж", "лет"
        ],
        "Достижения": [
            "олимпиад", "победитель", "призер", "всероссийских", "награда"
        ],
        "Экзамены": [
            "егэ", "огэ", "экзамен"
        ],
        "Индивидуальный подход": [
            "личный", "индивидуальный", "адаптация", "программе"
        ],
        "Маркетинговые ходы": [
            "бесплатно", "первое", "занятие"
        ],
        "Онлайн-формат": [
            "онлайн", "дистанционно"
        ],
        "Профессиональное развитие": [
            "сертификаты", "повышении", "квалификации"
        ]
    }

    text = (
        df["Достижения"].fillna("").astype(str) + " " +
        df["Особенности работы"].fillna("").astype(str)
    ).str.lower()

    rows = []
    for category, keywords in category_dict.items():
        count = text.apply(lambda x: any(word in x for word in keywords)).sum()
        rows.append({"Категория": category, "Количество упоминаний": int(count)})

    result = pd.DataFrame(rows).sort_values("Количество упоминаний", ascending=False)
    return result


def save_tables_to_excel(tables: dict, output_path: Path) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, table in tables.items():
            table.to_excel(writer, sheet_name=sheet_name[:31], index=True if table.index.name else False)


def build_basic_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str, path: Path, rotate=False) -> None:
    _, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df[x_col].astype(str), df[y_col])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if "Количество" in ylabel or "Частота" in ylabel:
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
    if rotate:
        plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def build_grouped_bar_from_pivot(pivot_df: pd.DataFrame, title: str, xlabel: str, ylabel: str, path: Path, integer_y_axis: bool = False) -> None:
    ax = pivot_df.plot(kind="bar", figsize=(10, 6))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if integer_y_axis:
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    try:
        print("Скрипт запустился")

        df = read_data()
        df = add_age_groups(df)
        df = add_experience_groups(df)

        print("Файл прочитан")

        current_dir = Path(__file__).parent
        excel_path, charts_dir = prepare_output_paths(current_dir)

        numeric_cols = ['Возраст', 'Стаж', 'Оценка', 'Количество отзывов', 'Стоимость часа']
        text_cols = ['Достижения', 'Особенности работы']

        numeric_summary = compute_numeric_summary(df, numeric_cols)

        platform_counts = compute_frequency_counts(df, 'Платформа')
        gender_counts = compute_frequency_counts(df, 'Пол')
        mode_counts = compute_frequency_counts(df, 'Офлайн / онлайн')

        median_platform = group_by_median(df, 'Платформа', 'Стоимость часа')
        median_gender = group_by_median(df, 'Пол', 'Стоимость часа')
        median_mode = group_by_median(df, 'Офлайн / онлайн', 'Стоимость часа')
        median_age_group = group_by_median(df, 'Возрастная группа', 'Стоимость часа')
        median_experience_group = group_by_median(df, 'Группа стажа', 'Стоимость часа')

        mean_rating_by_platform = group_by_mean(df, 'Платформа', 'Оценка')
        mean_reviews_by_platform = group_by_mean(df, 'Платформа', 'Количество отзывов')

        gender_platform_price = crosstab_median(df, 'Пол', 'Платформа', 'Стоимость часа')
        mode_platform_price = crosstab_median(df, 'Офлайн / онлайн', 'Платформа', 'Стоимость часа')
        gender_mode_price = crosstab_median(df, 'Пол', 'Офлайн / онлайн', 'Стоимость часа')
        platform_mode_distribution = pd.crosstab(df['Платформа'], df['Офлайн / онлайн'])

        top_words = compute_top_words(df, text_cols)
        content_categories = compute_content_categories(df)

        tables = {
            "Статистика": numeric_summary,
            "Платформы": platform_counts,
            "Пол": gender_counts,
            "Формат": mode_counts,
            "Медиана_платформа": median_platform,
            "Медиана_пол": median_gender,
            "Медиана_формат": median_mode,
            "Медиана_возраст": median_age_group,
            "Медиана_стаж": median_experience_group,
            "Рейтинг_платформа": mean_rating_by_platform,
            "Отзывы_платформа": mean_reviews_by_platform,
            "Пол_Платформа_Цена": gender_platform_price,
            "Формат_Платформа_Цена": mode_platform_price,
            "Пол_Формат_Цена": gender_mode_price,
            "Платформа_Формат_частоты": platform_mode_distribution,
            "Топ_слов": top_words,
            "Категории_контента": content_categories
        }

        save_tables_to_excel(tables, excel_path)

        build_basic_bar_chart(
            platform_counts,
            "Платформа",
            "Количество",
            "Количество репетиторов по платформам",
            "Платформа",
            "Количество",
            charts_dir / "platform_counts.png"
        )

        build_basic_bar_chart(
            median_platform,
            "Платформа",
            "Медианное значение: Стоимость часа",
            "Медианная стоимость часа по платформам",
            "Платформа",
            "Стоимость часа",
            charts_dir / "median_price_by_platform.png"
        )

        build_basic_bar_chart(
            median_gender,
            "Пол",
            "Медианное значение: Стоимость часа",
            "Медианная стоимость часа по полу",
            "Пол",
            "Стоимость часа",
            charts_dir / "median_price_by_gender.png"
        )

        build_basic_bar_chart(
            median_mode,
            "Офлайн / онлайн",
            "Медианное значение: Стоимость часа",
            "Медианная стоимость часа по формату занятий",
            "Формат занятий",
            "Стоимость часа",
            charts_dir / "median_price_by_mode.png"
        )

        build_basic_bar_chart(
            median_age_group,
            "Возрастная группа",
            "Медианное значение: Стоимость часа",
            "Медианная стоимость часа по возрастным группам",
            "Возрастная группа",
            "Стоимость часа",
            charts_dir / "median_price_by_age_group.png"
        )

        build_basic_bar_chart(
            median_experience_group,
            "Группа стажа",
            "Медианное значение: Стоимость часа",
            "Медианная стоимость часа по группам стажа",
            "Группа стажа",
            "Стоимость часа",
            charts_dir / "median_price_by_experience_group.png"
        )

        build_basic_bar_chart(
            top_words,
            "Слово",
            "Частота",
            "Топ слов в текстовых полях",
            "Слово",
            "Частота",
            charts_dir / "top_words.png",
            rotate=True
        )

        build_basic_bar_chart(
            content_categories,
            "Категория",
            "Количество упоминаний",
            "Категории самопрезентации репетиторов",
            "Категория",
            "Количество упоминаний",
            charts_dir / "content_categories.png",
            rotate=True
        )

        build_grouped_bar_from_pivot(
            gender_platform_price,
            "Медианная стоимость часа: пол × платформа",
            "Пол",
            "Стоимость часа",
            charts_dir / "gender_platform_price.png"
        )

        build_grouped_bar_from_pivot(
            mode_platform_price,
            "Медианная стоимость часа: формат × платформа",
            "Формат занятий",
            "Стоимость часа",
            charts_dir / "mode_platform_price.png"
        )

        build_grouped_bar_from_pivot(
            gender_mode_price,
            "Медианная стоимость часа: пол × формат занятий",
            "Пол",
            "Стоимость часа",
            charts_dir / "gender_mode_price.png"
        )

        build_grouped_bar_from_pivot(
            platform_mode_distribution,
            "Распределение формата занятий по платформам",
            "Платформа",
            "Количество",
            charts_dir / "platform_mode_distribution.png",
            integer_y_axis=True
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
