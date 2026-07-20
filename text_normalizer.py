import re
import regex
import inflect
from wetext import Normalizer

chinese_char_pattern = re.compile(r"[\u4e00-\u9fff]+")


def contains_chinese(text):
    return bool(chinese_char_pattern.search(text))


def replace_corner_mark(text):
    text = text.replace("²", "平方")
    text = text.replace("³", "立方")
    text = text.replace("√", "根号")
    text = text.replace("≈", "约等于")
    text = text.replace("<", "小于")
    return text


def remove_bracket(text):
    text = text.replace("（", " ").replace("）", " ")
    text = text.replace("【", " ").replace("】", " ")
    text = text.replace("`", "").replace("`", "")
    text = text.replace("——", " ")
    return text


def spell_out_number(text, inflect_parser):
    new_text = []
    st = None
    for i, c in enumerate(text):
        if not c.isdigit():
            if st is not None:
                num_str = inflect_parser.number_to_words(text[st:i])
                new_text.append(num_str)
                st = None
            new_text.append(c)
        else:
            if st is None:
                st = i
    if st is not None and st < len(text):
        num_str = inflect_parser.number_to_words(text[st:])
        new_text.append(num_str)
    return "".join(new_text)


def replace_blank(text):
    out_str = []
    for i, c in enumerate(text):
        if c == " ":
            if (
                i + 1 < len(text) and i - 1 >= 0
                and text[i + 1].isascii() and text[i + 1] != " "
                and text[i - 1].isascii() and text[i - 1] != " "
            ):
                out_str.append(c)
        else:
            out_str.append(c)
    return "".join(out_str)


def clean_markdown(md_text):
    md_text = re.sub(r"```.*?```", "", md_text, flags=re.DOTALL)
    md_text = re.sub(r"`[^`]*`", "", md_text)
    md_text = re.sub(r"!\[[^\]]*\]\([^\)]+\)", "", md_text)
    md_text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", md_text)
    md_text = re.sub(r"^(\s*)-\s+", r"\1", md_text, flags=re.MULTILINE)
    md_text = re.sub(r"<[^>]+>", "", md_text)
    md_text = re.sub(r"^#{1,6}\s*", "", md_text, flags=re.MULTILINE)
    md_text = re.sub(r"\n\s*\n", "\n", md_text)
    md_text = md_text.strip()
    return md_text


def clean_text(text):
    text = clean_markdown(text)
    text = regex.compile(r"\p{Emoji_Presentation}|\p{Emoji}\uFE0F", flags=regex.UNICODE).sub("", text)
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = text.replace("“", '"').replace("”", '"')
    return text


class TextNormalizer:
    def __init__(self):
        self.zh_tn_model = Normalizer(lang="zh", operator="tn", remove_erhua=True)
        self.en_tn_model = Normalizer(lang="en", operator="tn")
        self.inflect_parser = inflect.engine()

    def normalize(self, text):
        lang = "zh" if contains_chinese(text) else "en"
        text = clean_text(text)
        if lang == "zh":
            text = text.replace("=", "等于")
            if re.search(r"([\d$%^*_+≥≤≠×÷?=])", text):
                text = re.sub(r"(?<=[a-zA-Z0-9])-(?=\d)", " - ", text)
            text = self.zh_tn_model.normalize(text)
            text = replace_blank(text)
            text = replace_corner_mark(text)
            text = remove_bracket(text)
        else:
            text = self.en_tn_model.normalize(text)
            text = spell_out_number(text, self.inflect_parser)
        return text
