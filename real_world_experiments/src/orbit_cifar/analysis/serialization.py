import json

import pandas as pd

def _latex_escape(value: object) -> str:
    if not pd.api.types.is_scalar(value):
        value = json.dumps(value, ensure_ascii=False, default=str)
    text = "" if pd.isna(value) else (f"{value:.6g}" if isinstance(value, float) else str(value))
    replacements = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#",
                    "$": r"\$", "{": r"\{", "}": r"\}", "^": r"\textasciicircum{}", "~": r"\textasciitilde{}"}
    return "".join(replacements.get(character, character) for character in text)


def _to_latex(frame: pd.DataFrame) -> str:
    """Render a dependency-free tabular environment for paper appendices."""
    columns = [str(column) for column in frame.columns]
    if not columns:
        return "% No available data\n"
    alignment = "l" + "r" * (len(columns) - 1)
    lines = [f"\\begin{{tabular}}{{{alignment}}}", "\\hline", " & ".join(_latex_escape(column) for column in columns) + r" \\", "\\hline"]
    for row in frame.itertuples(index=False, name=None):
        lines.append(" & ".join(_latex_escape(value) for value in row) + r" \\")
    lines.extend(["\\hline", "\\end{tabular}", ""])
    return "\n".join(lines)


