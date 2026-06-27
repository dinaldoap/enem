import zipfile

from enem.__main__ import main


def test_main_generates_report_from_zip(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "nome,idade,nota\nAna,20,720.5\nBruno,19,680.0\n", encoding="utf-8"
    )

    archive_path = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(csv_path, arcname="sample.csv")

    output_dir = tmp_path / "extracted"
    report_path = tmp_path / "report.txt"

    result = main(
        [
            "--source",
            str(archive_path),
            "--output-dir",
            str(output_dir),
            "--report",
            str(report_path),
        ]
    )

    assert result == 0
    assert report_path.exists()

    report_text = report_path.read_text(encoding="utf-8")
    assert "EDA Report" in report_text
    assert "Rows: 2" in report_text
    assert "Columns: 3" in report_text
    assert "nota" in report_text
