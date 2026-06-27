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


def test_main_generates_report_from_latin1_csv(tmp_path):
    csv_path = tmp_path / "sample_latin1.csv"
    csv_path.write_text("nome,idade\nJosé,20\nAna,19\n", encoding="latin-1")

    archive_path = tmp_path / "sample_latin1.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(csv_path, arcname="sample_latin1.csv")

    output_dir = tmp_path / "extracted"
    report_path = tmp_path / "report_latin1.txt"

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
    assert "José" in report_text or "nome" in report_text


def test_main_uses_results_csv_when_archive_contains_multiple_data_files(tmp_path):
    items_path = tmp_path / "items.csv"
    items_path.write_text(
        "id,descricao\n1,questionario\n2,questionario\n",
        encoding="utf-8",
    )

    results_path = tmp_path / "results.csv"
    rows = [
        "CO_ESCOLA,NO_ESCOLA,SG_UF_ESC,NO_MUNICIPIO_ESC,NU_NOTA_CN,NU_NOTA_CH,NU_NOTA_LC,NU_NOTA_MT,NU_NOTA_REDACAO"
    ]
    rows.extend(
        f"{school_code},Escola {name},PE,RECIFE,700,680,690,710,720"
        for school_code, name in [
            (1001, "A"),
            (1001, "A"),
            (1001, "A"),
            (1001, "A"),
            (1001, "A"),
            (1001, "A"),
            (1001, "A"),
            (1001, "A"),
            (1001, "A"),
            (1001, "A"),
        ]
    )
    rows.append("1002,Escola B,PE,OLINDA,600,620,610,650,640")
    rows.append("1003,Escola C,PE,RECIFE,690,700,710,720,730")
    results_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    archive_path = tmp_path / "multi_data.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(items_path, arcname="DADOS/ITENS_PROVA_2025.csv")
        archive.write(results_path, arcname="DADOS/RESULTADOS_2025.csv")

    output_dir = tmp_path / "extracted"
    analysis_output = tmp_path / "analysis"
    report_path = tmp_path / "report.txt"

    result = main(
        [
            "--source",
            str(archive_path),
            "--output-dir",
            str(output_dir),
            "--report",
            str(report_path),
            "--analysis-output",
            str(analysis_output),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )

    assert result == 0
    assert report_path.exists()
    rankings_path = analysis_output / "recife_school_rankings.csv"
    assert rankings_path.exists()

    rankings = rankings_path.read_text(encoding="utf-8")
    assert "Escola A" in rankings
