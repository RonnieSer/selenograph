# Selenograph

Thesis research code by Veronica Serrano for: ""

## Start

Requires Python 3.12.

```sh
git clone https://github.com/RonnieSer/selenograph.git
cd selenograph
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run selenograph.py
```

On Windows, use `python` instead of `python3` and activate with
`.venv\Scripts\Activate.ps1` in PowerShell.

Use **Paint** to label maps, **Gallery** to browse them, and **Progress** to evaluate
predictions.

Run tests with `python -B -m unittest discover -s tests -q`.

## License

Code and documentation: [MIT](LICENSE), copyright 2026 Veronica Serrano.
Data, annotations, trained models, and generated outputs are kept locally and
retain their original terms. See [data sources](data/sources.md).
