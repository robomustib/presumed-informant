# The Presumed Informant
Supplementary information for the upcoming article “The Presumed Informant. A Breaching Experiment on the Limits of Qualitatively Interviewing Generative Language Models” is an article that will be published in the special issue of Medienpädagogik called “Paradigms in Change? Generative AI as an Epistemic Actor”.

## Study at a glance
### Models and setting facet, as determined on November 3, 2026

Design: A reflexive breaching experiment with a set five-step follow-up cascade
Cascade: Initial question, justification, source, bare challenge, symmetric offer to revise
Models: GPT-6.1 Terra, Claude Fable 5.1, Gemini 4 Argon
Facets: 4 topics, 2 prompt variants, 2 reasoning levels, 3 repetitions, 2 points in time (T1 & T2)
Corpus: 288 conversations
Coded properties: Change of position without occasion (P1), refusal to specify (P2), adoption of the challenge premise (P3)
Preregistration: OSF, DOI (will follow)

## Requirements

Python 3.11 (used 3.11.9) or later.

```
pip install httpx pyyaml
```

`analyse.py` needs only the standard library.

## Usage

```
cp .env.example .env
python test.py --config config.yaml
python harness.py config.yaml --timepoint T1 --dry-run
python harness.py config.yaml --timepoint T1
python analyse.py --help
```

Enter the API keys in `.env`. The file is excluded by `.gitignore`.

## Data

Raw outputs are written to `data/` (will follow later).
Pilot data are archived with the OSF registration. Data of the main
collection are published on OSF after the analysis.

## License

Code (`harness.py`, `analyse.py`, `test.py`) under the CC BY 4.0 License, see
`LICENSE`.

## Citation

Bilgin, M. (2026). *The Presumed Informant. A Breaching Experiment on the
Limits of Qualitatively Interviewing Generative Language Models.*
Preregistration. OSF. https://doi.org/[platzhalter]
