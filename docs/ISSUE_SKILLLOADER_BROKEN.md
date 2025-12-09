# Issue: SkillLoader Not Loading Skills

**Status**: ✅ CLOSED (Fixed in #67)
**Priority**: High (blocks Gap 3 from OPENQUESTIONS_SOLUTION.md)
**Created**: 2025-12-08
**Resolved**: 2025-12-08

---

## Summary

The `SkillLoader` class exists and 116 skill files are available in `kosmos-claude-scientific-skills/`, but skills are not being loaded into agent prompts. This breaks Gap 3 (Agent Integration & System Prompts) from the paper implementation.

## Evidence

```python
>>> from kosmos.agents.skill_loader import SkillLoader
>>> loader = SkillLoader()
>>> loader.load_skills_for_task(task_type='research', domain='biology')

Skill not found: seaborn
Skill not found: matplotlib
Skill not found: numpy
Skill not found: pandas
Skill not found: plotly

# Returns: None
```

## Why This Matters

From `archive/implementation/OPENQUESTIONS_SOLUTION.md`:

> **Gap 3: Agent Integration & System Prompts**
>
> "Key Insight: Instead of generic prompts, **inject domain expertise** into agent prompts based on task requirements."

**Without skills**: Agents use generic prompts like "Analyze the data using Python"

**With skills**: Agents get domain expertise:
- API documentation for scientific libraries
- Best practices and common workflows
- Code examples for typical tasks
- Domain-specific knowledge

The paper achieved 79.4% accuracy and 7 validated discoveries. Generic prompts will not reproduce these results.

## Root Cause Analysis

### Problem 1: COMMON_SKILLS references non-existent files

```python
# kosmos/agents/skill_loader.py lines 77-80
COMMON_SKILLS = [
    "pandas", "numpy", "matplotlib", "seaborn", "plotly",
    "scipy", "statsmodels", "jupyter-notebook"
]
```

These are Python library names, not skill files. The actual skill directory contains:

```
kosmos-claude-scientific-skills/scientific-skills/
├── anndata/          # Exists
├── biopython/        # Exists
├── scanpy/           # Exists
├── rdkit/            # Exists
└── ... (116 total)

# pandas, numpy, matplotlib do NOT exist as skill directories
```

### Problem 2: Domain-to-bundle mapping incomplete

```python
# Current bundles in skill_loader.py
SKILL_BUNDLES = {
    "single_cell_analysis": ["scanpy", "anndata", ...],
    "genomics_analysis": ["biopython", "pysam", ...],
    "drug_discovery": ["rdkit", "datamol", ...],
    ...
}
```

But `load_skills_for_task(domain='biology')` doesn't map to these bundles because:
1. No `domain` → `task_type` mapping exists
2. Falls back to COMMON_SKILLS which don't exist

### Problem 3: Integration added but skills return None

In Issue #51 fix, skills were integrated into ResearchDirectorAgent:

```python
# kosmos/agents/research_director.py
def _load_skills(self):
    skill_loader = SkillLoader()
    self.skills = skill_loader.load_skills_for_task(
        task_type="research",
        domain=self.domain
    )
    # Returns None because skills not found
```

## Available Skills (116 total)

```
aeon                    alphafold-database      anndata
arboreto                astropy                 benchling-integration
biomni                  biopython               biorxiv-database
bioservices             cellxgene-census        chembl-database
clinicaltrials-database clinpgx-database        clinvar-database
cobrapy                 cosmic-database         dask
datacommons-client      datamol                 deepchem
duckdb                  ensembl-database        galaxy-tool-util
gdc-client              genepattern             geofetch
gget                    gseapy                  gwas-catalog
harmonizome             hgvs-python             huggingface-datasets
intermine-python        ipywidgets              jupyter-ai
jupyter-notebooks       kegg-database           liftover
llmware                 matchms                 mne-python
mygene                  napari                  ncbi-datasets
ncbi-gene-database      neo4j-python            networkx
nibabel                 nilearn                 nipype
numpy-ml                omim-database           open-targets
opentrons               owlready2               pathways-commons
pegasus                 pharmgkb                phenotype-ontology
polars                  pronto                  pyarrow
pybedtools              pybiomart               pycaret
pydantic                pyensembl               pygwalker
pyliftover              pymol-visualization     pymongo
pyopenms                pypath-omnipath         pyranges
pysam                   pyteomics               pytorch
pytorch-geometric       pytorch-lightning       rdkit
refget                  requests                rnaseq-pipeline
scanpy                  scikit-allel            scikit-bio
scikit-image            scikit-learn            scipy
scvi-tools              seaborn-objects         shap
snakemake               sqlalchemy              stardist
statsmodels             streamlit               tensorflow
torch-geometric         transformers            uniprot-database
vcf-tools               xarray                  zarr
```

## Fix Requirements

### 1. Remove non-existent COMMON_SKILLS

```python
# Change from:
COMMON_SKILLS = ["pandas", "numpy", "matplotlib", ...]

# Change to:
COMMON_SKILLS = ["scipy", "statsmodels", "scikit-learn", "numpy-ml"]
# Or remove COMMON_SKILLS entirely and rely on bundles
```

### 2. Add domain-to-bundle mapping

```python
DOMAIN_BUNDLES = {
    "biology": ["genomics_analysis", "single_cell_analysis", "proteomics"],
    "chemistry": ["drug_discovery"],
    "neuroscience": ["neuroscience"],
    "physics": ["machine_learning"],  # Fallback
    "materials": ["machine_learning"],
}

def load_skills_for_task(self, task_type=None, domain=None, ...):
    if domain and domain in self.DOMAIN_BUNDLES:
        for bundle_name in self.DOMAIN_BUNDLES[domain]:
            if bundle_name in self.SKILL_BUNDLES:
                skills_to_load.update(self.SKILL_BUNDLES[bundle_name])
```

### 3. Verify skill file paths

```python
def load_skill(self, skill_name: str) -> Optional[Dict]:
    skill_path = self.skills_dir / skill_name / "SKILL.md"
    if not skill_path.exists():
        # Try alternative locations
        skill_path = self.skills_dir / skill_name / "references"
        ...
```

### 4. Add fallback for missing skills

```python
def load_skill(self, skill_name: str) -> Optional[Dict]:
    # Don't warn for every missing common library
    if skill_name in self._EXPECTED_MISSING:
        return None
    ...
```

## Files to Modify

| File | Changes |
|------|---------|
| `kosmos/agents/skill_loader.py` | Fix COMMON_SKILLS, add domain mapping, improve error handling |
| `kosmos/agents/research_director.py` | Already integrated, will work once SkillLoader fixed |

## Testing

After fix, verify:

```python
from kosmos.agents.skill_loader import SkillLoader

loader = SkillLoader()

# Should return formatted skills text, not None
skills = loader.load_skills_for_task(domain='biology')
assert skills is not None
assert 'scanpy' in skills or 'biopython' in skills

# Should load specific bundle
skills = loader.load_skills_for_task(task_type='single_cell_analysis')
assert 'scanpy' in skills
```

## Related Documents

- `archive/implementation/OPENQUESTIONS_SOLUTION.md` - Gap 3 specification
- `archive/implementation/OPEN_QUESTIONS.md` - Original gap identification
- `kosmos-claude-scientific-skills/README.md` - Skills repository documentation

## Acceptance Criteria

1. `load_skills_for_task(domain='biology')` returns non-None skill text
2. Skills are injected into HypothesisGeneratorAgent prompts
3. Skills are injected into ExperimentDesignerAgent prompts
4. No "Skill not found" warnings for expected libraries
5. 116 available skills are accessible by name or bundle
