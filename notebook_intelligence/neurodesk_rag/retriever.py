#!/usr/bin/env python3
"""
Neurodesk Multi-Step RAG Retriever (LangGraph)

A four-node LangGraph pipeline that narrows the vector-store search
progressively before returning chunks to a code-writing agent:

  classify_query
       |
  select_notebooks
       |
  retrieve_chunks
       |
  rank_and_return

Each node enriches the shared RetrieverState so the final output
is a focused list of Document chunks relevant to the user query.
"""

from __future__ import annotations

import os
from typing import Annotated, List, Optional
from operator import add
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

# Global variable to store the OpenAI API key (set by retrieve function)
_openai_api_key: Optional[str] = None


def _get_openai_api_key() -> Optional[str]:
    """Get the OpenAI API key from global variable or environment."""
    global _openai_api_key
    if _openai_api_key:
        return _openai_api_key
    return os.environ.get("OPENAI_API_KEY")

# ---------------------------------------------------------------------------
# Knowledge base - hard-coded context the LLM uses for routing decisions
# ---------------------------------------------------------------------------

CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "diffusion_imaging": (
        "Diffusion imaging techniques analyse water diffusion in brain tissue "
        "to study white-matter structure and connectivity. Covers preprocessing "
        "(denoising, distortion correction, Gibbs ringing removal), diffusion "
        "tensor fitting, fractional anisotropy (FA) maps, TBSS voxel-wise "
        "analysis, Constrained Spherical Deconvolution (CSD), Fibre Orientation "
        "Distributions (FODs), multi-shell multi-tissue (MSMT) modelling, and "
        "probabilistic/anatomically-constrained tractography."
    ),
    "structural_imaging": (
        "High-resolution anatomical brain and spinal-cord imaging. Covers brain "
        "extraction (FSL BET and competing tools), tissue segmentation (GM/WM/CSF), "
        "cortical and subcortical surface reconstruction with FreeSurfer, "
        "quantitative susceptibility mapping (QSM / QSMxT), and spinal-cord "
        "analysis with the Spinal Cord Toolbox (SCT)."
    ),
    "spectroscopy": (
        "Magnetic Resonance Spectroscopy (MRS) quantifies brain metabolite "
        "concentrations from localised voxels. Covers spectral acquisition, "
        "preprocessing, basis-function modelling, automated spectral fitting with "
        "LCModel and Osprey, anatomical co-registration with tissue segmentation, "
        "and metabolite concentration estimation (NAA, GABA, glutamate, etc.)."
    ),
}

NOTEBOOK_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "diffusion_imaging": {
        "Diffusion_TBSS_Demo.ipynb": (
            "FSL TBSS workflow: distortion correction with topup, tensor fitting "
            "with dtifit, spatial normalisation to MNI space for voxel-wise "
            "cross-subject analysis of diffusion metrics."
        ),
        "MRtrix_1.ipynb": (
            "MRtrix preprocessing part 1: denoising, Gibbs ringing removal, brain "
            "mask generation, motion and distortion correction (topup + eddy), bias "
            "field correction."
        ),
        "MRtrix_2.ipynb": (
            "MRtrix part 2: Constrained Spherical Deconvolution (CSD) for "
            "multi-tissue FOD estimation, FOD normalisation, and tissue boundary "
            "generation for tractography seeding."
        ),
        "MRtrix_3.ipynb": (
            "MRtrix part 3: co-registration of diffusion and anatomical images, "
            "grey-white matter boundary creation, probabilistic streamline generation "
            "with anatomically constrained tracking (ACT)."
        ),
        "mrtrix3tissue.ipynb": (
            "MRtrix3Tissue: 3-tissue response function estimation, upsampling, "
            "3-tissue CSD, bias field and intensity normalisation for multi-shell "
            "diffusion data."
        ),
    },
    "structural_imaging": {
        "FSL_course_bet.ipynb": (
            "FSL Brain Extraction Tool (BET): removing non-brain tissue from "
            "structural MRI with various parameter settings and mask refinement."
        ),
        "brain_extraction_different_tools.ipynb": (
            "Comparative evaluation of multiple brain-extraction algorithms across "
            "software packages (FSL, ANTs, HD-BET, etc.)."
        ),
        "freesurfer.ipynb": (
            "Full FreeSurfer pipeline for cortical and subcortical segmentation, "
            "surface reconstruction, and anatomical parcellation."
        ),
        "qsmxt.ipynb": (
            "Quantitative Susceptibility Mapping using QSMxT: computing susceptibility "
            "maps from structural (GRE) MRI data."
        ),
        "sct_toolbox.ipynb": (
            "Spinal Cord Toolbox (SCT): spinal-cord segmentation, template "
            "registration, and diffusion/structural metric extraction."
        ),
    },
    "spectroscopy": {
        "lcmodel.ipynb": (
            "LCModel workflow for rat hippocampus MR spectra: Varian FID -> RAW "
            "conversion, parameter extraction, automatic spectral fitting, and "
            "output visualisation with basis-set modelling."
        ),
        "osprey.ipynb": (
            "Osprey pipeline for human single-voxel MRS: data loading, preprocessing, "
            "LCModel spectral fitting, anatomical co-registration with tissue "
            "segmentation, and metabolite concentration quantification."
        ),
    },
}

# All valid categories and notebook names (used for structured LLM output)
ALL_CATEGORIES: list[str] = list(CATEGORY_DESCRIPTIONS.keys())
ALL_NOTEBOOKS: dict[str, list[str]] = {
    cat: list(nbs.keys()) for cat, nbs in NOTEBOOK_DESCRIPTIONS.items()
}

# ---------------------------------------------------------------------------
# Shared state schema
# ---------------------------------------------------------------------------


class RetrieverState(TypedDict):
    # Input
    query: str

    # Step 1 - category classification
    selected_categories: List[str]        # e.g. ["diffusion_imaging"]
    category_reasoning: str

    # Step 2 - notebook selection
    selected_notebooks: List[str]         # bare filenames
    notebook_reasoning: str

    # Step 3 - raw retrieved chunks
    raw_chunks: Annotated[List[Document], add]

    # Step 4 - final ranked chunks returned to the caller
    final_chunks: List[Document]


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def _llm() -> ChatOpenAI:
    api_key = _get_openai_api_key()
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)


# ---------------------------------------------------------------------------
# Node 1 - classify_query
# ---------------------------------------------------------------------------

_CLASSIFY_SYSTEM = """\
You are a neuroimaging assistant.  Given a user query, decide which of the
following Neurodesk notebook categories are relevant.

Available categories:
{category_block}

Rules:
- Return only category names from the list above, comma-separated.
- Return ALL categories that are relevant; return more than one if the query
  spans multiple modalities.
- If the query is generic (e.g. "show me examples"), return all three.
- After the category names, add a short explanation on a new line starting
  with "Reason:".

Example output:
diffusion_imaging
Reason: The query asks about tractography which is a diffusion technique.
"""


def classify_query(state: RetrieverState) -> dict:
    """Node 1: map the free-text query to one or more categories."""
    category_block = "\n".join(
        f'- "{cat}": {desc}' for cat, desc in CATEGORY_DESCRIPTIONS.items()
    )
    prompt = _CLASSIFY_SYSTEM.format(category_block=category_block)

    response = _llm().invoke(
        [{"role": "system", "content": prompt},
         {"role": "user", "content": state["query"]}]
    )
    text = response.content.strip()

    # Parse categories from first line(s) before "Reason:"
    lines = text.split("\n")
    cat_line = ""
    reasoning = ""
    for i, line in enumerate(lines):
        if line.lower().startswith("reason:"):
            reasoning = line[7:].strip()
            break
        cat_line += " " + line

    # Normalise: split on comma / whitespace, keep only valid category names
    tokens = [t.strip().strip('"').strip("'") for t in cat_line.replace(",", " ").split()]
    selected = [t for t in tokens if t in ALL_CATEGORIES]
    if not selected:
        selected = ALL_CATEGORIES  # fallback: search everything

    return {
        "selected_categories": selected,
        "category_reasoning": reasoning,
    }


# ---------------------------------------------------------------------------
# Node 2 - select_notebooks
# ---------------------------------------------------------------------------

_SELECT_SYSTEM = """\
You are a neuroimaging assistant.  Given a user query and a list of notebook
descriptions, choose which specific notebooks are most likely to contain
useful code or explanations for answering the query.

User query: {query}

Available notebooks (category -> filename: description):
{notebook_block}

Rules:
- Return only the bare filenames (e.g. "MRtrix_1.ipynb"), one per line.
- Include every notebook that is relevant; omit those that are clearly unrelated.
- After the filenames, add a short explanation on a new line starting with
  "Reason:".
"""


def select_notebooks(state: RetrieverState) -> dict:
    """Node 2: choose the specific notebooks to retrieve from."""
    # Build description block only for the selected categories
    lines: list[str] = []
    for cat in state["selected_categories"]:
        for nb_name, nb_desc in NOTEBOOK_DESCRIPTIONS.get(cat, {}).items():
            lines.append(f'  [{cat}] {nb_name}: {nb_desc}')
    notebook_block = "\n".join(lines)

    prompt = _SELECT_SYSTEM.format(
        query=state["query"],
        notebook_block=notebook_block,
    )

    response = _llm().invoke(
        [{"role": "system", "content": prompt}]
    )
    text = response.content.strip()

    # Parse filenames before "Reason:"
    selected: list[str] = []
    reasoning = ""
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith("reason:"):
            reasoning = stripped[7:].strip()
            break
        if stripped.endswith(".ipynb"):
            selected.append(stripped)

    # Fallback: use all notebooks in selected categories
    if not selected:
        for cat in state["selected_categories"]:
            selected.extend(ALL_NOTEBOOKS.get(cat, []))

    return {
        "selected_notebooks": selected,
        "notebook_reasoning": reasoning,
    }


# ---------------------------------------------------------------------------
# Node 3 - retrieve_chunks
# ---------------------------------------------------------------------------

# Path to the vector database (relative to this module's location)
_MODULE_DIR = Path(__file__).parent
CHROMA_DIR = str(_MODULE_DIR / "neurodesk_db")
_TOP_K = 20   # fetch more candidates, re-rank in node 4


def retrieve_chunks(state: RetrieverState) -> dict:
    """Node 3: similarity search filtered by the chosen notebooks."""
    api_key = _get_openai_api_key()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)
    store = Chroma(
        collection_name="neurodesk_notebooks",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    # Build Chroma $in filter for the selected notebook names
    notebook_filter: dict = {
        "notebook_name": {"$in": state["selected_notebooks"]}
    }

    results: List[Document] = store.similarity_search(
        state["query"],
        k=_TOP_K,
        filter=notebook_filter,
    )

    return {"raw_chunks": results}


# ---------------------------------------------------------------------------
# Node 4 - rank_and_return
# ---------------------------------------------------------------------------

_RANK_SYSTEM = """\
You are a neuroimaging assistant helping a code-writing agent.
Given a user query and a numbered list of retrieved text chunks, select and
rank the chunks that are most useful for writing code to answer the query.

User query: {query}

Chunks:
{chunks_block}

Rules:
- Return ONLY the chunk numbers (1-based), best first, one per line.
- Include at most 8 chunks; exclude chunks that are only tangentially relevant.
- After the numbers add a line starting with "Reason:" explaining briefly.
"""


def rank_and_return(state: RetrieverState) -> dict:
    """Node 4: LLM re-ranks the raw chunks and returns the top subset."""
    chunks = state["raw_chunks"]
    if not chunks:
        return {"final_chunks": []}

    chunks_block = "\n\n".join(
        f"[{i + 1}] (notebook: {c.metadata.get('notebook_name', '?')}, "
        f"category: {c.metadata.get('category', '?')})\n{c.page_content[:400]}"
        for i, c in enumerate(chunks)
    )

    prompt = _RANK_SYSTEM.format(
        query=state["query"],
        chunks_block=chunks_block,
    )

    response = _llm().invoke(
        [{"role": "system", "content": prompt}]
    )
    text = response.content.strip()

    # Parse selected indices
    selected_indices: list[int] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith("reason:"):
            break
        try:
            idx = int(stripped) - 1  # convert to 0-based
            if 0 <= idx < len(chunks):
                selected_indices.append(idx)
        except ValueError:
            pass

    final = [chunks[i] for i in selected_indices] if selected_indices else chunks[:8]
    return {"final_chunks": final}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    workflow = StateGraph(RetrieverState)

    workflow.add_node("classify_query",    classify_query)
    workflow.add_node("select_notebooks",  select_notebooks)
    workflow.add_node("retrieve_chunks",   retrieve_chunks)
    workflow.add_node("rank_and_return",   rank_and_return)

    workflow.set_entry_point("classify_query")
    workflow.add_edge("classify_query",   "select_notebooks")
    workflow.add_edge("select_notebooks", "retrieve_chunks")
    workflow.add_edge("retrieve_chunks",  "rank_and_return")
    workflow.add_edge("rank_and_return",  END)

    return workflow.compile()


# Public handle used by other agents
retrieve_graph = build_graph()


# ---------------------------------------------------------------------------
# Convenience function for direct programmatic use
# ---------------------------------------------------------------------------

def retrieve(query: str, api_key: Optional[str] = None) -> List[Document]:
    """
    Run the full retrieval pipeline for *query* and return the final chunks.

    Args:
        query: Natural-language question or task description from the user.
        api_key: Optional OpenAI API key. If not provided, uses OPENAI_API_KEY env var.

    Returns:
        List of Document objects with page_content and metadata
        (category, notebook_name, notebook_url, cell_type).
    """
    global _openai_api_key
    if api_key:
        _openai_api_key = api_key

    initial_state: RetrieverState = {
        "query": query,
        "selected_categories": [],
        "category_reasoning": "",
        "selected_notebooks": [],
        "notebook_reasoning": "",
        "raw_chunks": [],
        "final_chunks": [],
    }
    result = retrieve_graph.invoke(initial_state)
    return result["final_chunks"]


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "How do I run tractography on diffusion MRI data using MRtrix?"
    )

    print(f"\nQuery: {query}\n{'─' * 60}")

    initial: RetrieverState = {
        "query": query,
        "selected_categories": [],
        "category_reasoning": "",
        "selected_notebooks": [],
        "notebook_reasoning": "",
        "raw_chunks": [],
        "final_chunks": [],
    }

    for step in retrieve_graph.stream(initial, stream_mode="updates"):
        node_name = next(iter(step))
        node_out  = step[node_name]

        if node_name == "classify_query":
            print(f"[1] classify_query  -> {node_out['selected_categories']}")
            print(f"    reason: {node_out['category_reasoning']}")

        elif node_name == "select_notebooks":
            print(f"[2] select_notebooks -> {node_out['selected_notebooks']}")
            print(f"    reason: {node_out['notebook_reasoning']}")

        elif node_name == "retrieve_chunks":
            print(f"[3] retrieve_chunks  -> {len(node_out['raw_chunks'])} chunks")

        elif node_name == "rank_and_return":
            final = node_out["final_chunks"]
            print(f"[4] rank_and_return  -> {len(final)} chunks selected\n")
            for i, doc in enumerate(final, 1):
                print(f"  Chunk {i} [{doc.metadata.get('notebook_name')}]")
                print(f"  {doc.page_content[:200].strip()}")
                print()
