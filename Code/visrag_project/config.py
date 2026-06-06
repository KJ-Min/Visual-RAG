from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "Data"
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "VisRAG"

DATASETS = (
    "ArxivQA",
    "ChartQA",
    "MP-DocVQA",
    "InfoVQA",
    "PlotQA",
    "SlideVQA",
)

HF_DATASET_TEMPLATE = "openbmb/VisRAG-Ret-Test-{dataset}"

BASELINES = (
    "lvlm_only",
    "visrag_image",
    "naive_text",
    "image_and_its_text",
    "text_and_its_image",
    "text_and_image",
)

NO_RETRIEVAL_BASELINES = {
    "lvlm_only",
}

IMAGE_INDEX_BASELINES = {
    "visrag_image",
    "image_and_its_text",
}

TEXT_INDEX_BASELINES = {
    "naive_text",
    "text_and_its_image",
}

DUAL_INDEX_BASELINES = {
    "text_and_image",
}

DEFAULT_RETRIEVER_MODEL = "openbmb/VisRAG-Ret"
DEFAULT_GENERATOR_MODEL = "openbmb/MiniCPM-V-2_6"
DEFAULT_QUERY_INSTRUCTION = "Represent this query for retrieving relevant documents: "
